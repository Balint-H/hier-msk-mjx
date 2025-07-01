from typing import Dict, Tuple

import jax
import mujoco
import mujoco.viewer as viewer
import numpy as np
from brax.training.acme.running_statistics import normalize
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model
from functools import partial
import jax.numpy as jp
from mujoco.mjx._src import sensor
from ml_collections.config_dict import config_dict

from mujoco import mjx

from hierarchical_env import make_ll_network
from playground_hand_hierarchical import MjxHand, default_config


def _get_hl_obs(data: mjx.Data, info: Dict) -> jp.ndarray:
  """Get observations for the high-level policy."""
  ref_qpos = info['ref_qpos']
  ref_qvel = info['ref_qvel']
  return jp.concatenate([
    data.qpos,
    data.qvel,
    ref_qpos,  # Include reference state
    ref_qvel
  ])


def _get_ll_obs(data, info):
  return jp.concatenate([_get_ll_obs_base(data), info['desired_torque']])


def _get_ll_obs_base(data: mjx.Data) -> jp.ndarray:
  """Get base state observations for the low-level policy (before desired_torque)."""
  # Example: proprioceptive info relevant for muscle control
  return jp.concatenate([
    data.ctrl[:-1],
    data.actuator_length[:-1],
    data.actuator_velocity[:-1],
  ])

ppo_network = ppo_networks.make_ppo_networks(
      92,
      23,
       policy_hidden_layer_sizes=[128, 64, 64, 64],
      preprocess_observations_fn=normalize,
      policy_obs_key='hl_obs')


ll_network = make_ll_network(
      39,
      140,
      hidden_layer_sizes=[128, 64, 64],
      preprocess_observations_fn=normalize,
      obs_key='ll_obs')

model_path = './playground_params.pickle'
hl_params, ll_params = model.load_params(model_path)
del hl_params[0].mean['ll_obs']
del hl_params[0].std['ll_obs']
del hl_params[0].summed_variance['ll_obs']
del ll_params[0].mean['hl_obs']
del ll_params[0].std['hl_obs']
del ll_params[0].summed_variance['hl_obs']
def deterministic_hl_policy (input_data):
    logits = ppo_network.policy_network.apply(*hl_params[:2], input_data)
    brax_result = ppo_network.parametric_action_distribution.mode(logits)
    return brax_result

def deterministic_ll_policy (input_data):
    logits = ll_network.apply(*ll_params, input_data)
    return logits


def main():
    env = MjxHand(config=default_config())

    m, d = get_mj_model_data(env)

    # Note: the first two steps will be performed much slower as the function is being jitted.
    jit_reset = jax.jit(env.reset)
    jit_hl_step = jax.jit(env.hl_step)
    jit_step = jax.jit(env.step)
    key = jax.random.PRNGKey(0)
    state = jit_reset(key)
    i = 0
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
          hl_key, ll_key, key = jax.random.split(key, 3)
          hl_actions = deterministic_hl_policy({'hl_obs': state.obs['hl_obs']})
          mid_state = jit_hl_step(state, hl_actions)
          actions = deterministic_ll_policy({'ll_obs': mid_state.obs['ll_obs']})
          print(actions)
          # mid_state = mid_state.replace(data=mid_state.data.replace(qfrc_applied=mid_state.info['desired_torque']))
          state = jit_step(mid_state, actions)

          state = state.replace(data=state.data.replace(mocap_pos=d.mocap_pos,
                                                        xfrc_applied=d.xfrc_applied))
          d.ctrl = state.data.ctrl
          d.qpos = state.data.qpos
          print(state.data.time)
          #d.qpos = env._qpos_ref[i, :]
          mujoco.mj_forward(m, d)  # We only do forward step, no need to integrate.

          # We'll use the sensordata array to visualize key values as real-time bar-graphs (use F4 in viewer)
          d.sensordata[0] = state.reward

          # Pick up changes to the physics state, apply perturbations, update options from GUI.
          viewer.sync()
    pass


def hl_only():
  env = MjxHand(config=default_config())

  m, d = get_mj_model_data(env)

  # Note: the first two steps will be performed much slower as the function is being jitted.
  jit_reset = jax.jit(env.reset)
  jit_hl_step = jax.jit(env.hl_step)
  jit_step = jax.jit(step_with_torques)
  key = jax.random.PRNGKey(0)
  state = jit_reset(key)
  with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
      hl_key, ll_key, key = jax.random.split(key, 3)
      hl_actions = deterministic_hl_policy({'hl_obs': state.obs['hl_obs']})
      state = jit_hl_step(state, jp.zeros_like(hl_actions))

      state = state.replace(data=state.data.replace(mocap_pos=d.mocap_pos,
                                                    xfrc_applied=d.xfrc_applied))
      print(state.info['desired_torque'])
      nd = jit_step(env.mjx_model, state.data, state.info['desired_torque'])
      state = state.replace(data=nd)
      state.info['ref_time_idx'] = (state.info['ref_time_idx'] + 1) % env._ref_traj_len
      next_info_for_obs = {'ref_qpos': env._qpos_ref[(state.info['ref_time_idx'] + 1) % env._ref_traj_len]}
      state.obs['hl_obs'] = env._get_hl_obs(state.data, next_info_for_obs)

      d.qpos = state.data.qpos
      # d.qpos = env._qpos_ref[i, :]
      mujoco.mj_forward(m, d)  # We only do forward step, no need to integrate.

      # We'll use the sensordata array to visualize key values as real-time bar-graphs (use F4 in viewer)
      d.sensordata[0] = state.reward

      # Pick up changes to the physics state, apply perturbations, update options from GUI.
      viewer.sync()
  pass


def step_with_torques(m: mjx.Model, d: mjx.Data, qfrc_applied) -> mjx.Data:
  """Advance simulation."""
  d = mjx.fwd_position(m, d)
  d = mjx.fwd_velocity(m, d)
  d = mjx.fwd_actuation(m, d)

  d = d.replace(qfrc_actuator = jp.zeros_like(d.qfrc_actuator))
  d = d.replace(qfrc_applied = qfrc_applied)
  d = mjx.fwd_acceleration(m, d)
  if m.opt.integrator == mjx.IntegratorType.EULER:
    d = mjx.euler(m, d)
  elif m.opt.integrator == mjx.IntegratorType.RK4:
    d = mjx.rungekutta4(m, d)
  elif m.opt.integrator == mjx.IntegratorType.IMPLICITFAST:
    d = mjx.implicit(m, d)
  else:
    raise NotImplementedError(f'integrator {m.opt.integrator} not implemented.')

  return d

def get_mj_model_data(env):
    # We could get the model from the env, but we want to make some edits for convenience
    spec = mujoco.MjSpec.from_file(env.xml_path)
    spec = env.preprocess_spec(spec)
    # Add in dummy sensor we can write to later to visualize values
    spec.add_sensor(name="reward+target", type=mujoco.mjtSensor.mjSENS_USER, dim=1 + env.mjx_model.nv)


    m = spec.compile()
    d = mujoco.MjData(m)
    return m, d


if __name__ == '__main__':
    main()
