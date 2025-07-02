from typing import Any, Dict, Optional, Union, Tuple
import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco_playground import State
from hierarchical_control.envs.hierarchical_env import HierarchicalEnv, LLSupervisedData
from mujoco_playground._src import mjx_env


def default_config() -> config_dict.ConfigDict:
  env_config = config_dict.create(
    ctrl_dt=0.008,
    sim_dt=0.004,
    episode_length=2000,
    history_len=1,  # TODO
    noise_config=config_dict.create(
      reset_noise_scale=1e-3,
    ),
    reward_config=config_dict.create(
      angle_reward_weight=1,
      angle_reward_scale=5,
      ctrl_cost_weight=0.5,
      ctrl_cost_scale=0.001,
    ),
    pd_gains=config_dict.create(
      kp=4,
      kd=0
    )
  )

  rl_config = config_dict.create(
    num_timesteps=200_000_000,
    num_evals=10,
    episode_length=env_config.episode_length,
    hl_ppo_learning_config=config_dict.create(
      reward_scaling=1.0,
      clipping_epsilon=0.1,
      learning_rate=1e-6,
      entropy_cost=0.001,
      discounting=0.99,
      gae_lambda=0.95,
      max_grad_norm=1.0,
    ),
    normalize_observations=True,
    unroll_length=32,
    num_minibatches=64,
    num_updates_per_batch=1,
    num_resets_per_eval=1,
    num_envs=8192//4,
    batch_size=256//4,
    hl_network_factory=config_dict.create(
      policy_hidden_layer_sizes=(128, 64, 64, 64),
      value_hidden_layer_sizes=(128, 64, 64, 64),
      policy_obs_key="hl_obs",
      value_obs_key="hl_obs",
    ),
    ll_network_factory=config_dict.create(
      hidden_layer_sizes=(128, 64, 64),
      obs_key="ll_obs",
    ),
    ll_learning_config=config_dict.create(
      ll_opt_max_grad_norm=None,
      learning_rate=1e-5
    )
  )

  env_config["rl_config"] = rl_config
  return env_config


class MjxHand(HierarchicalEnv):
  """Hierarchical hand environment with internal PD + HL modulation."""

  FINGER_JOINTS = {"thumb": ['mp', 'ip'],
                   "index": ['2'],
                   "middle": ['3'],
                   "ring": ['4'],
                   "little": ['5']
                   }

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
      reference_trajectory: Optional[Tuple[jp.ndarray, jp.ndarray]] = None,  # Allow passing trajectory
      collisions_enabled: bool = False
  ) -> None:
    xml_path = rf"../assets/myohand_pose.xml"
    self.collisions_enabled = collisions_enabled
    self.flexion_joint_ids = {}
    self.flat_ref_ids = ()
    super().__init__(config, config_overrides, xml_path)

    # --- Reference Trajectory ---
    if reference_trajectory:
      self._qpos_ref, self._qvel_ref = reference_trajectory
    else:
      self._qpos_ref, self._qvel_ref = self._generate_normalized_trajectory(ctrl_dt=self.sim_dt, order=jp.arange(5) + 1)
      self._idx = jp.array([int(np.where([i in v for v in self.flexion_joint_ids.values()])[0][0]) for i in self.flat_ref_ids])
      self._scales = np.concat([[jp.diff(self.mjx_model.jnt_range[i])[0] / 2 for i in v]
                                 for v in self.flexion_joint_ids.values()])
      self._scale = jp.array(self._scales)
      self._offsets = np.concat([[jp.sum(self.mjx_model.jnt_range[i]) / 2 for i in v]
                           for v in self.flexion_joint_ids.values()])
      self._offsets = jp.array(self._offsets)

    self._ref_traj_len = self._qpos_ref.shape[0]
    self.kp = self._config.pd_gains.kp
    self.kd = self._config.pd_gains.kd

  def preprocess_spec(self, spec: mujoco.MjSpec):
    spec = super().preprocess_spec(spec)
    for s in spec.sites:
      if "_target" in s.name:
        print(f"Deleted target site \"{s.name}\"")
        s.delete()
    for t in spec.tendons:
      if "_err" in t.name:
        print(f"Deleted error tendon \"{t.name}\"")
        t.delete()
    # TODO: Verify visual geoms impact performance
    for g in spec.geoms:
      if not g.name or "floor" in g.name:
        print(f"Deleted visual geom")
        g.delete()
      elif not self.collisions_enabled:
        g.conaffinity = 0
        g.contype = 0
    flexion_joints = [j for j in spec.joints if "flexion" in j.name]
    temp_model = spec.compile()
    finger_joints = {f: [temp_model.joint(j.name).id for j in flexion_joints if any([l in j.name for l in labels])]
                     for f, labels in MjxHand.FINGER_JOINTS.items()}
    self.flexion_joint_ids = finger_joints
    self.flat_ref_ids = np.concatenate(list(finger_joints.values()))

    spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    spec.option.iterations = 4
    spec.option.ls_iterations = 8
    spec.option.disableflags = spec.option.disableflags | mjx.DisableBit.EULERDAMP

    return spec

  def _generate_normalized_trajectory(self,
                                      ctrl_dt=0.001,
                                      n_freqs=5,
                                      base_freq=0.02,
                                      amplitudes=(0.8, 0.8, 0.8, 0.8, 0.8),
                                      offsets=(0,),
                                      order=None) -> Tuple[jp.ndarray, jp.ndarray]:
    frequencies = (jp.arange(n_freqs) + 1 if order is None else order) * base_freq
    offsets = jp.array(offsets)
    amplitudes = jp.array(amplitudes)
    # TODO: scale by joint ranges
    times = jp.arange(1 / base_freq / ctrl_dt, dtype=jp.uint16) * ctrl_dt  # Time based on control dt
    qpos_ref = (amplitudes[None, :]* jp.sin(2 * jp.pi * frequencies[None, :] * times[:, None]) + offsets)
    qvel_ref = (amplitudes[None, :] * 2 * jp.pi * frequencies[None, :]
                * jp.cos(2 * jp.pi * frequencies[None, :] * times[:, None]))
    return qpos_ref, qvel_ref

  def reset(self, rng: jp.ndarray) -> State:
    """Resets the environment to an initial state."""
    rng_noise, rng_vel, rng_offset, rng_permute = jax.random.split(rng, 4)

    # Initial reference trajectory time index
    ref_time_idx = jax.random.randint(rng_offset, shape=(1,),
                                      minval=0, maxval=self._ref_traj_len, dtype=jp.uint16)[0]

    ref_random_idx = jax.random.permutation(rng_permute, jp.arange(5))  # TODO: This is 1DoF joints specific

    # Initial info dictionary
    info = {
      'rng': rng,
      'ref_time_idx': ref_time_idx,
      'ref_random_idx': ref_random_idx,

      # Placeholders for fields for the LL loss calculation
      'desired_torque': jp.zeros(self.mjx_model.nv),
      'actual_torque': jp.zeros(self.mjx_model.nv),
      'jac_torque_act': jp.zeros((self.mjx_model.nv, self.mjx_model.na))
    }

    # Initial position/velocity noise
    low, hi = -self._config.noise_config.reset_noise_scale, self._config.noise_config.reset_noise_scale
    qpos_noise = jax.random.uniform(rng_noise, (self.mjx_model.nq,), minval=low, maxval=hi)

    qvel_noise = jax.random.uniform(rng_vel, (self.mjx_model.nv,), minval=low, maxval=hi)
    qpos, qvel = self.get_ref_qpos_qvel(info)  # Start
    qpos += qpos_noise
    qvel += qvel_noise


    # Initial data state
    data = mjx.make_data(self.mjx_model)
    data = data.replace(qpos=qpos, qvel=qvel, ctrl=jp.zeros(self.mjx_model.nu),
                        act=jp.zeros(self.mjx_model.na))  # Reset activation
    data = mjx.forward(self.mjx_model, data)  # Compute initial derived quantities



    # Get initial hierarchical observations
    obs = {'hl_obs': self._get_hl_obs(data, info),
           'll_obs': self._get_ll_obs(data, info)
           }

    # Initial reward, done, metrics
    reward, done, zero = jp.zeros(3)
    metrics = {
      'angle_reward': zero,
      'reward_quadctrl': zero
    }

    return State(data, obs, reward, done, metrics, info)

  def hl_step(self, state: State, hl_action: jp.ndarray) -> State:
    """Calculate desired torque based on HL modulation of PD error."""
    # hl_action is the HL_modulation signal
    data = state.data

    # ref_time was incremented after the last physics step
    # Calculate PD errors for the *next* timestep (used in next high_level_step)
    qpos_ref_t, qvel_ref_t = self.get_ref_qpos_qvel(state.info)

    # TODO: handle potential ball joints with quat_sub instead
    raw_pos_error = qpos_ref_t - data.qpos
    raw_vel_error = qvel_ref_t - data.qvel

    # Modulate position error
    modulated_pos_error = raw_pos_error + hl_action

    # Calculate desired torque
    desired_torque = self.kp * modulated_pos_error + self.kd * raw_vel_error

    state.info['desired_torque'] = desired_torque

    # Prepare observations for the LL policy
    ll_obs = self._get_ll_obs(data, state.info)
    state.obs['ll_obs'] = ll_obs
    return state

  def step(self, state: State, action: jp.ndarray) -> State:
    """Apply LL ctrl action, steps physics, calculates rewards and next state."""
    data = state.data
    data = data.replace(act=action)  # Having activation dynamics or not?

    next_data = mjx_env.step(self.mjx_model, data, action, self.n_substeps)

    # Calculate Jacobian d(torque)/d(act) based on the state
    jac_torque_act = self.calculate_torque_activation_jacobian(data)
    actual_torque = next_data.qfrc_actuator  # Torque resulting from LL ctrl

    # --- Calculate HL Reward ---
    # TODO axis for sum?
    ctrl_cost = (self._config.reward_config.ctrl_cost_weight
                 * jp.sum(jp.square(self._config.reward_config.ctrl_cost_scale * state.info['desired_torque']))
                 / self.mjx_model.nv
                 )

    state.info['ref_time_idx'] = (state.info['ref_time_idx'] + 1) % self._ref_traj_len
    qpos_ref_t = self.get_ref_qpos(state.info)

    angle_error = qpos_ref_t - next_data.qpos
    angle_reward = (self._config.reward_config.angle_reward_weight
                    * jp.exp(-self._config.reward_config.angle_reward_scale
                             * jp.sum(jp.square(angle_error)) / self.mjx_model.nv))
    reward = angle_reward - ctrl_cost

    # Prepare next observations for HL controller
    next_info_for_obs = {'ref_qpos': qpos_ref_t}
    state.obs['hl_obs'] = self._get_hl_obs(next_data, next_info_for_obs)

    # Update metrics
    state.metrics['angle_reward'] = angle_reward
    state.metrics['reward_quadctrl'] = -ctrl_cost

    # Update info dictionary for the final returned state
    state.info['actual_torque'] = actual_torque
    state.info['jac_torque_act'] = jac_torque_act

    # TODO: termination conditions
    done = 0.0

    return state.replace(data=next_data, reward=reward, done=done)

  def _get_hl_obs(self, data: mjx.Data, info: Dict) -> jp.ndarray:
    """Get observations for the high-level policy."""
    ref_qpos = info.get('ref_qpos', jp.zeros_like(data.qpos))  # Get ref info if available
    ref_qvel = info.get('ref_qvel', jp.zeros_like(data.qvel))
    return jp.concatenate([
      data.qpos,
      data.qvel,
      ref_qpos,  # Include reference state
      ref_qvel
    ])

  def _get_ll_obs(self, data, info):
    return jp.concatenate([self._get_ll_obs_base(data), info['desired_torque']])

  def _get_ll_obs_base(self, data: mjx.Data) -> jp.ndarray:
    """Get base state observations for the low-level policy (before desired_torque)."""
    # Example: proprioceptive info relevant for muscle control
    return jp.concatenate([
      data.qpos,
      data.qvel,
    ])

  def get_ref_qpos_qvel(self, info):
    t_id = info['ref_time_idx']
    qpos_finger = self._qpos_ref[t_id, info['ref_random_idx']][self._idx] * self._scales + self._offsets
    qvel_finger = self._qvel_ref[t_id, info['ref_random_idx']][self._idx] * self._scales
    qpos_ref = jp.zeros(self.mjx_model.nq).at[self.flat_ref_ids].set(qpos_finger)
    qvel_ref = jp.zeros(self.mjx_model.nv).at[self.flat_ref_ids].set(qvel_finger)
    return qpos_ref, qvel_ref

  def get_ref_qpos(self, info):
    t_id = info['ref_time_idx']
    qpos_finger = self._qpos_ref[t_id, info['ref_random_idx']][self._idx] * self._scales + self._offsets
    qpos_ref = jp.zeros(self.mjx_model.nq).at[self.flat_ref_ids].set(qpos_finger)
    return qpos_ref


  def calculate_torque_activation_jacobian(self, data: mjx.Data) -> jp.ndarray:
    """Calculates d(torque)/d(act) analytically."""
    gains = mjx._src.scan.flat(
      self.mjx_model,
      mjx._src.support.muscle_gain,
      'uuuuu',
      'u',
      data.actuator_length,
      data.actuator_velocity,
      jp.array(self.mjx_model.actuator_lengthrange),
      jp.array(self.mjx_model.actuator_acc0),
      self.mjx_model.actuator_gainprm,
      group_by='u',
    )
    return gains[None, :] * data.actuator_moment.T

  def autodiff_torque_activation_jacobian(self, data: mjx.Data) -> jp.ndarray:
    """Calculates d(torque)/d(act) using JAX's automatic differentiation."""
    def _torque_from_act(act: jp.ndarray) -> jp.ndarray:
      data_with_new_act = data.replace(act=act)
      updated_data = mjx.fwd_actuation(self.mjx_model, data_with_new_act)
      return updated_data.qfrc_actuator
    jacobian_fn = jax.jacobian(_torque_from_act)
    return jacobian_fn(data.act)

  # --- Properties ---
  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    """Returns the size of the low-level action space (ctrl)."""
    return self._mjx_model.nu

  @property
  def hl_action_size(self) -> int:
    """Returns the size of the high-level action space (modulation)."""
    return self.mjx_model.nv

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model


if __name__ == '__main__':
  pass
