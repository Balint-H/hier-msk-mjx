# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Standalone script to train a low-level network with supervised learning."""

import time
import functools
from datetime import datetime
from typing import Any, Callable, Tuple, NamedTuple

import jax
import jax.numpy as jnp
import optax
import numpy as np
from absl import app
from absl import flags
from absl import logging
from ml_collections import config_dict
from etils import epath
from flax.training import orbax_utils
from flax import struct
import orbax.checkpoint as ocp
from tensorboardX import SummaryWriter
from brax import envs  # Import brax envs

# Assuming the following files are in the same directory or accessible in the path
from hierarchical_control.envs.playground_hand_hierarchical import MjxHand, default_config
from hierarchical_control.envs.hierarchical_env import make_ll_network
from brax.training.types import Params, Metrics
from brax.training.networks import FeedForwardNetwork
from brax.training.acme import running_statistics, specs
from mujoco import mjx

# Set verbosity for cleaner output
logging.set_verbosity(logging.WARNING)

# Define command-line flags for configuration
_SAVE_PATH = flags.DEFINE_string("save_path", "/tmp/ll_supervised", "Path to save checkpoints and logs.")
_SEED = flags.DEFINE_integer("seed", 0, "Random seed.")
_NUM_TIMESTEPS = flags.DEFINE_integer("num_timesteps", 5_000_000, "Number of data generation timesteps.")
_NUM_ENVS = flags.DEFINE_integer("num_envs", 2048, "Number of parallel environments for data generation.")
_NUM_EPOCHS = flags.DEFINE_integer("num_epochs", 100, "Number of training epochs.")
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 2048, "Batch size for training.")
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 3e-4, "Learning rate for the optimizer.")
_L2_REG = flags.DEFINE_float("l2_reg", 0.005, "L2 regularization strength for network activations.")
_NORMALIZE_OBSERVATIONS = flags.DEFINE_boolean("normalize_observations", True, "Whether to normalize observations.")


class SupervisedDataset(NamedTuple):
  """Data for supervised learning of the LL policy."""
  obs: jnp.ndarray
  desired_torque: jnp.ndarray
  qpos: jnp.ndarray
  qvel: jnp.ndarray


@struct.dataclass
class TrainingState:
  """Contains training state for the learner."""
  optimizer_state: optax.OptState
  params: Params
  normalizer_params: running_statistics.RunningStatisticsState
  steps: jnp.ndarray


def get_config() -> config_dict.ConfigDict:
  """Returns the default configuration for the training script."""
  config = config_dict.ConfigDict()

  # Environment and Data Generation
  config.env_config = default_config()
  config.env_config.num_envs = _NUM_ENVS.value

  # Network Configuration
  config.network_factory = config_dict.ConfigDict()
  config.network_factory.hidden_layer_sizes = (256, 128, 64)
  config.network_factory.obs_key = 'll_obs'  # This will be created on the fly

  # Training parameters
  config.training = config_dict.ConfigDict()
  config.training.learning_rate = _LEARNING_RATE.value
  config.training.num_epochs = _NUM_EPOCHS.value
  config.training.batch_size = _BATCH_SIZE.value
  config.training.num_timesteps = _NUM_TIMESTEPS.value
  config.training.l2_reg = _L2_REG.value
  config.training.normalize_observations = _NORMALIZE_OBSERVATIONS.value
  config.training.seed = _SEED.value

  return config


def generate_dataset(env: MjxHand, num_steps: int, num_envs: int) -> SupervisedDataset:
  """Generates a dataset by stepping parallel environments."""
  logging.info(f"Generating dataset with {num_steps} total steps across {num_envs} parallel environments...")
  start_time = time.time()

  # JIT the reset and step functions for performance
  jit_reset = jax.jit(jax.vmap(env.reset))
  jit_step = jax.jit(jax.vmap(env.step))

  # Initialize environments
  rng = jax.random.PRNGKey(0)
  rng, key_reset = jax.random.split(rng)
  keys = jax.random.split(key_reset, num_envs)
  state = jit_reset(keys)

  # Data buffers
  observations = []
  desired_torques = []
  qposes = []
  qvels = []

  # Main generation loop
  for _ in range(num_steps // num_envs):
    # Get reference trajectory for the current timestep in all envs
    qpos_ref, qvel_ref = jax.vmap(env.get_ref_qpos_qvel)(state.info)

    # Calculate PD torque (HL action is zero) for all envs
    pos_error = qpos_ref - state.data.qpos
    vel_error = qvel_ref - state.data.qvel
    desired_torque = env.kp * pos_error + env.kd * vel_error

    # Store data from all envs
    # The observation for the LL network is qpos and qvel
    obs = jnp.concatenate([state.data.qpos, state.data.qvel], axis=-1)
    observations.append(obs)
    desired_torques.append(desired_torque)
    qposes.append(state.data.qpos)
    qvels.append(state.data.qvel)

    # Step all environments with dummy actions
    rng, key_action = jax.random.split(rng)
    dummy_actions = jax.random.uniform(key_action, (num_envs, env.action_size))
    state = jit_step(state, dummy_actions)

    # Reset environments that are done
    rng, key_reset = jax.random.split(rng)
    keys = jax.random.split(key_reset, num_envs)
    new_state = jit_reset(keys)
    # Only replace the state for environments that were done
    state = jax.tree_util.tree_map(
      lambda x, y: jnp.where(state.done[:, None], x, y) if x.ndim > 1 else jnp.where(state.done, x, y),
      new_state,
      state,
    )

  # Combine and reshape data
  dataset = SupervisedDataset(
    obs=jnp.reshape(jnp.array(observations), (-1, observations[0].shape[-1])),
    desired_torque=jnp.reshape(jnp.array(desired_torques), (-1, desired_torques[0].shape[-1])),
    qpos=jnp.reshape(jnp.array(qposes), (-1, qposes[0].shape[-1])),
    qvel=jnp.reshape(jnp.array(qvels), (-1, qvels[0].shape[-1])),
  )

  duration = time.time() - start_time
  logging.info(f"Dataset generation finished in {duration:.2f}s. Total samples: {dataset.obs.shape[0]}")
  return dataset


def _compute_torque_loss_single(
    act: jnp.ndarray,
    qpos: jnp.ndarray,
    qvel: jnp.ndarray,
    desired_torque: jnp.ndarray,
    model: mjx.Model,
    data_template: mjx.Data,
):
  """
  Computes torque loss for a single state-action pair.
  This function is designed to be vmapped.
  """
  # Create a new data object with the given state and action
  d = data_template.replace(qpos=qpos, qvel=qvel, act=act, ctrl=act)
  # Run forward dynamics to get the actuator forces
  updated_data = mjx.forward(model, d)
  actual_torque = updated_data.qfrc_actuator

  # Compute the squared error between actual and desired torque
  torque_error = actual_torque - desired_torque
  loss = 0.5 * jnp.sum(jnp.square(torque_error))
  return loss


def make_loss_fn(network: FeedForwardNetwork, model: mjx.Model, l2_reg: float, normalize_fn: Callable):
  """Creates the loss function for supervised training."""
  data_template = mjx.make_data(model)

  # Vectorize the single loss computation over the batch dimension
  batch_loss_fn = jax.vmap(
    _compute_torque_loss_single,
    in_axes=(0, 0, 0, 0, None, None)
  )

  def loss_fn(
      params: Params,
      normalizer_params: running_statistics.RunningStatisticsState,
      batch: SupervisedDataset,
  ) -> Tuple[jnp.ndarray, Metrics]:
    """Computes the total loss for a batch of data."""
    # Normalize observations
    normalized_obs = normalize_fn(batch.obs, normalizer_params)

    # Get network output (activations)
    activations = network.apply(params, normalized_obs)

    # Compute the primary torque-matching loss
    torque_losses = batch_loss_fn(
      activations,
      batch.qpos,
      batch.qvel,
      batch.desired_torque,
      model,
      data_template
    )
    mean_torque_loss = jnp.mean(torque_losses)

    # Add L2 regularization on the activations to encourage smaller values
    l2_loss_activations = l2_reg * 0.5 * jnp.sum(jnp.square(activations))

    total_loss = mean_torque_loss + l2_loss_activations

    metrics = {
      'total_loss': total_loss,
      'torque_loss': mean_torque_loss,
      'l2_loss_activations': l2_loss_activations,
    }
    return total_loss, metrics

  return jax.jit(loss_fn)


def main(argv):
  del argv  # Unused

  # --- Configuration and Setup ---
  config = get_config()
  logdir = epath.Path(_SAVE_PATH.value)
  timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
  logdir = logdir / timestamp
  logdir.mkdir(parents=True, exist_ok=True)
  writer = SummaryWriter(logdir)

  logging.info("Starting Standalone LL Supervised Training")
  logging.info(f"Configuration:\n{config}")
  logging.info(f"Log directory: {logdir}")

  # --- Environment and Dataset ---
  env = MjxHand(config=config.env_config)
  print("Generating dataset")
  dataset = generate_dataset(env, num_steps=config.training.num_timesteps, num_envs=config.env_config.num_envs)

  # --- Network and Optimizer ---
  # The observation is qpos and qvel concatenated
  obs_size = env.mjx_model.nq + env.mjx_model.nv
  ll_network = make_ll_network(
    param_size=env.action_size,
    obs_size=obs_size,
    **config.network_factory,
  )
  optimizer = optax.adam(learning_rate=config.training.learning_rate)

  # --- Training State Initialization ---
  key = jax.random.PRNGKey(config.training.seed)
  key_net, key_data = jax.random.split(key)

  init_params = ll_network.init(key_net)
  optimizer_state = optimizer.init(init_params)

  # Initialize normalizer
  if config.training.normalize_observations:
    normalizer_params = running_statistics.init_state_from_data(dataset.obs)
    normalize = running_statistics.normalize
  else:
    # Create dummy normalizer state and identity function
    normalizer_params = running_statistics.init_state(specs.Array((obs_size,), jnp.float32))
    normalize = lambda x, y: x

  training_state = TrainingState(
    optimizer_state=optimizer_state,
    params=init_params,
    normalizer_params=normalizer_params,
    steps=jnp.array(0)
  )

  # --- Loss and Gradient Update Function ---
  loss_fn = make_loss_fn(ll_network, env.mjx_model, config.training.l2_reg, normalize)

  @jax.jit
  def train_step(state: TrainingState, batch: SupervisedDataset) -> Tuple[TrainingState, Metrics]:
    """Performs a single gradient update step."""
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params, state.normalizer_params, batch)

    updates, new_optimizer_state = optimizer.apply_updates(grads, state.optimizer_state)
    new_params = optax.apply_updates(state.params, updates)

    new_state = state.replace(
      optimizer_state=new_optimizer_state,
      params=new_params,
      steps=state.steps + 1
    )
    return new_state, metrics

  # --- Training Loop ---
  logging.info("Starting training loop...")
  num_samples = dataset.obs.shape[0]
  steps_per_epoch = num_samples // config.training.batch_size

  for epoch in range(config.training.num_epochs):
    start_time = time.time()

    # Shuffle data for the new epoch
    key_data, perm_key = jax.random.split(key_data)
    shuffled_indices = jax.random.permutation(perm_key, num_samples)
    shuffled_dataset = jax.tree_util.tree_map(lambda x: x[shuffled_indices], dataset)

    epoch_metrics = []
    for i in range(steps_per_epoch):
      start_idx = i * config.training.batch_size
      end_idx = start_idx + config.training.batch_size
      batch = jax.tree_util.tree_map(lambda x: x[start_idx:end_idx], shuffled_dataset)

      training_state, metrics = train_step(training_state, batch)
      epoch_metrics.append(metrics)

    # Aggregate and log metrics
    epoch_metrics = jax.tree_util.tree_map(lambda *x: jnp.mean(jnp.array(x)), *epoch_metrics)
    duration = time.time() - start_time

    print(
      f"Epoch {epoch + 1}/{config.training.num_epochs} | "
      f"Loss: {epoch_metrics['total_loss']:.4f} | "
      f"Torque Loss: {epoch_metrics['torque_loss']:.4f} | "
      f"L2 Loss: {epoch_metrics['l2_loss_activations']:.4f} | "
      f"Duration: {duration:.2f}s"
    )

    for key, val in epoch_metrics.items():
      writer.add_scalar(f'training/{key}', val, epoch)
    writer.flush()

  # --- Save Final Model ---
  logging.info("Training finished. Saving final model.")
  orbax_checkpointer = ocp.PyTreeCheckpointer()
  final_params = (training_state.normalizer_params, training_state.params)
  save_args = orbax_utils.save_args_from_target(final_params)
  path = logdir / "final_model"
  orbax_checkpointer.save(path, final_params, force=True, save_args=save_args)
  logging.info(f"Model saved to {path}")


if __name__ == "__main__":
  app.run(main)
