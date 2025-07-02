import functools
from typing import Any, Tuple

import jax
import jax.numpy as jnp
from brax.training.types import Params, Metrics
# Assume these are correctly imported
from hierarchical_control.learning.train_hierarchical import LLSupervisedData
from brax.training.networks import FeedForwardNetwork
from mujoco import mjx
# Import the necessary sub-modules from your provided files
from mujoco.mjx._src import forward as mjx_forward



def hierarchical_ll_loss_l2(
    params: Params,
    normalizer_params: Any,
    data: LLSupervisedData, # Assumes this now contains qpos, qvel, ctrl, desired_torque
    network: FeedForwardNetwork,
    model: mjx.Model,
    l2_reg
) -> Tuple[jnp.ndarray, Metrics]:

    flat_obs = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1, x.shape[-1])), data.ll_obs)
    logits = network.apply(normalizer_params, params, flat_obs)
    l2_loss_logits = 0.5 * jnp.sum(logits * logits) * l2_reg

    data_template = mjx.make_data(model)
    batch_loss_fn = jax.vmap(
        _compute_torque_loss_single,
        in_axes=(0, 0, 0, 0, None, None)
    )

    flat_data = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1, *x.shape[2:])), data)

    losses = batch_loss_fn(
        logits,
        flat_data.qpos,
        flat_data.qvel,
        flat_data.hl_desired_torque,
        model,
        data_template
    )

    mean_torque_loss = jnp.mean(losses)

    metrics = {
      'torque_loss': mean_torque_loss,
      'l2_loss': l2_loss_logits
    }

    return mean_torque_loss+l2_loss_logits, metrics


def hierarchical_ll_loss(
    params: Params,
    normalizer_params: Any,
    data: LLSupervisedData, # Assumes this now contains qpos, qvel, ctrl, desired_torque
    network: FeedForwardNetwork,
    model: mjx.Model,
) -> Tuple[jnp.ndarray, Metrics]:

    flat_obs = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1, x.shape[-1])), data.ll_obs)
    logits = network.apply(normalizer_params, params, flat_obs)

    data_template = mjx.make_data(model)
    batch_loss_fn = jax.vmap(
        _compute_torque_loss_single,
        in_axes=(0, 0, 0, 0, None, None)
    )

    flat_data = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1, *x.shape[2:])), data)

    losses = batch_loss_fn(
        logits,
        flat_data.qpos,
        flat_data.qvel,
        flat_data.hl_desired_torque,
        model,
        data_template
    )

    mean_torque_loss = jnp.mean(losses)

    metrics = {
      'torque_loss': mean_torque_loss
    }

    return mean_torque_loss, metrics

def _compute_torque_loss_single(
    act: jnp.ndarray,
    qpos: jnp.ndarray,
    qvel: jnp.ndarray,
    desired_torque: jnp.ndarray,
    model: mjx.Model,
    data_template: mjx.Data,
):
    """Computes torque loss for one item, reconstructing state as needed."""
    d = data_template.replace(qpos=qpos, qvel=qvel, act=act, ctrl=act)
    updated_data = mjx_forward.forward(model, d)
    actual_torque = updated_data.qfrc_actuator

    torque_error = actual_torque - desired_torque
    loss = 0.5 * jnp.sum(jnp.square(torque_error))

    return loss
