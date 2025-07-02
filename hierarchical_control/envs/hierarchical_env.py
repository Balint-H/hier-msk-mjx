from brax.training import networks, types
from brax.training.networks import ActivationFn, Initializer, FeedForwardNetwork, MLP, _get_obs_state_size
from brax.training.types import Params, PRNGKey
from flax import struct, linen
from datetime import datetime
from typing import Any, Dict, Optional, Union, Callable, Tuple, NamedTuple, Sequence
from brax.envs.wrappers import training as brax_training
from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco_playground import State
import abc
from mujoco_playground._src import mjx_env
from mujoco_playground._src.wrapper import Wrapper, BraxDomainRandomizationVmapWrapper, BraxAutoResetWrapper


class HierarchicalEnv(mjx_env.MjxEnv, abc.ABC):

  def __init__(self, config, config_overrides, xml_path=None):
    super().__init__(config, config_overrides)
    if xml_path is not None:
      spec = mujoco.MjSpec.from_file(xml_path)
      spec = self.preprocess_spec(spec)

      self._mj_model = spec.compile()
      self._mj_model.opt.timestep = self.sim_dt

      self._mjx_model = mjx.put_model(self._mj_model)
      self._xml_path = xml_path

      self._mjx_model = mjx.put_model(self._mj_model)

  def preprocess_spec(self, spec: mujoco.MjSpec) -> mujoco.MjSpec:
    return spec

  @abc.abstractmethod
  def hl_step(self, state: State, action: jax.Array) -> State:
    """Run high-level control, calculate input to low-level systems. Don't run dynamics yet."""

  @property
  @abc.abstractmethod
  def hl_action_size(self) -> int:
    """Returns the size of the high-level action space (e.g., target angle)."""


class LLSupervisedData(NamedTuple):
  """Data collected for training the low-level supervised policy."""
  ll_obs: Dict[str, jp.ndarray]
  activation_designated: jp.ndarray  # Could be different from logits e.g., stochastic
  hl_desired_torque: jp.ndarray
  # Pre-computed Jacobian: d(torque)/d(act)
  jacobian: Optional[jp.ndarray] = None
  torque_designated: Optional[jp.ndarray] = None
  qpos: Optional[jp.ndarray] = None
  qvel: Optional[jp.ndarray] = None


class HierarchicalBraxDomainRandomizationVmapWrapper(BraxDomainRandomizationVmapWrapper):
  """Brax wrapper for domain randomization of hierarchical environment."""

  def __init__(
      self,
      env: HierarchicalEnv,
      randomization_fn: Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]],
  ):

    super().__init__(env, randomization_fn)

  def _env_fn(self, mjx_model: mjx.Model) -> HierarchicalEnv:
    env = self.env
    env.unwrapped._mjx_model = mjx_model
    return env

  def hl_step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    def _hl_step(mjx_model, s, a):
      env = self._env_fn(mjx_model=mjx_model)
      return env.hl_step(s, a)

    res = jax.vmap(_hl_step, in_axes=[self._in_axes, 0, 0])(
        self._mjx_model_v, state, action
    )
    return res

  @property
  def hl_action_size(self) -> int:
    return self.env.hl_action_size


class HierarchicalVmapWrapper(brax_training.VmapWrapper):
  """Vectorizes q hierarchical Brax env."""

  def __init__(self, env: HierarchicalEnv, batch_size: Optional[int] = None):
    self.env: HierarchicalEnv = env
    super().__init__(env, batch_size)

  def hl_step(self, state: State, action: jax.Array) -> State:
    return jax.vmap(self.env.hl_step)(state, action)

  @property
  def hl_action_size(self) -> int:
    return self.env.hl_action_size


class HierarchicalEpisodeWrapper(brax_training.EpisodeWrapper):

  def __init__(self, env: HierarchicalEnv, episode_length: int, action_repeat: int):
    self.env: HierarchicalEnv = env
    super().__init__(env, episode_length, action_repeat)

  def hl_step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    return self.env.hl_step(state, action)

  @property
  def hl_action_size(self) -> int:
    return self.env.hl_action_size


class HierarchicalBraxAutoResetWrapper(BraxAutoResetWrapper):
  def hl_step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    return self.env.hl_step(state, action)

  @property
  def hl_action_size(self) -> int:
    return self.env.hl_action_size


def wrap_for_hierarchical_brax_training(
    env: mjx_env.MjxEnv,
    num_vision_envs: int = 1,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[
        Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]]
    ] = None,
) -> Wrapper:
  """Common wrapper pattern for all brax training agents.

  Args:
    env: environment to be wrapped
    vision: whether the environment will be vision based
    num_vision_envs: number of environments the renderer should generate,
      should equal the number of batched envs
    episode_length: length of episode
    action_repeat: how many repeated actions to take per step
    randomization_fn: randomization function that produces a vectorized model
      and in_axes to vmap over

  Returns:
    An environment that is wrapped with Episode and AutoReset wrappers.  If the
    environment did not already have batch dimensions, it is additional Vmap
    wrapped.
  """
  if randomization_fn is None:
    env = HierarchicalVmapWrapper(env)  # pytype: disable=wrong-arg-types
  else:
    env = HierarchicalBraxDomainRandomizationVmapWrapper(env, randomization_fn)
  env = HierarchicalEpisodeWrapper(env, episode_length, action_repeat)
  env = HierarchicalBraxAutoResetWrapper(env)
  return env


def wrap_for_hierarchical_brax_debug(
    env: mjx_env.MjxEnv,
    num_vision_envs: int = 1,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[
        Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]]
    ] = None,
) -> Wrapper:
  env = HierarchicalEpisodeWrapper(env, episode_length, action_repeat)
  env = HierarchicalBraxAutoResetWrapper(env)
  return env


def make_ll_inference_fn(network: networks.FeedForwardNetwork):
  # TODO: Should we keep format to support stochastic inference?
  def make_ll_policy(
      params: Params, deterministic: bool = True
  ) -> types.Policy:

    def ll_policy(
        observations: types.Observation, key_sample: PRNGKey
    ) -> Tuple[types.Action, types.Extra]:
      param_subset = (params[0], params[1])  # normalizer and policy params
      logits = network.apply(*param_subset, observations)

      return logits, {}

    return ll_policy

  return make_ll_policy



def make_ll_network(
    param_size: int,
    obs_size: types.ObservationSize,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
    layer_norm: bool = False,
    obs_key: str = 'state',
) -> FeedForwardNetwork:
  """Creates a policy network with 0-1 outputs with a sigmoid activation."""
  policy_module = MLP(
      layer_sizes=list(hidden_layer_sizes) + [param_size],
      activation=activation,
      kernel_init=kernel_init,
      layer_norm=layer_norm,
  )

  def apply(processor_params, policy_params, obs):
    obs = preprocess_observations_fn(obs, processor_params)
    obs = obs if isinstance(obs, jax.Array) else obs[obs_key]
    logits = policy_module.apply(policy_params, obs)
    return linen.sigmoid(logits)

  obs_size = _get_obs_state_size(obs_size, obs_key)
  dummy_obs = jp.zeros((1, obs_size))
  return FeedForwardNetwork(
      init=lambda key: policy_module.init(key, dummy_obs), apply=apply
  )
