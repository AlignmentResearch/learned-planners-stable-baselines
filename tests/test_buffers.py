import gymnasium as gym
import numpy as np
import optree as ot
import pytest
import torch as th
from gymnasium import spaces

from stable_baselines3.common.buffers import DictReplayBuffer, DictRolloutBuffer, ReplayBuffer, RolloutBuffer
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.pytree_dataclass import OT_NAMESPACE
from stable_baselines3.common.type_aliases import DictReplayBufferSamples, ReplayBufferSamples
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize


class DummyEnv(gym.Env):
    """
    Custom gym environment for testing purposes
    """

    def __init__(self):
        self.action_space = spaces.Box(1, 5, (1,))
        self.observation_space = spaces.Box(1, 5, (1,))
        self._observations = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
        self._rewards = [1, 2, 3, 4, 5]
        self._t = 0
        self._ep_length = 100

    def reset(self, *, seed=None, options=None):
        self._t = 0
        obs = self._observations[0]
        return obs, {}

    def step(self, action):
        self._t += 1
        index = self._t % len(self._observations)
        obs = self._observations[index]
        terminated = False
        truncated = self._t >= self._ep_length
        reward = self._rewards[index]
        return obs, reward, terminated, truncated, {}


class DummyDictEnv(gym.Env):
    """
    Custom gym environment for testing purposes
    """

    def __init__(self):
        # Test for multi-dim action space
        self.action_space = spaces.Box(1, 5, shape=(10, 7))
        space = spaces.Box(1, 5, (1,))
        self.observation_space = spaces.Dict({"observation": space, "achieved_goal": space, "desired_goal": space})
        self._observations = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
        self._rewards = [1, 2, 3, 4, 5]
        self._t = 0
        self._ep_length = 100

    def reset(self, seed=None, options=None):
        self._t = 0
        obs = {key: self._observations[0] for key in self.observation_space.spaces.keys()}
        return obs, {}

    def step(self, action):
        self._t += 1
        index = self._t % len(self._observations)
        obs = {key: self._observations[index] for key in self.observation_space.spaces.keys()}
        terminated = False
        truncated = self._t >= self._ep_length
        reward = self._rewards[index]
        return obs, reward, terminated, truncated, {}


@pytest.mark.parametrize("env_cls", [DummyEnv, DummyDictEnv])
def test_env(env_cls):
    # Check the env used for testing
    # Do not warn for assymetric space
    check_env(env_cls(), warn=False, skip_render_check=True)


@pytest.mark.parametrize("replay_buffer_cls", [ReplayBuffer, DictReplayBuffer])
def test_replay_buffer_normalization(replay_buffer_cls):
    env = {ReplayBuffer: DummyEnv, DictReplayBuffer: DummyDictEnv}[replay_buffer_cls]
    env = make_vec_env(env)
    env = VecNormalize(env)

    buffer = replay_buffer_cls(100, env.observation_space, env.action_space, recurrent_state_example=(), device="cpu")

    # Interract and store transitions
    env.reset()
    obs = env.get_original_obs()
    for _ in range(100):
        action = th.as_tensor(env.action_space.sample())
        _, _, done, info = env.step(action)
        next_obs = env.get_original_obs()
        reward = env.get_original_reward()
        buffer.add(obs, next_obs, action, reward, done, info, recurrent_states=())
        obs = next_obs

    sample = buffer.sample(50, env)
    # Test observation normalization
    for observations in [sample.observations, sample.next_observations]:
        if isinstance(sample, DictReplayBufferSamples):
            for key in observations.keys():
                assert th.allclose(observations[key].mean(0), th.zeros(1), atol=1)
        elif isinstance(sample, ReplayBufferSamples):
            assert th.allclose(observations.mean(0), th.zeros(1), atol=1)
    # Test reward normalization
    assert np.allclose(sample.rewards.mean(0), np.zeros(1), atol=1)


@pytest.mark.parametrize("replay_buffer_cls", [DictReplayBuffer, DictRolloutBuffer, ReplayBuffer, RolloutBuffer])
@pytest.mark.parametrize("device", ["cpu", "cuda", "auto"])
def test_device_buffer(replay_buffer_cls, device):
    if device == "cuda" and not th.cuda.is_available():
        pytest.skip("CUDA not available")

    env = {
        RolloutBuffer: DummyEnv,
        DictRolloutBuffer: DummyDictEnv,
        ReplayBuffer: DummyEnv,
        DictReplayBuffer: DummyDictEnv,
    }[replay_buffer_cls]
    env = make_vec_env(env)

    if replay_buffer_cls in [RolloutBuffer, DictRolloutBuffer]:
        recurrent_state_example = {"a": {"b": th.rand((4,))}}
    else:
        recurrent_state_example = ()

    buffer = replay_buffer_cls(
        100, env.observation_space, env.action_space, recurrent_state_example=recurrent_state_example, device=device
    )

    # Interract and store transitions
    obs = env.reset()
    for _ in range(100):
        action = th.as_tensor(env.action_space.sample())

        next_obs, reward, done, info = env.step(action)
        if replay_buffer_cls in [RolloutBuffer, DictRolloutBuffer]:
            recurrent_states = {"a": {"b": th.rand((env.num_envs, 4))}}
            episode_start, values, log_prob = th.zeros(1), th.zeros(1), th.ones(1)
            buffer.add(obs, action, reward, episode_start, values, log_prob, recurrent_states=recurrent_states)
        else:
            recurrent_states = ()
            buffer.add(obs, next_obs, action, reward, done, info, recurrent_states=recurrent_states)
        obs = next_obs

    # Check that all data are on the desired device
    desired_device = get_device(device).type

    def _check_data_device(data):
        def _assert_device(value):
            assert value.device.type == desired_device

        ot.tree_map(_assert_device, data, namespace=OT_NAMESPACE)

    # Get data from the buffer
    if replay_buffer_cls in [RolloutBuffer, DictRolloutBuffer]:
        _check_data_device(list(buffer.get(50)))
        _check_data_device(list(buffer.get()))
    elif replay_buffer_cls in [ReplayBuffer, DictReplayBuffer]:
        _check_data_device(list(buffer.sample(50)))

    # Check _get_samples with slices that have None
    buffer._get_samples(slice(None))
    buffer._get_samples(slice(2))
    buffer._get_samples(slice(2, 4))
    buffer._get_samples(slice(0, 4, 2))
