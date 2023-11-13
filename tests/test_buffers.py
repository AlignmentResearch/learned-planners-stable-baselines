import gymnasium as gym
import numpy as np
import pytest
import torch as th
from gymnasium import spaces

from stable_baselines3.common.buffers import (
    DictReplayBuffer,
    DictRolloutBuffer,
    ReplayBuffer,
    RolloutBuffer,
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.pytree_dataclass import tree_flatten
from stable_baselines3.common.recurrent.buffers import RecurrentRolloutBuffer
from stable_baselines3.common.recurrent.type_aliases import RecurrentRolloutBufferData
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    ReplayBufferSamples,
)
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize

EP_LENGTH: int = 100


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
        self._ep_length = EP_LENGTH

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
        self._ep_length = EP_LENGTH

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

    buffer = replay_buffer_cls(EP_LENGTH, env.observation_space, env.action_space, device="cpu")

    # Interract and store transitions
    env.reset()
    obs = env.get_original_obs()
    for _ in range(EP_LENGTH):
        action = th.as_tensor(env.action_space.sample())
        _, _, done, info = env.step(action)
        next_obs = env.get_original_obs()
        reward = env.get_original_reward()
        buffer.add(obs, next_obs, action, reward, done, info)
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


HIDDEN_STATES_EXAMPLE = {"a": {"b": th.zeros(2, 4)}}


@pytest.mark.parametrize(
    "replay_buffer_cls", [DictReplayBuffer, DictRolloutBuffer, ReplayBuffer, RolloutBuffer, RecurrentRolloutBuffer]
)
@pytest.mark.parametrize("device", ["cpu", "cuda", "auto"])
def test_device_buffer(replay_buffer_cls, device):
    if device == "cuda" and not th.cuda.is_available():
        pytest.skip("CUDA not available")

    env = {
        RolloutBuffer: DummyEnv,
        DictRolloutBuffer: DummyDictEnv,
        ReplayBuffer: DummyEnv,
        DictReplayBuffer: DummyDictEnv,
        RecurrentRolloutBuffer: DummyDictEnv,
    }[replay_buffer_cls]
    env = make_vec_env(env)
    hidden_states_shape = HIDDEN_STATES_EXAMPLE["a"]["b"].shape
    N_ENVS_HIDDEN_STATES = {"a": {"b": th.zeros((hidden_states_shape[0], env.num_envs, *hidden_states_shape[1:]))}}

    if replay_buffer_cls == RecurrentRolloutBuffer:
        buffer = RecurrentRolloutBuffer(
            EP_LENGTH, env.observation_space, env.action_space, hidden_state_example=N_ENVS_HIDDEN_STATES, device=device
        )
    else:
        buffer = replay_buffer_cls(EP_LENGTH, env.observation_space, env.action_space, device=device)

    # Interract and store transitions
    obs = env.reset()
    for _ in range(EP_LENGTH):
        action = th.as_tensor(env.action_space.sample())

        next_obs, reward, done, info = env.step(action)
        if replay_buffer_cls in [RolloutBuffer, DictRolloutBuffer]:
            episode_start, values, log_prob = th.zeros(1), th.zeros(1), th.ones(1)
            buffer.add(obs, action, reward, episode_start, values, log_prob)
        elif replay_buffer_cls == RecurrentRolloutBuffer:
            episode_start, values, log_prob = th.zeros(1), th.zeros(1), th.ones(1)
            buffer.add(RecurrentRolloutBufferData(obs, action, reward, episode_start, values, log_prob, N_ENVS_HIDDEN_STATES))
        else:
            buffer.add(obs, next_obs, action, reward, done, info)
        obs = next_obs

    # Get data from the buffer
    if replay_buffer_cls in [RolloutBuffer, DictRolloutBuffer]:
        data = buffer.get(50)
    elif replay_buffer_cls in [ReplayBuffer, DictReplayBuffer]:
        data = [buffer.sample(50)]
    elif replay_buffer_cls == RecurrentRolloutBuffer:
        data = buffer.get(EP_LENGTH)

    # Check that all data are on the desired device
    desired_device = get_device(device).type
    for minibatch in list(data):
        flattened_tensors, _ = tree_flatten(minibatch)
        assert len(flattened_tensors) > 3
        for value in flattened_tensors:
            assert isinstance(value, th.Tensor)
            assert value.device.type == desired_device
