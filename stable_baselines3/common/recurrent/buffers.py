import dataclasses
import logging
from functools import partial
from typing import Callable, Generator, Optional, Tuple, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.pytree_dataclass import (
    TensorTree,
    tree_flatten,
    tree_index,
    tree_map,
)
from stable_baselines3.common.recurrent.type_aliases import (
    RecurrentRolloutBufferData,
    RecurrentRolloutBufferSamples,
)
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.util import as_torch_dtype

log = logging.getLogger(__name__)


def pad(
    seq_start_indices: th.Tensor,
    seq_end_indices: th.Tensor,
    tensor: th.Tensor,
    padding_value: float = 0.0,
) -> th.Tensor:
    """
    Chunk sequences and pad them to have constant dimensions.

    :param seq_start_indices: Indices of the transitions that start a sequence
    :param seq_end_indices: Indices of the transitions that end a sequence
    :param tensor: Tensor of shape (batch_size, *tensor_shape)
    :param padding_value: Value used to pad sequence to the same length
        (zero padding by default)
    :return: (n_seq, max_length, *tensor_shape)
    """
    # Create sequences given start and end
    seq = [tensor[start : end + 1] for start, end in zip(seq_start_indices, seq_end_indices)]
    return th.nn.utils.rnn.pad_sequence(seq, batch_first=True, padding_value=padding_value)


def pad_and_flatten(
    seq_start_indices: th.Tensor,
    seq_end_indices: th.Tensor,
    tensor: th.Tensor,
    padding_value: float = 0.0,
) -> th.Tensor:
    """
    Pad and flatten the sequences of scalar values,
    while keeping the sequence order.
    From (batch_size, 1) to (n_seq, max_length, 1) -> (n_seq * max_length,)

    :param seq_start_indices: Indices of the transitions that start a sequence
    :param seq_end_indices: Indices of the transitions that end a sequence
    :param tensor: Tensor of shape (max_length, n_seq, 1)
    :param padding_value: Value used to pad sequence to the same length
        (zero padding by default)
    :return: (n_seq * max_length,) aka (padded_batch_size,)
    """
    return pad(seq_start_indices, seq_end_indices, tensor, padding_value).flatten()


def create_sequencers(
    episode_starts: th.Tensor,
    env_change: th.Tensor,
    n_envs: int,
) -> Tuple[th.Tensor, int, Callable, Callable]:
    """
    Create the utility function to chunk data into
    sequences and pad them to create fixed size tensors.

    :param episode_starts: Indices where an episode starts
    :param env_change: Indices where the data collected
        come from a different env (when using multiple env for data collection)
    :return: Indices of the transitions that start a sequence,
        pad and pad_and_flatten utilities tailored for this batch
        (sequence starts and ends indices are fixed)
    """
    # Create sequence if env changes too
    seq_start = (episode_starts | env_change).flatten()
    # First index is always the beginning of a sequence
    seq_start[0] = True
    # Retrieve indices of sequence starts
    seq_start_indices = th.argwhere(seq_start).squeeze(1)
    # End of sequence are just before sequence starts
    # Last index is also always end of a sequence
    seq_end_indices = th.cat(
        [
            (seq_start_indices - 1)[1:],
            th.tensor([len(episode_starts)], device=seq_start_indices.device, dtype=seq_start_indices.dtype),
        ]
    )

    lengths_except_last = seq_start_indices[1:] - seq_start_indices[:-1]
    last_length = len(episode_starts) - seq_start_indices[-1].item()
    if lengths_except_last.numel():
        max_length = int(max(lengths_except_last.max().item(), last_length))
        mean_length = float(lengths_except_last.sum().item() + last_length) / (len(lengths_except_last) + 1)
    else:
        max_length = int(last_length)
        mean_length = float(last_length)

    n_episodes = len(lengths_except_last) + 1
    if n_episodes > n_envs * 3:
        log.warn(
            f"Episodes are too short. Probably this is an error. You should reduce the number of steps collected per "
            f"step by RecurrentPPO algorithm, or use an environment with longer episodes. "
            f"{n_episodes=}, {n_envs=}, {mean_length=}, {max_length=}"
        )

    # Create padding method for this minibatch
    # to avoid repeating arguments (seq_start_indices, seq_end_indices)
    local_pad = partial(pad, seq_start_indices, seq_end_indices)
    local_pad_and_flatten = partial(pad_and_flatten, seq_start_indices, seq_end_indices)
    return seq_start_indices, max_length, local_pad, local_pad_and_flatten


def space_to_example(
    batch_shape: Tuple[int, ...],
    space: spaces.Space,
    *,
    device: Optional[th.device] = None,
    ensure_non_batch_dim: bool = False,
) -> TensorTree:
    def _zeros_with_batch(x: np.ndarray) -> th.Tensor:
        shape = x.shape
        if ensure_non_batch_dim and len(shape) == 0:
            shape = (1,)
        return th.zeros((*batch_shape, *shape), device=device, dtype=as_torch_dtype(x.dtype))

    return tree_map(_zeros_with_batch, space.sample())


class RecurrentRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer that also stores the RNN states.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param hidden_state_example: Example buffer that will collect RNN states.
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        hidden_state_example: TensorTree,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super(RolloutBuffer, self).__init__(buffer_size, observation_space, action_space, device=device, n_envs=n_envs)
        self.hidden_state_example = hidden_state_example
        self.gae_lambda = gae_lambda
        self.gamma = gamma

        batch_shape = (self.buffer_size, self.n_envs)
        self.device = device = get_device(device)

        self.observation_space_example = space_to_example((), observation_space)

        self.advantages = th.zeros(batch_shape, dtype=th.float32, device=device)
        self.returns = th.zeros(batch_shape, dtype=th.float32, device=device)
        self.data = RecurrentRolloutBufferData(
            observations=space_to_example(batch_shape, self.observation_space, device=device, ensure_non_batch_dim=True),
            actions=th.zeros(
                (*batch_shape, self.action_dim),
                dtype=th.long if isinstance(self.action_space, (spaces.Discrete, spaces.MultiDiscrete)) else th.float32,
                device=device,
            ),
            rewards=th.zeros(batch_shape, dtype=th.float32, device=device),
            episode_starts=th.zeros(batch_shape, dtype=th.bool, device=device),
            values=th.zeros(batch_shape, dtype=th.float32, device=device),
            log_probs=th.zeros(batch_shape, dtype=th.float32, device=device),
            hidden_states=tree_map(
                lambda x: th.zeros((self.buffer_size, *x.shape), dtype=x.dtype, device=device), hidden_state_example
            ),
        )

    # Expose attributes of the RecurrentRolloutBufferData in the top-level to conform to the RolloutBuffer interface
    @property
    def episode_starts(self) -> th.Tensor:  # type: ignore[override]
        return self.data.episode_starts

    @property
    def values(self) -> th.Tensor:  # type: ignore[override]
        return self.data.values

    @property
    def rewards(self) -> th.Tensor:  # type: ignore[override]
        return self.data.rewards

    def reset(self):
        self.returns.zero_()
        self.advantages.zero_()
        tree_map(lambda x: x.zero_(), self.data)
        super(RolloutBuffer, self).reset()

    def extend(self, data: RecurrentRolloutBufferData) -> None:  # type: ignore[override]
        """
        Add a new batch of transitions to the buffer
        """

        # Do a for loop along the batch axis.
        # Treat lists as leaves to avoid flattening the infos.
        def _is_list(t):
            return isinstance(t, list)

        tensors: list[th.Tensor]
        tensors, _ = tree_flatten(data, is_leaf=_is_list)  # type: ignore
        len_tensors = len(tensors[0])
        assert all(len(t) == len_tensors for t in tensors), "All tensors must have the same batch size"
        for i in range(len_tensors):
            self.add(tree_index(data, i, is_leaf=_is_list))

    def add(self, data: RecurrentRolloutBufferData, **kwargs) -> None:  # type: ignore[override]
        """
        :param hidden_states: Hidden state of the RNN
        """
        new_data = dataclasses.replace(
            data, actions=data.actions.reshape((self.n_envs, self.action_dim))  # type: ignore[misc]
        )

        tree_map(
            lambda buf, x: buf[self.pos].copy_(x if x.ndim + 1 == buf.ndim else x.unsqueeze(-1), non_blocking=True),
            self.data,
            new_data,
        )
        # Increment pos
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(  # type: ignore[override]
        self, batch_size: Optional[int] = None
    ) -> Generator[RecurrentRolloutBufferSamples, None, None]:
        assert self.full, "Rollout buffer must be full before sampling from it"

        hidden_states = tree_map(lambda x: x.swapaxes(1, 2), self.data.hidden_states)
        data: RecurrentRolloutBufferData = tree_map(
            self.swap_and_flatten, dataclasses.replace(self.data, hidden_states=hidden_states)  # type: ignore[misc]
        )
        returns = self.swap_and_flatten(self.returns)
        advantages = self.swap_and_flatten(self.advantages)

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        # Sampling strategy that allows any mini batch size but requires
        # more complexity and use of padding
        # Trick to shuffle a bit: keep the sequence order
        # but split the indices in two
        split_index = int(np.random.randint(self.buffer_size * self.n_envs))
        indices = th.arange(self.buffer_size * self.n_envs)
        indices = th.cat((indices[split_index:], indices[:split_index]))

        env_change = th.zeros((self.buffer_size, self.n_envs), dtype=th.bool, device=self.device)
        # Flag first timestep as change of environment
        env_change[0, :] = True
        env_change = self.swap_and_flatten(env_change)

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            batch_inds = indices[start_idx : start_idx + batch_size]
            yield self._get_samples(data, returns, advantages, batch_inds, env_change)
            start_idx += batch_size

    def _get_samples(  # type: ignore[override]
        self,
        data: RecurrentRolloutBufferData,
        returns: th.Tensor,
        advantages: th.Tensor,
        batch_inds: th.Tensor,
        env_change: th.Tensor,
        env: Optional[VecNormalize] = None,
    ) -> RecurrentRolloutBufferSamples:
        # Retrieve sequence starts and utility function
        local_seq_start_indices, max_length, local_pad, local_pad_and_flatten = create_sequencers(
            data.episode_starts[batch_inds], env_change[batch_inds], self.n_envs
        )

        # Number of sequences
        n_seq = len(local_seq_start_indices)
        padded_batch_size = n_seq * max_length
        # We retrieve the RNN hidden states that will allow
        # to properly initialize the RNN at the beginning of each sequence
        # NOTE: this only gets the first index. For transformers we'll have to get *all* previous KV-vectors.

        rnn_states = tree_map(
            lambda x: self.to_device(x[batch_inds][local_seq_start_indices].swapaxes(0, 1)).contiguous(), data.hidden_states
        )

        observations = tree_map(
            lambda x, example: local_pad(x[batch_inds]).view(padded_batch_size, *example.shape),
            data.observations,
            self.observation_space_example,
        )

        return RecurrentRolloutBufferSamples(
            # (batch_size, obs_dim) -> (n_seq, max_length, obs_dim) -> (n_seq * max_length, obs_dim)
            observations=observations,
            actions=local_pad(data.actions[batch_inds]).view(padded_batch_size, *data.actions.shape[1:]),
            old_values=local_pad_and_flatten(data.values[batch_inds]),
            old_log_prob=local_pad_and_flatten(data.log_probs[batch_inds]),
            advantages=local_pad_and_flatten(advantages[batch_inds]),
            returns=local_pad_and_flatten(returns[batch_inds]),
            hidden_states=rnn_states,
            episode_starts=local_pad_and_flatten(data.episode_starts[batch_inds]),
            mask=local_pad_and_flatten(th.ones(returns[batch_inds].shape, dtype=th.bool, device=self.device)),
        )


RecurrentDictRolloutBuffer = RecurrentRolloutBuffer
