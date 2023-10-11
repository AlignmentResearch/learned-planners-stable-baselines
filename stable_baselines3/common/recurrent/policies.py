import abc
from typing import Any, Dict, Generic, List, Optional, Tuple, Type, Union

import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.pytree_dataclass import tree_flatten
from stable_baselines3.common.recurrent.torch_layers import (
    ExtractorInput,
    GRUNatureCNNExtractor,
    GRUWrappedFeaturesExtractor,
    LSTMFlattenExtractor,
    RecurrentFeaturesExtractor,
    RecurrentState,
)
from stable_baselines3.common.recurrent.type_aliases import (
    ActorCriticStates,
    LSTMRecurrentState,
    non_null,
)
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
)
from stable_baselines3.common.type_aliases import Schedule, TorchGymObs


class BaseRecurrentActorCriticPolicy(ActorCriticPolicy, Generic[RecurrentState]):
    @abc.abstractmethod
    def recurrent_initial_state(
        self, n_envs: Optional[int] = None, *, device: Optional[th.device | str] = None
    ) -> RecurrentState:
        ...

    @abc.abstractmethod
    def forward(  # type: ignore[override]
        self,
        obs: TorchGymObs,
        state: RecurrentState,
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, RecurrentState]:
        """Advances to the next hidden state, and computes all the outputs of a recurrent policy.

        In this docstring the dimension letters are: Time (T), Batch (B) and others (...).

        :param obs: shape (T, B, ...) the policy will be applied in sequence to all the observations.
        :param state: shape (B, ...), the hidden state of the recurrent network
        :param episode_starts: shape (T, B), whether the current state is the start of an episode. This should be be 0
            everywhere except for T=0, where it may be 1.
        :param deterministic: if True return the best action, else a sample.
        :returns: (actions, values, log_prob, state). The actions, values and log-action-probabilities for every time
            step T, and the final state.
        """
        ...

    @abc.abstractmethod
    def get_distribution(  # type: ignore[override]
        self,
        obs: TorchGymObs,
        state: RecurrentState,
        episode_starts: th.Tensor,
    ) -> Tuple[Distribution, RecurrentState]:
        """
        Get the policy distribution for each step given the observations.

        :param obs: shape (T, B, ...) the policy will be applied in sequence to all the observations.
        :param state: shape (B, ...), the hidden state of the recurrent network
        :param episode_starts: shape (T, B), whether the current state is the start of an episode. This should be be 0
            everywhere except for T=0, where it may be 1.
        :return: the action distribution, the new hidden states.
        """
        ...

    @abc.abstractmethod
    def predict_values(  # type: ignore[override]
        self,
        obs: TorchGymObs,
        state: RecurrentState,
        episode_starts: th.Tensor,
    ) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: shape (T, B, ...) the policy will be applied in sequence to all the observations.
        :param state: shape (B, ...), the hidden state of the recurrent network
        :param episode_starts: shape (T, B), whether the current state is the start of an episode. This should be be 0
            everywhere except for T=0, where it may be 1.
        :return: The value for each time step.
        """
        ...

    @abc.abstractmethod
    def evaluate_actions(  # type: ignore[override]
        self, obs: TorchGymObs, actions: th.Tensor, state: RecurrentState, episode_starts: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: shape (T, B, ...) the policy will be applied in sequence to all the observations.
        :param actions: The actions taken at each step.
        :param state: shape (B, ...), the hidden state of the recurrent network
        :param episode_starts: shape (T, B), whether the current state is the start of an episode. This should be be 0
            everywhere except for T=0, where it may be 1.
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        ...

    @abc.abstractmethod
    def _predict(  # type: ignore[override]
        self,
        observation: TorchGymObs,
        state: RecurrentState,
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, RecurrentState]:
        """
        Get the action according to the policy for a given observation.

        :param obs: shape (T, B, ...) the policy will be applied in sequence to all the observations.
        :param state: shape (B, ...), the hidden state of the recurrent network
        :param episode_starts: shape (T, B), whether the current state is the start of an episode. This should be be 0
            everywhere except for T=0, where it may be 1.
        :param deterministic: if True return the best action, else a sample.
        :return: the model's action and the next hidden state
        """
        ...

    def predict(  # type: ignore[override]
        self,
        obs: TorchGymObs,
        state: Optional[RecurrentState] = None,
        episode_start: Optional[th.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, Optional[RecurrentState]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param obs: shape (T, B, ...) the policy will be applied in sequence to all the observations.
        :param state: shape (B, ...), the hidden state of the recurrent network
        :param episode_starts: shape (T, B), whether the current state is the start of an episode. This should be be 0
            everywhere except for T=0, where it may be 1.
        :param deterministic: if True return the best action, else a sample.
        :return: the model's action and the next hidden state
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        obs, vectorized_env = self.obs_to_tensor(obs)
        one_obs_tensor: th.Tensor
        (one_obs_tensor, *_), _ = tree_flatten(obs)  # type: ignore
        n_envs = len(one_obs_tensor)

        if state is None:
            state = self.recurrent_initial_state(n_envs, device=self.device)

        if episode_start is None:
            episode_start = th.zeros(n_envs, dtype=th.bool)

        with th.no_grad():
            # Convert to PyTorch tensors
            actions, state = self._predict(obs, state=state, episode_starts=episode_start, deterministic=deterministic)

        if isinstance(self.action_space, spaces.Box):
            if callable(self.squash_output):
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = th.clip(
                    actions, th.as_tensor(self.action_space.low).to(actions), th.as_tensor(self.action_space.high).to(actions)
                )

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(dim=0)

        return actions, state


class RecurrentActorCriticPolicy(BaseRecurrentActorCriticPolicy):
    """
    Recurrent policy class for actor-critic algorithms (has both policy and value prediction).
    To be used with A2C, PPO and the likes.
    It assumes that both the actor and the critic LSTM
    have the same architecture.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param lstm_hidden_size: Number of hidden units for each LSTM layer.
    :param n_lstm_layers: Number of LSTM layers.
    :param shared_lstm: Whether the LSTM is shared between the actor and the critic
        (in that case, only the actor gradient is used)
        By default, the actor and the critic have two separate LSTM.
    :param enable_critic_lstm: Use a seperate LSTM for the critic.
    :param lstm_kwargs: Additional keyword arguments to pass the the LSTM
        constructor.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        shared_lstm: bool = False,
        enable_critic_lstm: bool = True,
        lstm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.lstm_output_dim = lstm_hidden_size
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

        self.lstm_kwargs = lstm_kwargs or {}
        self.shared_lstm = shared_lstm
        self.enable_critic_lstm = enable_critic_lstm
        self.lstm_actor = LSTMFlattenExtractor(
            spaces.Box(-1e9, 1e9, (self.features_dim,)),
            features_dim=lstm_hidden_size,
            num_layers=n_lstm_layers,
            **self.lstm_kwargs,
        )
        # For the predict() method, to initialize hidden states
        # (n_lstm_layers, batch_size, lstm_hidden_size)
        self.lstm_hidden_state_shape = (n_lstm_layers, 1, lstm_hidden_size)
        self.critic = None
        self.lstm_critic = None
        assert not (
            self.shared_lstm and self.enable_critic_lstm
        ), "You must choose between shared LSTM, seperate or no LSTM for the critic."

        assert not (
            self.shared_lstm and not self.share_features_extractor
        ), "If the features extractor is not shared, the LSTM cannot be shared."

        # No LSTM for the critic, we still need to convert
        # output of features extractor to the correct size
        # (size of the output of the actor lstm)
        if not (self.shared_lstm or self.enable_critic_lstm):
            self.critic = nn.Linear(self.features_dim, lstm_hidden_size)

        # Use a separate LSTM for the critic
        if self.enable_critic_lstm:
            self.lstm_critic = LSTMFlattenExtractor(
                spaces.Box(-1e9, 1e9, (self.features_dim,)),
                features_dim=lstm_hidden_size,
                num_layers=n_lstm_layers,
                **self.lstm_kwargs,
            )

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs  # type: ignore[call-arg]
        )

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        self.mlp_extractor = MlpExtractor(
            self.lstm_output_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def recurrent_initial_state(self, n_envs: Optional[int] = None, *, device: Optional[th.device | str] = None):
        shape: tuple[int, ...]
        if n_envs is None:
            shape = (self.lstm_hidden_state_shape[0], self.lstm_hidden_state_shape[2])
        else:
            shape = (self.lstm_hidden_state_shape[0], n_envs, self.lstm_hidden_state_shape[2])
        return ActorCriticStates(
            (th.zeros(shape, device=device), th.zeros(shape, device=device)),
            (th.zeros(shape, device=device), th.zeros(shape, device=device)),
        )

    # Methods for getting `latent_vf` or `latent_pi`
    def _recurrent_latent_pi_and_vf(
        self, obs: TorchGymObs, state: ActorCriticStates, episode_starts: th.Tensor
    ) -> Tuple[Tuple[th.Tensor, th.Tensor], ActorCriticStates]:
        features = self.extract_features(obs)
        pi_features: th.Tensor
        vf_features: th.Tensor
        if self.share_features_extractor:
            assert isinstance(features, th.Tensor)
            pi_features = vf_features = features
        else:
            assert isinstance(features, tuple)
            pi_features, vf_features = features
        latent_pi, lstm_states_pi = self._process_sequence(pi_features, state.pi, episode_starts, self.lstm_actor)
        latent_vf, lstm_states_vf = self._recurrent_latent_vf_from_features(vf_features, state, episode_starts)
        if lstm_states_vf is None:
            lstm_states_vf = (lstm_states_pi[0].detach(), lstm_states_pi[1].detach())
        return ((latent_pi, latent_vf), ActorCriticStates(lstm_states_pi, lstm_states_vf))

    def _recurrent_latent_vf_from_features(
        self, vf_features: th.Tensor, state: ActorCriticStates, episode_starts: th.Tensor
    ) -> Tuple[th.Tensor, Optional[LSTMRecurrentState]]:
        "Get only the vf features, not advancing the hidden state"
        if self.lstm_critic is None:
            if self.shared_lstm:
                with th.no_grad():
                    latent_vf, _ = self._process_sequence(vf_features, state.pi, episode_starts, self.lstm_actor)
            else:
                latent_vf = non_null(self.critic)(vf_features)
            state_vf = None
        else:
            latent_vf, state_vf = self._process_sequence(vf_features, state.vf, episode_starts, self.lstm_critic)
        return latent_vf, state_vf

    def _recurrent_latent_vf_nostate(self, obs: TorchGymObs, state: ActorCriticStates, episode_starts: th.Tensor) -> th.Tensor:
        vf_features: th.Tensor = super(ActorCriticPolicy, self).extract_features(obs, self.vf_features_extractor)
        return self._recurrent_latent_vf_from_features(vf_features, state, episode_starts)[0]

    def forward(  # type: ignore[override]
        self,
        obs: TorchGymObs,
        state: ActorCriticStates,
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, ActorCriticStates]:
        (latent_pi, latent_vf), state = self._recurrent_latent_pi_and_vf(obs, state, episode_starts)
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob, state

    def get_distribution(  # type: ignore[override]
        self,
        obs: TorchGymObs,
        state: ActorCriticStates,
        episode_starts: th.Tensor,
    ) -> Tuple[Distribution, ActorCriticStates]:
        (latent_pi, _), state = self._recurrent_latent_pi_and_vf(obs, state, episode_starts)
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        return self._get_action_dist_from_latent(latent_pi), state

    def predict_values(  # type: ignore[override]
        self,
        obs: TorchGymObs,
        state: ActorCriticStates,
        episode_starts: th.Tensor,
    ) -> th.Tensor:
        latent_vf = self._recurrent_latent_vf_nostate(obs, state, episode_starts)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        return self.value_net(latent_vf)

    def evaluate_actions(  # type: ignore[override]
        self, obs: TorchGymObs, actions: th.Tensor, state: ActorCriticStates, episode_starts: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        (latent_pi, latent_vf), state = self._recurrent_latent_pi_and_vf(obs, state, episode_starts)
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, non_null(distribution.entropy())

    def _predict(  # type: ignore[override]
        self,
        observation: TorchGymObs,
        state: ActorCriticStates,
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, ActorCriticStates]:
        distribution, state = self.get_distribution(observation, state, episode_starts)
        return distribution.get_actions(deterministic=deterministic), state


class RecurrentActorCriticCnnPolicy(RecurrentActorCriticPolicy):
    """
    CNN recurrent policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param lstm_hidden_size: Number of hidden units for each LSTM layer.
    :param n_lstm_layers: Number of LSTM layers.
    :param shared_lstm: Whether the LSTM is shared between the actor and the critic.
        By default, only the actor has a recurrent network.
    :param enable_critic_lstm: Use a seperate LSTM for the critic.
    :param lstm_kwargs: Additional keyword arguments to pass the the LSTM
        constructor.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        shared_lstm: bool = False,
        enable_critic_lstm: bool = True,
        lstm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            lstm_hidden_size,
            n_lstm_layers,
            shared_lstm,
            enable_critic_lstm,
            lstm_kwargs,
        )


class RecurrentMultiInputActorCriticPolicy(RecurrentActorCriticPolicy):
    """
    MultiInputActorClass policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param lstm_hidden_size: Number of hidden units for each LSTM layer.
    :param n_lstm_layers: Number of LSTM layers.
    :param shared_lstm: Whether the LSTM is shared between the actor and the critic.
        By default, only the actor has a recurrent network.
    :param enable_critic_lstm: Use a seperate LSTM for the critic.
    :param lstm_kwargs: Additional keyword arguments to pass the the LSTM
        constructor.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        shared_lstm: bool = False,
        enable_critic_lstm: bool = True,
        lstm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            lstm_hidden_size,
            n_lstm_layers,
            shared_lstm,
            enable_critic_lstm,
            lstm_kwargs,
        )


class RecurrentFeaturesExtractorActorCriticPolicy(ActorCriticPolicy, Generic[ExtractorInput, RecurrentState]):
    features_extractor: RecurrentFeaturesExtractor[ExtractorInput, RecurrentState]

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = GRUNatureCNNExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}
        # Automatically deactivate dtype and bounds checks
        if normalize_images is False and issubclass(features_extractor_class, GRUNatureCNNExtractor):
            features_extractor_kwargs = features_extractor_kwargs.copy()
            features_extractor_kwargs.update(dict(normalized_image=True))

        if not issubclass(features_extractor_class, RecurrentFeaturesExtractor):
            base_features_extractor = features_extractor_class(observation_space, **features_extractor_kwargs)

            features_extractor_class = GRUWrappedFeaturesExtractor
            new_features_extractor_kwargs = dict(base_extractor=base_features_extractor)
            if "features_dim" in features_extractor_kwargs:
                new_features_extractor_kwargs["features_dim"] = features_extractor_kwargs["features_dim"]
            features_extractor_kwargs = new_features_extractor_kwargs
            print(features_extractor_class, features_extractor_kwargs)

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

    def recurrent_initial_state(
        self, n_envs: Optional[int] = None, *, device: Optional[th.device | str] = None
    ) -> RecurrentState:
        return self.features_extractor.recurrent_initial_state(n_envs, device=device)

    def _recurrent_extract_features(
        self, obs: TorchGymObs, state: RecurrentState, episode_starts: th.Tensor
    ) -> Tuple[th.Tensor, RecurrentState]:
        if not self.share_features_extractor:
            raise NotImplementedError("Non-shared features extractor not supported for recurrent extractors")

        preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)  # type: ignore
        return self.features_extractor(preprocessed_obs, state, episode_starts)

    def forward(  # type: ignore[override]
        self,
        obs: TorchGymObs,
        state: RecurrentState,
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, RecurrentState]:
        """Advances to the next hidden state, and computes all the outputs of a recurrent policy.

        In this docstring the dimension letters are: Time (T), Batch (B) and others (...).

        :param obs: shape (T, B, ...) the policy will be applied in sequence to all the observations.
        :param state: shape (B, ...), the hidden state of the recurrent network
        :param episode_starts: shape (T, B), whether the current state is the start of an episode. This should be be 0
            everywhere except for T=0, where it may be 1.
        :param deterministic: if True return the best action, else a sample.
        :returns: (actions, values, log_prob, state). The actions, values and log-action-probabilities for every time
            step T, and the final state.
        """
        latents, state = self._recurrent_extract_features(obs, state, episode_starts)
        latent_pi = self.mlp_extractor.forward_actor(latents)
        latent_vf = self.mlp_extractor.forward_critic(latents)

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob, state

    def get_distribution(  # type: ignore[override]
        self,
        obs: TorchGymObs,
        state: RecurrentState,
        episode_starts: th.Tensor,
    ) -> Tuple[Distribution, RecurrentState]:
        """
        Get the policy distribution for each step given the observations.

        :param obs: shape (T, B, ...) the policy will be applied in sequence to all the observations.
        :param state: shape (B, ...), the hidden state of the recurrent network
        :param episode_starts: shape (T, B), whether the current state is the start of an episode. This should be be 0
            everywhere except for T=0, where it may be 1.
        :return: the action distribution, the new hidden states.
        """
        latent_pi, state = self._recurrent_extract_features(obs, state, episode_starts)
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        return self._get_action_dist_from_latent(latent_pi), state

    def predict_values(  # type: ignore[override]
        self,
        obs: TorchGymObs,
        state: RecurrentState,
        episode_starts: th.Tensor,
    ) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: shape (T, B, ...) the policy will be applied in sequence to all the observations.
        :param state: shape (B, ...), the hidden state of the recurrent network
        :param episode_starts: shape (T, B), whether the current state is the start of an episode. This should be be 0
            everywhere except for T=0, where it may be 1.
        :return: The value for each time step.
        """
        latent_vf, _ = self._recurrent_extract_features(obs, state, episode_starts)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        return self.value_net(latent_vf)

    def evaluate_actions(  # type: ignore[override]
        self, obs: TorchGymObs, actions: th.Tensor, state: RecurrentState, episode_starts: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: shape (T, B, ...) the policy will be applied in sequence to all the observations.
        :param actions: The actions taken at each step.
        :param state: shape (B, ...), the hidden state of the recurrent network
        :param episode_starts: shape (T, B), whether the current state is the start of an episode. This should be be 0
            everywhere except for T=0, where it may be 1.
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        latents, state = self._recurrent_extract_features(obs, state, episode_starts)
        latent_pi = self.mlp_extractor.forward_actor(latents)
        latent_vf = self.mlp_extractor.forward_critic(latents)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, non_null(distribution.entropy())

    def _predict(  # type: ignore[override]
        self,
        observation: TorchGymObs,
        state: RecurrentState,
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, RecurrentState]:
        """
        Get the action according to the policy for a given observation.

        :param obs: shape (T, B, ...) the policy will be applied in sequence to all the observations.
        :param state: shape (B, ...), the hidden state of the recurrent network
        :param episode_starts: shape (T, B), whether the current state is the start of an episode. This should be be 0
            everywhere except for T=0, where it may be 1.
        :param deterministic: if True return the best action, else a sample.
        :return: the model's action and the next hidden state
        """
        distribution, state = self.get_distribution(observation, state, episode_starts)
        return distribution.get_actions(deterministic=deterministic), state
