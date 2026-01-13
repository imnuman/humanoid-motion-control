#!/usr/bin/env python3
"""
Policy Networks for Humanoid Locomotion
Actor-Critic architecture with recurrent option

Author: Al Numan
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.distributions import Normal


@dataclass
class PolicyConfig:
    """Policy network configuration"""
    obs_dim: int = 48              # Observation dimension
    action_dim: int = 19           # Action dimension (joint commands)
    hidden_dims: Tuple[int, ...] = (256, 256, 128)
    activation: str = "elu"
    use_lstm: bool = False
    lstm_hidden_dim: int = 256
    init_noise_std: float = 1.0
    min_noise_std: float = 0.1


def get_activation(name: str) -> nn.Module:
    """Get activation function by name"""
    activations = {
        'relu': nn.ReLU(),
        'elu': nn.ELU(),
        'tanh': nn.Tanh(),
        'leaky_relu': nn.LeakyReLU(0.2),
        'selu': nn.SELU()
    }
    return activations.get(name, nn.ELU())


class MLP(nn.Module):
    """Multi-layer perceptron"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Tuple[int, ...],
        activation: str = "elu",
        output_activation: str = None
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(get_activation(activation))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        if output_activation:
            layers.append(get_activation(output_activation))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Actor(nn.Module):
    """Actor network (policy)"""

    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.config = config

        # Main network
        self.backbone = MLP(
            input_dim=config.obs_dim,
            output_dim=config.action_dim,
            hidden_dims=config.hidden_dims,
            activation=config.activation
        )

        # Learnable log standard deviation
        self.log_std = nn.Parameter(
            torch.ones(config.action_dim) * np.log(config.init_noise_std)
        )

    def forward(
        self,
        obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            obs: Observations (batch, obs_dim)

        Returns:
            Tuple of (action_mean, action_std)
        """
        mean = self.backbone(obs)

        # Clamp log_std for stability
        log_std = torch.clamp(
            self.log_std,
            min=np.log(self.config.min_noise_std),
            max=2.0
        )
        std = torch.exp(log_std)

        return mean, std

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action from policy

        Args:
            obs: Observations
            deterministic: If True, return mean action

        Returns:
            Tuple of (action, log_prob)
        """
        mean, std = self.forward(obs)

        if deterministic:
            action = mean
            log_prob = torch.zeros(obs.shape[0], device=obs.device)
        else:
            dist = Normal(mean, std)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob


class Critic(nn.Module):
    """Critic network (value function)"""

    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.config = config

        self.network = MLP(
            input_dim=config.obs_dim,
            output_dim=1,
            hidden_dims=config.hidden_dims,
            activation=config.activation
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            obs: Observations (batch, obs_dim)

        Returns:
            Value estimate (batch, 1)
        """
        return self.network(obs)


class ActorCritic(nn.Module):
    """Combined Actor-Critic network"""

    def __init__(self, config: PolicyConfig = None):
        super().__init__()
        self.config = config or PolicyConfig()

        self.actor = Actor(self.config)
        self.critic = Critic(self.config)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Returns:
            Tuple of (action_mean, action_std, value)
        """
        mean, std = self.actor(obs)
        value = self.critic(obs)
        return mean, std, value

    def act(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action and value

        Returns:
            Tuple of (action, log_prob, value)
        """
        action, log_prob = self.actor.get_action(obs, deterministic)
        value = self.critic(obs).squeeze(-1)
        return action, log_prob, value

    def evaluate(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update

        Returns:
            Tuple of (log_prob, value, entropy)
        """
        mean, std = self.actor(obs)
        value = self.critic(obs).squeeze(-1)

        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, value, entropy


class RecurrentActorCritic(nn.Module):
    """Actor-Critic with LSTM for temporal memory"""

    def __init__(self, config: PolicyConfig = None):
        super().__init__()
        self.config = config or PolicyConfig()

        # Encoder
        self.encoder = MLP(
            input_dim=config.obs_dim,
            output_dim=config.lstm_hidden_dim,
            hidden_dims=(256,),
            activation=config.activation
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=config.lstm_hidden_dim,
            hidden_size=config.lstm_hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # Actor head
        self.actor_head = MLP(
            input_dim=config.lstm_hidden_dim,
            output_dim=config.action_dim,
            hidden_dims=(128,),
            activation=config.activation
        )

        # Critic head
        self.critic_head = MLP(
            input_dim=config.lstm_hidden_dim,
            output_dim=1,
            hidden_dims=(128,),
            activation=config.activation
        )

        # Log std
        self.log_std = nn.Parameter(
            torch.ones(config.action_dim) * np.log(config.init_noise_std)
        )

    def forward(
        self,
        obs: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        """
        Forward pass with hidden state

        Args:
            obs: Observations (batch, seq_len, obs_dim) or (batch, obs_dim)
            hidden: LSTM hidden state tuple

        Returns:
            Tuple of (action_mean, action_std, value, new_hidden)
        """
        # Handle single step input
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # Add sequence dimension

        batch_size = obs.shape[0]
        seq_len = obs.shape[1]

        # Encode
        encoded = self.encoder(obs)

        # LSTM
        if hidden is None:
            hidden = self._init_hidden(batch_size, obs.device)

        lstm_out, new_hidden = self.lstm(encoded, hidden)

        # Actor output
        mean = self.actor_head(lstm_out)
        std = torch.exp(torch.clamp(self.log_std, -2, 2))

        # Critic output
        value = self.critic_head(lstm_out)

        # Remove sequence dimension if single step
        if seq_len == 1:
            mean = mean.squeeze(1)
            value = value.squeeze(1)

        return mean, std, value, new_hidden

    def _init_hidden(
        self,
        batch_size: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden state"""
        h0 = torch.zeros(1, batch_size, self.config.lstm_hidden_dim, device=device)
        c0 = torch.zeros(1, batch_size, self.config.lstm_hidden_dim, device=device)
        return (h0, c0)
