#!/usr/bin/env python3
"""
PPO Trainer for Humanoid Locomotion
Proximal Policy Optimization with vectorized environments

Author: Al Numan
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .policy import ActorCritic, PolicyConfig
from .env import HumanoidEnv, EnvConfig


@dataclass
class TrainConfig:
    """Training configuration"""
    # PPO hyperparameters
    num_steps: int = 24              # Steps per environment per update
    num_mini_batches: int = 4        # Mini-batches per update
    num_epochs: int = 5              # Epochs per update
    clip_param: float = 0.2          # PPO clip parameter
    value_loss_coef: float = 0.5     # Value loss coefficient
    entropy_coef: float = 0.01       # Entropy bonus coefficient
    max_grad_norm: float = 1.0       # Gradient clipping

    # Learning rates
    learning_rate: float = 3e-4
    lr_schedule: str = "adaptive"    # "fixed", "linear", "adaptive"

    # GAE
    gamma: float = 0.99              # Discount factor
    gae_lambda: float = 0.95         # GAE lambda

    # Training
    max_iterations: int = 1000
    save_interval: int = 100
    log_interval: int = 10

    # Paths
    save_dir: str = "checkpoints"
    log_dir: str = "logs"


class RolloutBuffer:
    """Buffer for storing rollout data"""

    def __init__(
        self,
        num_envs: int,
        num_steps: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device
    ):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.device = device

        # Allocate buffers
        self.observations = torch.zeros(num_steps, num_envs, obs_dim, device=device)
        self.actions = torch.zeros(num_steps, num_envs, action_dim, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.dones = torch.zeros(num_steps, num_envs, device=device)
        self.values = torch.zeros(num_steps, num_envs, device=device)
        self.log_probs = torch.zeros(num_steps, num_envs, device=device)

        # Advantages and returns (computed after rollout)
        self.advantages = torch.zeros(num_steps, num_envs, device=device)
        self.returns = torch.zeros(num_steps, num_envs, device=device)

        self.step = 0

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor
    ):
        """Add a transition to the buffer"""
        self.observations[self.step] = obs
        self.actions[self.step] = action
        self.rewards[self.step] = reward
        self.dones[self.step] = done.float()
        self.values[self.step] = value
        self.log_probs[self.step] = log_prob
        self.step += 1

    def compute_returns_and_advantages(
        self,
        last_values: torch.Tensor,
        gamma: float,
        gae_lambda: float
    ):
        """Compute GAE and returns"""
        last_gae_lam = 0

        for step in reversed(range(self.num_steps)):
            if step == self.num_steps - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]

            next_non_terminal = 1.0 - self.dones[step]
            delta = (
                self.rewards[step] +
                gamma * next_values * next_non_terminal -
                self.values[step]
            )

            self.advantages[step] = last_gae_lam = (
                delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            )

        self.returns = self.advantages + self.values

    def get_batches(
        self,
        num_mini_batches: int
    ):
        """Generate mini-batches for training"""
        batch_size = self.num_envs * self.num_steps
        mini_batch_size = batch_size // num_mini_batches

        # Flatten
        obs = self.observations.view(-1, self.observations.shape[-1])
        actions = self.actions.view(-1, self.actions.shape[-1])
        log_probs = self.log_probs.view(-1)
        advantages = self.advantages.view(-1)
        returns = self.returns.view(-1)
        values = self.values.view(-1)

        # Random permutation
        indices = torch.randperm(batch_size)

        for start in range(0, batch_size, mini_batch_size):
            end = start + mini_batch_size
            batch_indices = indices[start:end]

            yield (
                obs[batch_indices],
                actions[batch_indices],
                log_probs[batch_indices],
                advantages[batch_indices],
                returns[batch_indices],
                values[batch_indices]
            )

    def reset(self):
        """Reset buffer for new rollout"""
        self.step = 0


class PPOTrainer:
    """
    Proximal Policy Optimization trainer

    Features:
    - Vectorized environment support
    - GAE for advantage estimation
    - Adaptive learning rate
    - Tensorboard logging
    """

    def __init__(
        self,
        env: HumanoidEnv,
        policy: ActorCritic,
        config: TrainConfig = None,
        device: str = "cuda"
    ):
        """
        Initialize trainer

        Args:
            env: Vectorized environment
            policy: Actor-critic policy
            config: Training configuration
            device: Compute device
        """
        self.env = env
        self.policy = policy.to(device)
        self.config = config or TrainConfig()
        self.device = torch.device(device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.config.learning_rate
        )

        # Rollout buffer
        self.buffer = RolloutBuffer(
            num_envs=env.num_envs,
            num_steps=self.config.num_steps,
            obs_dim=env.num_obs,
            action_dim=env.num_actions,
            device=self.device
        )

        # Logging
        self.writer = SummaryWriter(self.config.log_dir)
        self.iteration = 0

        # Statistics
        self.episode_rewards = []
        self.episode_lengths = []

    def collect_rollouts(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Collect rollout data from environment

        Args:
            obs: Initial observations

        Returns:
            Final observations
        """
        self.buffer.reset()

        with torch.no_grad():
            for step in range(self.config.num_steps):
                # Get action from policy
                action, log_prob, value = self.policy.act(obs)

                # Step environment
                next_obs, reward, done, info = self.env.step(action)

                # Store transition
                self.buffer.add(obs, action, reward, done, value, log_prob)

                obs = next_obs

            # Get value for last observation (for GAE)
            _, _, last_value = self.policy.act(obs)

        # Compute advantages and returns
        self.buffer.compute_returns_and_advantages(
            last_value,
            self.config.gamma,
            self.config.gae_lambda
        )

        return obs

    def update_policy(self) -> Dict[str, float]:
        """
        Update policy using collected data

        Returns:
            Dictionary of training statistics
        """
        # Normalize advantages
        advantages = self.buffer.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.buffer.advantages = advantages

        # Training statistics
        value_losses = []
        policy_losses = []
        entropy_losses = []
        clip_fractions = []

        for epoch in range(self.config.num_epochs):
            for batch in self.buffer.get_batches(self.config.num_mini_batches):
                obs, actions, old_log_probs, advantages, returns, old_values = batch

                # Evaluate actions
                log_probs, values, entropy = self.policy.evaluate(obs, actions)

                # Policy loss (PPO clip)
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_param, 1 + self.config.clip_param) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = 0.5 * ((returns - values) ** 2).mean()

                # Entropy loss
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss +
                    self.config.value_loss_coef * value_loss +
                    self.config.entropy_coef * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                # Statistics
                value_losses.append(value_loss.item())
                policy_losses.append(policy_loss.item())
                entropy_losses.append(entropy_loss.item())

                clip_fraction = ((ratio - 1).abs() > self.config.clip_param).float().mean()
                clip_fractions.append(clip_fraction.item())

        return {
            'value_loss': np.mean(value_losses),
            'policy_loss': np.mean(policy_losses),
            'entropy_loss': np.mean(entropy_losses),
            'clip_fraction': np.mean(clip_fractions)
        }

    def train(self):
        """Main training loop"""
        print("=" * 60)
        print("PPO Training for Humanoid Locomotion")
        print("=" * 60)
        print(f"Environments: {self.env.num_envs}")
        print(f"Steps per update: {self.config.num_steps}")
        print(f"Max iterations: {self.config.max_iterations}")
        print("-" * 60)

        # Create save directory
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Initial reset
        obs = self.env.reset()

        start_time = time.time()

        for iteration in range(self.config.max_iterations):
            self.iteration = iteration

            # Collect rollouts
            obs = self.collect_rollouts(obs)

            # Update policy
            stats = self.update_policy()

            # Update learning rate
            if self.config.lr_schedule == "adaptive":
                self._update_learning_rate(stats['clip_fraction'])

            # Logging
            if iteration % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                fps = (iteration + 1) * self.env.num_envs * self.config.num_steps / elapsed

                # Compute mean reward
                mean_reward = self.buffer.rewards.mean().item()

                print(f"Iter {iteration:4d} | "
                      f"Reward: {mean_reward:7.2f} | "
                      f"Value Loss: {stats['value_loss']:.4f} | "
                      f"Policy Loss: {stats['policy_loss']:.4f} | "
                      f"Clip: {stats['clip_fraction']:.3f} | "
                      f"FPS: {fps:.0f}")

                # Tensorboard logging
                self.writer.add_scalar('reward/mean', mean_reward, iteration)
                self.writer.add_scalar('loss/value', stats['value_loss'], iteration)
                self.writer.add_scalar('loss/policy', stats['policy_loss'], iteration)
                self.writer.add_scalar('loss/entropy', stats['entropy_loss'], iteration)
                self.writer.add_scalar('train/clip_fraction', stats['clip_fraction'], iteration)
                self.writer.add_scalar('train/fps', fps, iteration)

            # Save checkpoint
            if iteration % self.config.save_interval == 0:
                self._save_checkpoint(iteration)

        # Final save
        self._save_checkpoint(self.config.max_iterations)

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)

    def _update_learning_rate(self, clip_fraction: float):
        """Adaptively update learning rate based on clip fraction"""
        target_clip = 0.2

        if clip_fraction > target_clip * 1.5:
            # Decrease LR
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.9
        elif clip_fraction < target_clip * 0.5:
            # Increase LR
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = min(param_group['lr'] * 1.1, 1e-3)

    def _save_checkpoint(self, iteration: int):
        """Save training checkpoint"""
        path = Path(self.config.save_dir) / f"checkpoint_{iteration:06d}.pt"
        torch.save({
            'iteration': iteration,
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.iteration = checkpoint['iteration']
        print(f"Loaded checkpoint: {path}")


def main():
    """Run PPO training"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_envs', type=int, default=4096)
    parser.add_argument('--max_iterations', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # Create environment
    env_config = EnvConfig(num_envs=args.num_envs)
    env = HumanoidEnv(config=env_config, device=args.device)

    # Create policy
    policy_config = PolicyConfig(
        obs_dim=env.num_obs,
        action_dim=env.num_actions
    )
    policy = ActorCritic(config=policy_config)

    # Create trainer
    train_config = TrainConfig(max_iterations=args.max_iterations)
    trainer = PPOTrainer(env, policy, config=train_config, device=args.device)

    # Train
    trainer.train()

    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
