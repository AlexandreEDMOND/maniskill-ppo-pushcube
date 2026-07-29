"""Small PPO building blocks shared by training and evaluation."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Normal

CHECKPOINT_FORMAT = "custom-ppo-v2"
LEGACY_CHECKPOINT_FORMAT = "custom-ppo-v1"
CHECKPOINT_FORMATS = (LEGACY_CHECKPOINT_FORMAT, CHECKPOINT_FORMAT)


class CustomPPOAgent(nn.Module):
    """A compact Gaussian actor-critic for state observations."""

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        hidden_size: int,
        squash_actions: bool = True,
    ):
        super().__init__()
        self.squash_actions = squash_actions
        self.critic = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_size),
        )
        self.actor_logstd = nn.Parameter(torch.full((1, action_size), -0.5))

    def get_action_and_value(
        self, observation: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actor_mean = self.actor_mean(observation)
        distribution = Normal(actor_mean, self.actor_logstd.exp().expand_as(actor_mean))
        if action is None:
            action = distribution.sample()
        return (
            action,
            distribution.log_prob(action).sum(-1),
            distribution.entropy().sum(-1),
            self.critic(observation).squeeze(-1),
        )

    def environment_action(self, action: torch.Tensor) -> torch.Tensor:
        """Map a latent Gaussian action into the environment's [-1, 1] range."""
        return action.tanh() if self.squash_actions else action

    def deterministic_action(self, observation: torch.Tensor) -> torch.Tensor:
        """Return the action used for deterministic evaluation."""
        return self.environment_action(self.actor_mean(observation))

    def value(self, observation: torch.Tensor) -> torch.Tensor:
        return self.critic(observation).squeeze(-1)


def portable_path(path: Path, root: Path) -> str:
    """Use a repository-relative path when possible, otherwise preserve the absolute path."""
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path.resolve())


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return generalized advantages and discounted value targets for one rollout."""
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros_like(last_value)
    for step in reversed(range(len(rewards))):
        next_value = last_value if step == len(rewards) - 1 else values[step + 1]
        not_done = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_value * not_done - values[step]
        last_advantage = delta + gamma * gae_lambda * not_done * last_advantage
        advantages[step] = last_advantage
    return advantages, advantages + values


def clipped_policy_loss(
    new_log_probabilities: torch.Tensor,
    old_log_probabilities: torch.Tensor,
    advantages: torch.Tensor,
    clip_coefficient: float,
) -> torch.Tensor:
    """PPO's clipped surrogate objective, expressed as a loss to minimize."""
    ratio = (new_log_probabilities - old_log_probabilities).exp()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_coefficient, 1.0 + clip_coefficient) * advantages
    return -torch.minimum(unclipped, clipped).mean()
