import math
from pathlib import Path
import tempfile
import unittest

import torch

from scripts.custom_ppo import CustomPPOAgent, clipped_policy_loss, compute_gae, portable_path


class CustomPPOTests(unittest.TestCase):
    def test_gae_stops_at_episode_boundary(self):
        advantages, returns = compute_gae(
            rewards=torch.tensor([1.0, 1.0]),
            values=torch.tensor([0.5, 0.5]),
            dones=torch.tensor([0.0, 1.0]),
            last_value=torch.tensor(42.0),
            gamma=0.9,
            gae_lambda=0.95,
        )
        torch.testing.assert_close(advantages, torch.tensor([1.3775, 0.5]))
        torch.testing.assert_close(returns, torch.tensor([1.8775, 1.0]))

    def test_gae_handles_parallel_episode_boundaries(self):
        advantages, returns = compute_gae(
            rewards=torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
            values=torch.tensor([[0.5, 0.5], [0.5, 0.5]]),
            dones=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
            last_value=torch.tensor([42.0, 0.5]),
            gamma=0.9,
            gae_lambda=0.95,
        )
        torch.testing.assert_close(
            advantages, torch.tensor([[1.3775, 0.5], [0.5, 0.95]])
        )
        torch.testing.assert_close(
            returns, torch.tensor([[1.8775, 1.0], [1.0, 1.45]])
        )

    def test_gae_bootstraps_a_time_limit_truncation(self):
        advantages, returns = compute_gae(
            rewards=torch.tensor([1.0, 1.0]),
            values=torch.tensor([0.5, 0.5]),
            dones=torch.tensor([0.0, 1.0]),
            last_value=torch.tensor(42.0),
            bootstrap_values=torch.tensor([0.0, 42.0]),
            gamma=0.9,
            gae_lambda=0.95,
        )
        torch.testing.assert_close(advantages, torch.tensor([33.6965, 38.3]))
        torch.testing.assert_close(returns, torch.tensor([34.1965, 38.8]))

    def test_clipped_loss_limits_positive_advantage(self):
        loss = clipped_policy_loss(
            new_log_probabilities=torch.tensor([math.log(1.3)]),
            old_log_probabilities=torch.tensor([0.0]),
            advantages=torch.tensor([1.0]),
            clip_coefficient=0.2,
        )
        torch.testing.assert_close(loss, torch.tensor(-1.2))

    def test_squashed_actions_stay_within_environment_limits(self):
        agent = CustomPPOAgent(observation_size=2, action_size=2, hidden_size=4)
        with torch.no_grad():
            agent.actor_mean[-1].bias.fill_(10.0)
        action = agent.deterministic_action(torch.zeros(1, 2))
        self.assertTrue(torch.all(action <= 1.0))
        self.assertTrue(torch.all(action >= -1.0))

    def test_portable_path_preserves_an_external_output_path(self):
        root = Path("/workspace/project")
        with tempfile.TemporaryDirectory() as temporary_directory:
            external_path = Path(temporary_directory) / "final_ckpt.pt"
            self.assertEqual(portable_path(external_path, root), str(external_path))


if __name__ == "__main__":
    unittest.main()
