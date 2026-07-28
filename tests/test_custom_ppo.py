import math
import unittest

import torch

from scripts.custom_ppo import clipped_policy_loss, compute_gae


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

    def test_clipped_loss_limits_positive_advantage(self):
        loss = clipped_policy_loss(
            new_log_probabilities=torch.tensor([math.log(1.3)]),
            old_log_probabilities=torch.tensor([0.0]),
            advantages=torch.tensor([1.0]),
            clip_coefficient=0.2,
        )
        torch.testing.assert_close(loss, torch.tensor(-1.2))


if __name__ == "__main__":
    unittest.main()
