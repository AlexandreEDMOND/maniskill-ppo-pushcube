"""Train the project's minimal PPO implementation on PushCube-v1."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from _platform import configure_macos_vulkan
from custom_ppo import CHECKPOINT_FORMAT, CustomPPOAgent, clipped_policy_loss, compute_gae

configure_macos_vulkan()

import gymnasium as gym  # noqa: E402
import mani_skill.envs  # noqa: E402,F401
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/custom_ppo_cpu.json"


def make_env(config: dict):
    environment = config["environment"]
    return gym.make(
        environment["id"],
        robot_uids=environment["robot"],
        obs_mode=environment["observation_mode"],
        control_mode=environment["control_mode"],
        reward_mode=environment["reward_mode"],
        sim_backend=environment["simulation_backend"],
        render_mode=None,
        num_envs=1,
    )


def evaluate(agent: CustomPPOAgent, config: dict, seeds: list[int]) -> dict[str, float]:
    env = make_env(config)
    successes = []
    returns = []
    try:
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            done = False
            episode_return = 0.0
            success = False
            while not done:
                with torch.inference_mode():
                    action = agent.actor_mean(observation)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward.item())
                success = success or bool(info["success"].item())
                done = bool((terminated | truncated).item())
            successes.append(float(success))
            returns.append(episode_return)
    finally:
        env.close()
    return {
        "success_rate": sum(successes) / len(successes),
        "return_mean": sum(returns) / len(returns),
    }


def train(config: dict, output_dir: Path, total_timesteps: int, seed: int) -> dict:
    training = config["training"]
    random.seed(seed)
    torch.manual_seed(seed)

    env = make_env(config)
    try:
        observation_size = env.observation_space.shape[-1]
        action_size = env.action_space.shape[-1]
        agent = CustomPPOAgent(observation_size, action_size, training["hidden_size"])
        optimizer = torch.optim.Adam(agent.parameters(), lr=training["learning_rate"], eps=1e-5)
        action_low = torch.as_tensor(env.action_space.low)
        action_high = torch.as_tensor(env.action_space.high)
        evaluation_seeds = config["evaluation"]["seeds"]
        initial_evaluation = evaluate(agent, config, evaluation_seeds)

        observation, _ = env.reset(seed=seed)
        completed_steps = 0
        while completed_steps < total_timesteps:
            rollout_steps = min(training["rollout_steps"], total_timesteps - completed_steps)
            observations = torch.zeros(rollout_steps, observation_size)
            actions = torch.zeros(rollout_steps, action_size)
            log_probabilities = torch.zeros(rollout_steps)
            rewards = torch.zeros(rollout_steps)
            dones = torch.zeros(rollout_steps)
            values = torch.zeros(rollout_steps)

            for step in range(rollout_steps):
                observations[step] = observation.squeeze(0)
                with torch.no_grad():
                    action, log_probability, _, value = agent.get_action_and_value(observation)
                actions[step] = action.squeeze(0)
                log_probabilities[step] = log_probability.item()
                values[step] = value.item()
                next_observation, reward, terminated, truncated, _ = env.step(
                    action.clamp(action_low, action_high)
                )
                done = bool((terminated | truncated).item())
                rewards[step] = reward.item()
                dones[step] = float(done)
                observation, _ = env.reset() if done else (next_observation, {})
                completed_steps += 1

            with torch.no_grad():
                last_value = agent.value(observation).squeeze(0)
                advantages, returns = compute_gae(
                    rewards,
                    values,
                    dones,
                    last_value,
                    training["gamma"],
                    training["gae_lambda"],
                )

            for _ in range(training["update_epochs"]):
                indices = torch.randperm(rollout_steps)
                for batch_indices in indices.split(training["minibatch_size"]):
                    _, new_log_probabilities, entropy, new_values = agent.get_action_and_value(
                        observations[batch_indices], actions[batch_indices]
                    )
                    normalized_advantages = advantages[batch_indices]
                    normalized_advantages = (normalized_advantages - normalized_advantages.mean()) / (
                        normalized_advantages.std(unbiased=False) + 1e-8
                    )
                    policy_loss = clipped_policy_loss(
                        new_log_probabilities,
                        log_probabilities[batch_indices],
                        normalized_advantages,
                        training["clip_coefficient"],
                    )
                    value_loss = torch.nn.functional.mse_loss(new_values, returns[batch_indices])
                    loss = policy_loss + training["value_coefficient"] * value_loss - training["entropy_coefficient"] * entropy.mean()
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), training["max_grad_norm"])
                    optimizer.step()

        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / "final_ckpt.pt"
        torch.save(
            {
                "format": CHECKPOINT_FORMAT,
                "observation_size": observation_size,
                "action_size": action_size,
                "hidden_size": training["hidden_size"],
                "state_dict": agent.state_dict(),
            },
            checkpoint_path,
        )
        final_evaluation = evaluate(agent, config, evaluation_seeds)
    finally:
        env.close()

    report = {
        "checkpoint": str(checkpoint_path.resolve().relative_to(ROOT)),
        "total_timesteps": completed_steps,
        "seed": seed,
        "initial_evaluation": initial_evaluation,
        "final_evaluation": final_evaluation,
    }
    (output_dir / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/custom-ppo-cpu")
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    training = config["training"]
    report = train(
        config,
        args.output_dir,
        args.total_timesteps or training["total_timesteps"],
        args.seed if args.seed is not None else training["seed"],
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
