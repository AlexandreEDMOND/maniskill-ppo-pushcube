"""Train the project's minimal PPO implementation on PushCube-v1."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

from _platform import configure_macos_vulkan
from custom_ppo import (
    CHECKPOINT_FORMAT,
    CustomPPOAgent,
    clipped_policy_loss,
    compute_gae,
    portable_path,
)

configure_macos_vulkan()

import gymnasium as gym  # noqa: E402
import mani_skill.envs  # noqa: E402,F401
import torch  # noqa: E402
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/custom_ppo_cpu.json"


def format_duration(seconds: float) -> str:
    """Format a non-negative duration for the training progress display."""
    rounded_seconds = max(0, round(seconds))
    minutes, seconds = divmod(rounded_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


def show_progress(completed_steps: int, planned_steps: int, start_time: float) -> None:
    """Render one compact training progress line with throughput and ETA."""
    elapsed = time.perf_counter() - start_time
    rate = completed_steps / elapsed if elapsed else 0.0
    remaining_steps = planned_steps - completed_steps
    eta = remaining_steps / rate if rate else 0.0
    width = 30
    filled = round(width * completed_steps / planned_steps)
    bar = "#" * filled + "-" * (width - filled)
    message = (
        f"Training [{bar}] {completed_steps / planned_steps:6.2%} "
        f"{completed_steps:,}/{planned_steps:,} steps | {rate:,.0f} SPS "
        f"| elapsed {format_duration(elapsed)} | ETA {format_duration(eta)}"
    )
    print(message, end="\r" if sys.stdout.isatty() else "\n", flush=True)


def make_env(config: dict, num_envs: int = 1):
    environment = config["environment"]
    return gym.make(
        environment["id"],
        robot_uids=environment["robot"],
        obs_mode=environment["observation_mode"],
        control_mode=environment["control_mode"],
        reward_mode=environment["reward_mode"],
        sim_backend=environment["simulation_backend"],
        render_mode=None,
        num_envs=num_envs,
        reconfiguration_freq=0,
    )


def make_training_env(config: dict, num_envs: int) -> ManiSkillVectorEnv:
    """Create an autoresetting vector environment for rollout collection."""
    environment = config["environment"]
    if environment["simulation_backend"] == "physx_cuda" and not torch.cuda.is_available():
        raise RuntimeError("The CUDA PPO configuration requires a CUDA-enabled PyTorch build")
    return ManiSkillVectorEnv(
        make_env(config, num_envs=num_envs),
        num_envs=num_envs,
        auto_reset=True,
        ignore_terminations=False,
        record_metrics=True,
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
                    action = agent.deterministic_action(observation)
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

    num_envs = training.get("num_envs", 1)
    rollout_steps = training["rollout_steps"]
    batch_size = num_envs * rollout_steps
    if total_timesteps < batch_size:
        raise ValueError("total_timesteps must cover at least one complete vector rollout")
    planned_steps = math.ceil(total_timesteps / batch_size) * batch_size
    if planned_steps != total_timesteps:
        print(
            f"Requested {total_timesteps:,} steps; vector rollouts will run "
            f"{planned_steps:,} steps ({batch_size:,} per update)."
        )

    env = make_training_env(config, num_envs)
    try:
        observation_size = env.single_observation_space.shape[-1]
        action_size = env.single_action_space.shape[-1]
        agent = CustomPPOAgent(
            observation_size,
            action_size,
            training["hidden_size"],
            hidden_layers=training.get("hidden_layers", 2),
            initial_logstd=training.get("initial_logstd", -0.5),
        ).to(env.device)
        optimizer = torch.optim.Adam(agent.parameters(), lr=training["learning_rate"], eps=1e-5)
        evaluation_seeds = config["evaluation"]["seeds"]
        initial_evaluation = evaluate(agent, config, evaluation_seeds)

        observation, _ = env.reset(seed=seed)
        completed_steps = 0
        start_time = time.perf_counter()
        while completed_steps < planned_steps:
            observations = torch.zeros(
                rollout_steps, num_envs, observation_size, device=env.device
            )
            latent_actions = torch.zeros(
                rollout_steps, num_envs, action_size, device=env.device
            )
            log_probabilities = torch.zeros(rollout_steps, num_envs, device=env.device)
            rewards = torch.zeros(rollout_steps, num_envs, device=env.device)
            dones = torch.zeros(rollout_steps, num_envs, device=env.device)
            values = torch.zeros(rollout_steps, num_envs, device=env.device)

            for step in range(rollout_steps):
                observations[step] = observation
                with torch.no_grad():
                    latent_action, log_probability, _, value = agent.get_action_and_value(
                        observation
                    )
                latent_actions[step] = latent_action
                log_probabilities[step] = log_probability
                values[step] = value
                next_observation, reward, terminated, truncated, _ = env.step(
                    agent.environment_action(latent_action)
                )
                rewards[step] = reward
                dones[step] = (terminated | truncated).float()
                observation = next_observation
                completed_steps += num_envs

            with torch.no_grad():
                last_value = agent.value(observation)
                advantages, returns = compute_gae(
                    rewards,
                    values,
                    dones,
                    last_value,
                    training["gamma"],
                    training["gae_lambda"],
                )

            flat_observations = observations.flatten(0, 1)
            flat_latent_actions = latent_actions.flatten(0, 1)
            flat_log_probabilities = log_probabilities.flatten()
            flat_advantages = advantages.flatten()
            flat_returns = returns.flatten()
            for _ in range(training["update_epochs"]):
                indices = torch.randperm(batch_size, device=env.device)
                for batch_indices in indices.split(training["minibatch_size"]):
                    _, new_log_probabilities, entropy, new_values = agent.get_action_and_value(
                        flat_observations[batch_indices], flat_latent_actions[batch_indices]
                    )
                    normalized_advantages = flat_advantages[batch_indices]
                    normalized_advantages = (normalized_advantages - normalized_advantages.mean()) / (
                        normalized_advantages.std(unbiased=False) + 1e-8
                    )
                    policy_loss = clipped_policy_loss(
                        new_log_probabilities,
                        flat_log_probabilities[batch_indices],
                        normalized_advantages,
                        training["clip_coefficient"],
                    )
                    value_loss = torch.nn.functional.mse_loss(
                        new_values, flat_returns[batch_indices]
                    )
                    loss = policy_loss + training["value_coefficient"] * value_loss - training["entropy_coefficient"] * entropy.mean()
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), training["max_grad_norm"])
                    optimizer.step()
            show_progress(completed_steps, planned_steps, start_time)

        if sys.stdout.isatty():
            print()
        training_duration = time.perf_counter() - start_time

        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / "final_ckpt.pt"
        torch.save(
            {
                "format": CHECKPOINT_FORMAT,
                "observation_size": observation_size,
                "action_size": action_size,
                "hidden_size": training["hidden_size"],
                "hidden_layers": agent.hidden_layers,
                "initial_logstd": agent.initial_logstd,
                "squash_actions": agent.squash_actions,
                "state_dict": agent.state_dict(),
            },
            checkpoint_path,
        )
        final_evaluation = evaluate(agent, config, evaluation_seeds)
    finally:
        env.close()

    report = {
        "checkpoint": portable_path(checkpoint_path, ROOT),
        "requested_timesteps": total_timesteps,
        "total_timesteps": completed_steps,
        "num_envs": num_envs,
        "rollout_steps": rollout_steps,
        "throughput_steps_per_second": completed_steps / training_duration,
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
