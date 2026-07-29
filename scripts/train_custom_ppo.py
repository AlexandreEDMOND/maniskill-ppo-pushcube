"""Train the project's minimal PPO implementation on PushCube-v1."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from custom_ppo import (
    CHECKPOINT_FORMAT,
    CustomPPOAgent,
    clipped_policy_loss,
    compute_gae,
    portable_path,
)

import gymnasium as gym  # noqa: E402
import mani_skill.envs  # noqa: E402,F401
import torch  # noqa: E402
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/custom_ppo_cpu.json"


def make_env(
    config: dict, num_envs: int = 1, simulation_backend: str | None = None
):
    environment = config["environment"]
    return gym.make(
        environment["id"],
        robot_uids=environment["robot"],
        obs_mode=environment["observation_mode"],
        control_mode=environment["control_mode"],
        reward_mode=environment["reward_mode"],
        sim_backend=simulation_backend or environment["simulation_backend"],
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
    """Run the fixed deterministic evaluator used to select custom checkpoints."""
    evaluation_backend = config["evaluation"].get(
        "simulation_backend", config["environment"]["simulation_backend"]
    )
    evaluation_agent = agent
    if evaluation_backend == "physx_cpu" and next(agent.parameters()).is_cuda:
        evaluation_agent = CustomPPOAgent(
            observation_size=agent.actor_mean[0].in_features,
            action_size=agent.actor_logstd.shape[-1],
            hidden_size=agent.actor_mean[0].out_features,
            squash_actions=agent.squash_actions,
            hidden_layers=agent.hidden_layers,
            initial_logstd=agent.initial_logstd,
        )
        evaluation_agent.load_state_dict(agent.state_dict())
    evaluation_agent.eval()

    env = make_env(config, simulation_backend=evaluation_backend)
    successes = []
    returns = []
    final_distances = []
    try:
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            done = False
            episode_return = 0.0
            success = False
            while not done:
                with torch.inference_mode():
                    action = evaluation_agent.deterministic_action(
                        observation.to(next(evaluation_agent.parameters()).device)
                    )
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward.item())
                success = success or bool(info["success"].item())
                done = bool((terminated | truncated).item())
            successes.append(float(success))
            returns.append(episode_return)
            base_env = env.unwrapped
            final_distances.append(
                torch.linalg.vector_norm(
                    base_env.obj.pose.p[0, :2] - base_env.goal_region.pose.p[0, :2]
                ).item()
            )
    finally:
        env.close()
    return {
        "success_rate": sum(successes) / len(successes),
        "return_mean": sum(returns) / len(returns),
        "final_cube_to_goal_distance_mean": sum(final_distances) / len(final_distances),
    }


def checkpoint_payload(
    agent: CustomPPOAgent,
    observation_size: int,
    action_size: int,
    step: int,
) -> dict:
    return {
        "format": CHECKPOINT_FORMAT,
        "observation_size": observation_size,
        "action_size": action_size,
        "hidden_size": agent.actor_mean[0].out_features,
        "hidden_layers": agent.hidden_layers,
        "initial_logstd": agent.initial_logstd,
        "squash_actions": agent.squash_actions,
        "step": step,
        "state_dict": agent.state_dict(),
    }


def checkpoint_score(metrics: dict[str, float]) -> tuple[float, float, float]:
    """Rank checkpoints by success first, then final distance and return."""
    return (
        metrics["success_rate"],
        -metrics["final_cube_to_goal_distance_mean"],
        metrics["return_mean"],
    )


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
    checkpoint_interval_steps = training.get("checkpoint_interval_steps")
    if checkpoint_interval_steps is not None and checkpoint_interval_steps <= 0:
        raise ValueError("checkpoint_interval_steps must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_snapshot.json").write_text(
        json.dumps(
            {
                "config": config,
                "requested_timesteps": total_timesteps,
                "planned_timesteps": planned_steps,
                "seed": seed,
            },
            indent=2,
        )
        + "\n"
    )
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_metrics_path = output_dir / "checkpoint_metrics.json"

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
        time_penalty_per_step = float(
            config.get("reward", {}).get("time_penalty_per_step", 0.0)
        )
        if time_penalty_per_step < 0:
            raise ValueError("time_penalty_per_step must be non-negative")

        checkpoint_results: list[dict] = []
        best_checkpoint_path: Path | None = None
        best_score: tuple[float, float, float] | None = None
        diagnostics = {
            "approx_kl": torch.zeros((), device=env.device),
            "clip_fraction": torch.zeros((), device=env.device),
            "policy_loss": torch.zeros((), device=env.device),
            "value_loss": torch.zeros((), device=env.device),
            "entropy": torch.zeros((), device=env.device),
        }
        diagnostic_count = 0

        def save_and_evaluate_checkpoint(step: int, filename: str) -> dict:
            nonlocal best_checkpoint_path, best_score, diagnostic_count
            checkpoint_path = checkpoint_dir / filename
            torch.save(
                checkpoint_payload(agent, observation_size, action_size, step),
                checkpoint_path,
            )
            metrics = evaluate(agent, config, evaluation_seeds)
            diagnostic_means = {
                name: (total / diagnostic_count).item() if diagnostic_count else 0.0
                for name, total in diagnostics.items()
            }
            diagnostic_means["action_std_mean"] = agent.actor_logstd.exp().mean().item()
            diagnostic_means["action_std_min"] = agent.actor_logstd.exp().min().item()
            diagnostic_means["learning_rate"] = optimizer.param_groups[0]["lr"]
            result = {
                "step": step,
                "checkpoint": portable_path(checkpoint_path, ROOT),
                "evaluation": metrics,
                "diagnostics": diagnostic_means,
            }
            checkpoint_results.append(result)
            if best_score is None or checkpoint_score(metrics) > best_score:
                best_score = checkpoint_score(metrics)
                best_checkpoint_path = checkpoint_dir / "best_ckpt.pt"
                torch.save(
                    checkpoint_payload(agent, observation_size, action_size, step),
                    best_checkpoint_path,
                )
                result["is_best"] = True
            checkpoint_metrics_path.write_text(
                json.dumps(checkpoint_results, indent=2) + "\n"
            )
            for total in diagnostics.values():
                total.zero_()
            diagnostic_count = 0
            return metrics

        observation, _ = env.reset(seed=seed)
        completed_steps = 0
        next_checkpoint_step = checkpoint_interval_steps
        start_time = time.perf_counter()
        with tqdm(
            total=planned_steps,
            desc="Training",
            unit="step",
            unit_scale=True,
            dynamic_ncols=True,
        ) as progress:
            while completed_steps < planned_steps:
                if training.get("anneal_learning_rate", False):
                    progress_fraction = completed_steps / planned_steps
                    optimizer.param_groups[0]["lr"] = training["learning_rate"] * (
                        1.0 - progress_fraction
                    )
                observations = torch.zeros(
                    rollout_steps, num_envs, observation_size, device=env.device
                )
                latent_actions = torch.zeros(
                    rollout_steps, num_envs, action_size, device=env.device
                )
                log_probabilities = torch.zeros(rollout_steps, num_envs, device=env.device)
                rewards = torch.zeros(rollout_steps, num_envs, device=env.device)
                dones = torch.zeros(rollout_steps, num_envs, device=env.device)
                bootstrap_values = torch.zeros(
                    rollout_steps, num_envs, device=env.device
                )
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
                    next_observation, reward, terminated, truncated, info = env.step(
                        agent.environment_action(latent_action)
                    )
                    rewards[step] = reward - time_penalty_per_step
                    dones[step] = (terminated | truncated).float()
                    if truncated.any():
                        with torch.no_grad():
                            bootstrap_values[step, truncated] = agent.value(
                                info["final_observation"][truncated]
                            )
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
                        bootstrap_values,
                    )

                flat_observations = observations.flatten(0, 1)
                flat_latent_actions = latent_actions.flatten(0, 1)
                flat_log_probabilities = log_probabilities.flatten()
                flat_advantages = advantages.flatten()
                flat_returns = returns.flatten()
                target_kl = training.get("target_kl")
                for _ in range(training["update_epochs"]):
                    indices = torch.randperm(batch_size, device=env.device)
                    epoch_kl = torch.zeros((), device=env.device)
                    epoch_batches = 0
                    for batch_indices in indices.split(training["minibatch_size"]):
                        _, new_log_probabilities, entropy, new_values = agent.get_action_and_value(
                            flat_observations[batch_indices], flat_latent_actions[batch_indices]
                        )
                        log_ratio = new_log_probabilities - flat_log_probabilities[batch_indices]
                        ratio = log_ratio.exp()
                        with torch.no_grad():
                            approx_kl = ((ratio - 1.0) - log_ratio).mean()
                            clip_fraction = (
                                (ratio - 1.0).abs() > training["clip_coefficient"]
                            ).float().mean()
                        normalized_advantages = flat_advantages[batch_indices]
                        normalized_advantages = (
                            normalized_advantages - normalized_advantages.mean()
                        ) / (normalized_advantages.std(unbiased=False) + 1e-8)
                        policy_loss = clipped_policy_loss(
                            new_log_probabilities,
                            flat_log_probabilities[batch_indices],
                            normalized_advantages,
                            training["clip_coefficient"],
                        )
                        value_loss = torch.nn.functional.mse_loss(
                            new_values, flat_returns[batch_indices]
                        )
                        loss = (
                            policy_loss
                            + training["value_coefficient"] * value_loss
                            - training["entropy_coefficient"] * entropy.mean()
                        )
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(agent.parameters(), training["max_grad_norm"])
                        optimizer.step()
                        diagnostics["approx_kl"] += approx_kl
                        diagnostics["clip_fraction"] += clip_fraction
                        diagnostics["policy_loss"] += policy_loss.detach()
                        diagnostics["value_loss"] += value_loss.detach()
                        diagnostics["entropy"] += entropy.detach().mean()
                        diagnostic_count += 1
                        epoch_kl += approx_kl
                        epoch_batches += 1
                    if target_kl is not None and (epoch_kl / epoch_batches).item() > target_kl:
                        break
                progress.update(batch_size)
                if (
                    next_checkpoint_step is not None
                    and completed_steps >= next_checkpoint_step
                ):
                    checkpoint_metrics = save_and_evaluate_checkpoint(
                        completed_steps, f"ckpt_{completed_steps}.pt"
                    )
                    progress.set_postfix(
                        success=f"{checkpoint_metrics['success_rate']:.0%}",
                        distance=f"{checkpoint_metrics['final_cube_to_goal_distance_mean']:.3f}",
                    )
                    next_checkpoint_step += checkpoint_interval_steps
        training_duration = time.perf_counter() - start_time

        checkpoint_path = output_dir / "final_ckpt.pt"
        torch.save(
            checkpoint_payload(agent, observation_size, action_size, completed_steps),
            checkpoint_path,
        )
        if checkpoint_results and checkpoint_results[-1]["step"] == completed_steps:
            final_evaluation = checkpoint_results[-1]["evaluation"]
        else:
            final_evaluation = save_and_evaluate_checkpoint(
                completed_steps, f"ckpt_final_{completed_steps}.pt"
            )
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
        "best_checkpoint": (
            portable_path(best_checkpoint_path, ROOT) if best_checkpoint_path else None
        ),
        "checkpoint_metrics": portable_path(checkpoint_metrics_path, ROOT),
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
