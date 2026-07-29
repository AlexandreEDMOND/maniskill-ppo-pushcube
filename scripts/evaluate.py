"""Evaluate an official ManiSkill PPO checkpoint with a fixed protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import gymnasium as gym  # noqa: E402
import mani_skill.envs  # noqa: E402,F401
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from mani_skill.utils.wrappers.record import RecordEpisode  # noqa: E402

from custom_ppo import (  # noqa: E402
    CHECKPOINT_FORMAT,
    CHECKPOINT_FORMATS,
    CustomPPOAgent,
    portable_path,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/official_baseline.json"


class OfficialPPOAgent(nn.Module):
    """Network shape shared by ManiSkill v3.0.1 ppo.py and ppo_fast.py."""

    def __init__(self, observation_size: int, action_size: int):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(observation_size, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(observation_size, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_size),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_size))


def make_env(config: dict, render: bool):
    environment = config["environment"]
    return gym.make(
        environment["id"],
        robot_uids=environment["robot"],
        obs_mode=environment["observation_mode"],
        control_mode=environment["control_mode"],
        reward_mode=environment["reward_mode"],
        sim_backend=config["evaluation"]["simulation_backend"],
        render_backend="sapien_cpu" if render else "none",
        render_mode="rgb_array" if render else None,
        num_envs=1,
    )


def load_agent(checkpoint: Path, env) -> nn.Module:
    observation_size = env.observation_space.shape[-1]
    action_size = env.action_space.shape[-1]
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if checkpoint_data.get("format") in CHECKPOINT_FORMATS:
        if checkpoint_data["observation_size"] != observation_size:
            raise RuntimeError("Custom checkpoint observation size does not match the environment")
        if checkpoint_data["action_size"] != action_size:
            raise RuntimeError("Custom checkpoint action size does not match the environment")
        agent = CustomPPOAgent(
            observation_size,
            action_size,
            checkpoint_data["hidden_size"],
            squash_actions=checkpoint_data.get("squash_actions", False),
            hidden_layers=checkpoint_data.get("hidden_layers", 2),
            initial_logstd=checkpoint_data.get("initial_logstd", -0.5),
        )
        state_dict = checkpoint_data["state_dict"]
    else:
        agent = OfficialPPOAgent(observation_size, action_size)
        state_dict = checkpoint_data
    agent.load_state_dict(state_dict, strict=True)
    agent.eval()
    return agent


def run_episode(env, agent: nn.Module, seed: int) -> dict:
    observation, _ = env.reset(seed=seed)
    episode_return = 0.0
    success_once = False
    steps_to_success = None
    steps = 0
    done = False

    while not done:
        with torch.inference_mode():
            action = (
                agent.deterministic_action(observation)
                if isinstance(agent, CustomPPOAgent)
                else agent.actor_mean(observation)
            )
        observation, reward, terminated, truncated, info = env.step(action)
        steps += 1
        episode_return += float(reward.item())
        current_success = bool(info["success"].item())
        if current_success and steps_to_success is None:
            steps_to_success = steps
        success_once = success_once or current_success
        done = bool((terminated | truncated).item())

    base_env = env.unwrapped
    final_distance = torch.linalg.vector_norm(
        base_env.obj.pose.p[0, :2] - base_env.goal_region.pose.p[0, :2]
    ).item()
    return {
        "seed": seed,
        "success": success_once,
        "final_success": bool(info["success"].item()),
        "return": episode_return,
        "episode_length": steps,
        "steps_to_success": steps_to_success,
        "final_cube_to_goal_distance": final_distance,
    }


def mean_and_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
    }


def aggregate(episodes: list[dict]) -> dict:
    successful_steps = [
        episode["steps_to_success"]
        for episode in episodes
        if episode["steps_to_success"] is not None
    ]
    return {
        "episodes": len(episodes),
        "success_rate": statistics.fmean(episode["success"] for episode in episodes),
        "final_success_rate": statistics.fmean(
            episode["final_success"] for episode in episodes
        ),
        "return": mean_and_std([episode["return"] for episode in episodes]),
        "episode_length": mean_and_std(
            [episode["episode_length"] for episode in episodes]
        ),
        "steps_to_success_mean": (
            statistics.fmean(successful_steps) if successful_steps else None
        ),
        "final_cube_to_goal_distance": mean_and_std(
            [episode["final_cube_to_goal_distance"] for episode in episodes]
        ),
    }


def record_video(config: dict, agent: nn.Module, seed: int, output_dir: Path) -> Path:
    video_dir = output_dir / "videos"
    existing_videos = set(video_dir.glob("*.mp4"))
    env = make_env(config, render=True)
    env = RecordEpisode(
        env,
        output_dir=str(video_dir),
        save_trajectory=False,
        save_video=True,
        video_fps=20,
        avoid_overwriting_video=True,
    )
    try:
        run_episode(env, agent, seed)
    finally:
        env.close()

    new_videos = set(video_dir.glob("*.mp4")) - existing_videos
    if len(new_videos) != 1:
        raise RuntimeError(f"Expected one new video, found {len(new_videos)}")
    return new_videos.pop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    output_dir = args.output_dir or ROOT / "artifacts/evaluations" / args.checkpoint.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(config, render=False)
    try:
        agent = load_agent(args.checkpoint, env)
        episodes = [
            run_episode(env, agent, seed) for seed in config["evaluation"]["seeds"]
        ]
    finally:
        env.close()

    report = {
        "agent": (
            CHECKPOINT_FORMAT if isinstance(agent, CustomPPOAgent) else config["name"]
        ),
        "evaluation_protocol": config["name"],
        "checkpoint": portable_path(args.checkpoint, ROOT),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "config": portable_path(args.config, ROOT),
        "aggregate": aggregate(episodes),
        "episodes": episodes,
    }
    if not args.no_video:
        video_path = record_video(
            config,
            agent,
            config["evaluation"].get("video_seed", config["evaluation"]["seeds"][0]),
            output_dir,
        )
        report["video"] = portable_path(video_path, ROOT)

    report_path = output_dir / "evaluation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["aggregate"], indent=2))
    print(f"Report: {report_path}")
    if "video" in report:
        print(f"Video: {report['video']}")


if __name__ == "__main__":
    main()
