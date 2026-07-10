"""Validate the local PushCube environment and record one random episode."""

from __future__ import annotations

import os
import platform
from pathlib import Path


def configure_macos_vulkan() -> None:
    """Expose a Homebrew MoltenVK installation to SAPIEN on macOS."""
    if platform.system() != "Darwin" or "VK_ICD_FILENAMES" in os.environ:
        return

    for prefix in (Path("/opt/homebrew"), Path("/usr/local")):
        icd_path = prefix / "etc/vulkan/icd.d/MoltenVK_icd.json"
        if icd_path.exists():
            os.environ["VK_ICD_FILENAMES"] = str(icd_path)
            library_path = str(prefix / "lib")
            current_paths = os.environ.get("DYLD_LIBRARY_PATH", "").split(":")
            if library_path not in current_paths:
                os.environ["DYLD_LIBRARY_PATH"] = ":".join(
                    [library_path, *filter(None, current_paths)]
                )
            return


configure_macos_vulkan()

import gymnasium as gym  # noqa: E402
import mani_skill  # noqa: E402
import mani_skill.envs  # noqa: E402,F401
import torch  # noqa: E402
from mani_skill.utils.wrappers.record import RecordEpisode  # noqa: E402

ENV_ID = "PushCube-v1"
CONTROL_MODE = "pd_ee_delta_pose"
EPISODE_STEPS = 50
SEED = 0
VIDEO_DIR = Path("videos/smoke-test")


def main() -> None:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    existing_videos = set(VIDEO_DIR.glob("*.mp4"))

    env = gym.make(
        ENV_ID,
        obs_mode="state",
        control_mode=CONTROL_MODE,
        reward_mode="dense",
        render_mode="rgb_array",
        sim_backend="physx_cpu",
        num_envs=1,
    )

    try:
        first_observation, _ = env.reset(seed=SEED)
        repeated_observation, _ = env.reset(seed=SEED)
        torch.testing.assert_close(
            first_observation,
            repeated_observation,
            rtol=0,
            atol=0,
            msg="Resetting with the same seed changed the initial observation",
        )

        env = RecordEpisode(
            env,
            output_dir=str(VIDEO_DIR),
            save_trajectory=False,
            save_video=True,
            video_fps=20,
            avoid_overwriting_video=True,
        )
        observation, _ = env.reset(seed=SEED)
        env.action_space.seed(SEED)

        total_reward = 0.0
        steps = 0
        done = False
        while not done:
            observation, reward, terminated, truncated, info = env.step(
                env.action_space.sample()
            )
            total_reward += float(reward.item())
            steps += 1
            done = bool((terminated | truncated).item())

        assert steps == EPISODE_STEPS, f"Expected {EPISODE_STEPS} steps, got {steps}"
        assert bool(truncated.item()), "The random episode did not reach its time limit"
    finally:
        env.close()

    new_videos = set(VIDEO_DIR.glob("*.mp4")) - existing_videos
    assert len(new_videos) == 1, f"Expected one new video, found {len(new_videos)}"
    video_path = new_videos.pop()
    assert video_path.stat().st_size > 0, "The recorded video is empty"

    print(f"Python: {platform.python_version()}")
    print(f"ManiSkill: {mani_skill.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Environment: {ENV_ID}")
    print(f"Backend: physx_cpu")
    print(f"Observation shape: {tuple(observation.shape)}")
    print(f"Action shape: {env.action_space.shape}")
    print(f"Deterministic reset: passed (seed={SEED})")
    print(f"Episode: {steps} steps, return={total_reward:.3f}")
    print(f"Success: {bool(info['success'].item())}")
    print(f"Video: {video_path} ({video_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
