"""Fetch and run the pinned ManiSkill PPO baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "official": ROOT / "configs/official_baseline.json",
    "cpu-smoke": ROOT / "configs/cpu_smoke.json",
}
CACHE_DIR = ROOT / ".cache/maniskill-baselines"


def load_config(profile: str) -> dict:
    return json.loads(CONFIGS[profile].read_text())


def fetch_source(config: dict) -> Path:
    source = config["source"]
    filename = Path(source["entrypoint"]).name
    destination = CACHE_DIR / source["commit"] / filename
    url = (
        "https://raw.githubusercontent.com/haosulab/ManiSkill/"
        f"{source['commit']}/{source['entrypoint']}"
    )

    if destination.exists():
        content = destination.read_bytes()
    else:
        with urllib.request.urlopen(url) as response:
            content = response.read()

    digest = hashlib.sha256(content).hexdigest()
    if digest != source["sha256"]:
        raise RuntimeError(f"Upstream checksum mismatch: expected {source['sha256']}, got {digest}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return destination


def cpu_compatible_copy(source_path: Path) -> Path:
    content = source_path.read_text()
    original = 'sim_backend="physx_cuda"'
    if content.count(original) != 1:
        raise RuntimeError("Expected one hard-coded physx_cuda backend in upstream ppo.py")
    destination = source_path.with_name("ppo_physx_cpu.py")
    destination.write_text(content.replace(original, 'sim_backend="physx_cpu"'))
    return destination


def official_command(config: dict, source_path: Path, seed: int) -> list[str]:
    environment = config["environment"]
    training = config["training"]
    return [
        sys.executable,
        str(source_path),
        f"--env-id={environment['id']}",
        f"--control-mode={environment['control_mode']}",
        f"--seed={seed}",
        f"--num-envs={training['num_envs']}",
        f"--num-steps={training['num_steps']}",
        f"--update-epochs={training['update_epochs']}",
        f"--num-minibatches={training['num_minibatches']}",
        f"--total-timesteps={training['total_timesteps']}",
        f"--eval-freq={training['eval_frequency_iterations']}",
        f"--num-eval-envs={training['num_eval_envs']}",
        f"--num-eval-steps={training['num_eval_steps']}",
        "--cudagraphs",
        "--track",
        f"--exp-name=ppo-PushCube-v1-state-{seed}-walltime_efficient",
    ]


def smoke_command(config: dict, source_path: Path) -> list[str]:
    environment = config["environment"]
    training = config["training"]
    return [
        sys.executable,
        str(source_path),
        f"--env-id={environment['id']}",
        f"--control-mode={environment['control_mode']}",
        f"--seed={training['seed']}",
        f"--num-envs={training['num_envs']}",
        f"--num-steps={training['num_steps']}",
        f"--update-epochs={training['update_epochs']}",
        f"--num-minibatches={training['num_minibatches']}",
        f"--total-timesteps={training['total_timesteps']}",
        f"--eval-freq={training['eval_frequency_iterations']}",
        f"--num-eval-envs={training['num_eval_envs']}",
        f"--num-eval-steps={training['num_eval_steps']}",
        "--no-cuda",
        "--no-capture-video",
        f"--exp-name={config['name']}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=CONFIGS, default="cpu-smoke")
    parser.add_argument("--seed", type=int, default=9351)
    args = parser.parse_args()

    config = load_config(args.profile)
    source_path = fetch_source(config)

    if args.profile == "official":
        allowed_seeds = config["training"]["seeds"]
        if args.seed not in allowed_seeds:
            raise SystemExit(f"Seed must be one of {allowed_seeds}")
        if not torch.cuda.is_available():
            raise SystemExit("The official profile requires a CUDA-capable NVIDIA GPU")
        command = official_command(config, source_path, args.seed)
    else:
        command = smoke_command(config, cpu_compatible_copy(source_path))

    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(
        args.seed if args.profile == "official" else config["training"]["seed"]
    )
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
