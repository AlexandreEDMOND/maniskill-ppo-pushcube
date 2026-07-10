"""Small platform-specific setup shared by executable scripts."""

from __future__ import annotations

import os
import platform
from pathlib import Path


def configure_macos_vulkan(environment: dict[str, str] | None = None) -> dict[str, str]:
    """Expose a Homebrew MoltenVK installation to SAPIEN on macOS."""
    target = os.environ if environment is None else environment
    if platform.system() != "Darwin" or "VK_ICD_FILENAMES" in target:
        return target

    for prefix in (Path("/opt/homebrew"), Path("/usr/local")):
        icd_path = prefix / "etc/vulkan/icd.d/MoltenVK_icd.json"
        if icd_path.exists():
            target["VK_ICD_FILENAMES"] = str(icd_path)
            library_path = str(prefix / "lib")
            current_paths = target.get("DYLD_LIBRARY_PATH", "").split(":")
            if library_path not in current_paths:
                target["DYLD_LIBRARY_PATH"] = ":".join(
                    [library_path, *filter(None, current_paths)]
                )
            return target

    return target
