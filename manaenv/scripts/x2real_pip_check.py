"""Run ``pip check`` and tolerate only NVIDIA SBSA platform-tag false positives.

NVIDIA publishes some aarch64 CUDA wheels (e.g. ``nvidia-cusparselt-cu13``,
required by the cu130 torch build) with the ``manylinux2014_sbsa`` platform tag.
pip does not recognise ``sbsa`` as a compatible tag, so ``pip check`` reports
"<dist> <version> is not supported on this platform" even though the wheel is
the correct Arm SBSA build. Every other ``pip check`` finding still fails.
"""

from __future__ import annotations

import platform
import re
import subprocess
import sys
from importlib import metadata

_UNSUPPORTED = re.compile(r"^(?P<name>\S+) (?P<version>\S+) is not supported on this platform$")


def _is_sbsa_wheel(name: str, version: str) -> bool:
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return False
    if dist.version != version:
        return False
    wheel = dist.read_text("WHEEL") or ""
    tags = [line.split(":", 1)[1].strip() for line in wheel.splitlines() if line.startswith("Tag:")]
    return bool(tags) and all(tag.endswith("_sbsa") for tag in tags)


def main() -> int:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"], capture_output=True, text=True, check=False
    )
    if result.returncode == 0:
        sys.stdout.write(result.stdout)
        return 0
    arm = platform.machine().lower() in {"aarch64", "arm64"}
    remaining = []
    for line in result.stdout.splitlines():
        match = _UNSUPPORTED.match(line.strip())
        if arm and match and _is_sbsa_wheel(match["name"], match["version"]):
            print(f"Ignoring SBSA platform-tag false positive: {line.strip()}")
            continue
        remaining.append(line)
    if remaining or result.stderr.strip():
        sys.stdout.write("\n".join(remaining) + "\n")
        sys.stderr.write(result.stderr)
        return 1
    print("No broken requirements found (SBSA tag false positives ignored).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
