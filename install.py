"""Install ManaEnv and its public-release source dependencies."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shlex
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
X2REAL_BRANCH = "x2real"
POLICY_SPACE_BRANCH = "main"
# Isaac Sim runtime installation remains NVIDIA pip; the gitlink is provenance only.
ISAACSIM_VERSION = "6.0.1.0"
ISAACSIM_INSTALL_METHOD = "pip"
NEWTON_PIP_SPEC = "newton[importers] @ git+https://github.com/X-Square-Robot/Newton.git@x2real"
NEWTON_SCHEMAS_PIP_SPEC = "newton-usd-schemas @ git+https://github.com/X-Square-Robot/Newton.git@x2real#subdirectory=extern/newton-usd-schemas"
DEFAULT_ISAACLAB_URL = "https://github.com/X-Square-Robot/IsaacLab.git"
DEFAULT_ISAACSIM_URL = "https://github.com/X-Square-Robot/IsaacSim.git"
DEFAULT_ARENA_URL = "https://github.com/X-Square-Robot/IsaacLabArena.git"
DEFAULT_POLICY_SPACE_URL = "https://github.com/X-Square-Robot/policy-space.git"
# The public IsaacLab x2real branch still pins Newton to the vendor's internal
# GitLab, which is unreachable outside their network. Rewrite those pins to the
# public Newton fork/branch that this installer installs anyway (NEWTON_PIP_SPEC).
INTERNAL_NEWTON_REF = "git+ssh://git@gitlab.zbl.local:50022/mana/engine/newton.git@publication"
PUBLIC_NEWTON_REF = "git+https://github.com/X-Square-Robot/Newton.git@x2real"
ISAACLAB_NEWTON_PIN_FILES = ("pyproject.toml", "tools/wheel_builder/uv-overrides.txt")
PYTORCH_INDEX_URL = "https://download.pytorch.org/whl"
# ``uv pip install`` ignores Arena/IsaacLab ``[tool.uv] override-dependencies``,
# so the resolver floats these past Isaac Sim's hard pins (llvmlite via numba,
# idna via openai>=3 -> httpx2, psutil/decorator via ipython>=9, packaging via
# virtualenv). Re-align them to versions that satisfy isaacsim-core/kernel and
# Arena together; validated with a clean ``pip check`` on aarch64 (DGX Spark).
RESOLVER_DRIFT_PINS = (
    "coacd==1.0.7",
    "llvmlite==0.46.0",
    "numba==0.64.0",
    "idna==3.10",
    "psutil==5.9.8",
    "packaging==26.0",
    "openai==2.54.0",
    "ipython==8.39.0",
    "decorator>=4.0.2,<5",
    "virtualenv==21.14.5",
)
# Left behind by openai>=3; nothing else requires it and it needs idna>=3.18.
RESOLVER_DRIFT_ORPHANS = ("httpx2",)
# conda-forge git-lfs (3.8.x) rewrites the tracked LFS hooks under Arena's
# ``.githooks/`` into its newer format, which is not a user source change.
ARENA_IGNORED_DIRTY_PREFIXES = (".githooks/",)
# PyPI's warp-lang 1.19.0 is a CUDA 13 build and needs an R580+ driver; on older
# drivers it fails with "Insufficient CUDA driver version" and then
# "Invalid device identifier: cuda:0". NVIDIA publishes the same release built
# against CUDA 12 on GitHub. sha256 values are the release asset digests.
WARP_CUDA13_MIN_DRIVER_MAJOR = 580
WARP_CU12_VERSION = "1.19.0"
WARP_CU12_RELEASE_URL = "https://github.com/NVIDIA/warp/releases/download/v{version}/{filename}"
WARP_CU12_WHEELS = {
    "x86_64": (
        "warp_lang-1.19.0+cu12-py3-none-manylinux_2_28_x86_64.whl",
        "f95d9581ed12663df272e45a50101b171e143fde1ef7b2433a8bfc9fba096598",
    ),
    "aarch64": (
        "warp_lang-1.19.0+cu12-py3-none-manylinux_2_34_aarch64.whl",
        "b3d0222132811b1db8c5c5d1d04bdc92923b77b7df8556d665116569a361a5d2",
    ),
}


class Installer:
    def __init__(self, *, dry_run: bool) -> None:
        self.dry_run = dry_run

    @staticmethod
    def _display(args: tuple[object, ...]) -> str:
        return shlex.join(str(arg) for arg in args)

    def run(self, *args: object, cwd: Path | None = None) -> None:
        print(f"+ {self._display(args)}")
        if not self.dry_run:
            subprocess.run([str(arg) for arg in args], cwd=cwd, check=True)

    @staticmethod
    def output(*args: object, cwd: Path | None = None) -> str:
        return subprocess.check_output(
            [str(arg) for arg in args], cwd=cwd, text=True
        ).strip()


def _is_git_checkout(path: Path) -> bool:
    if not path.is_dir():
        return False
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return False
    try:
        return Path(result.stdout.strip()).resolve() == path.resolve()
    except OSError:
        return False


def _git_output(installer: Installer, path: Path, *args: str) -> str:
    return installer.output("git", "-C", path, *args)


def _source_url(
    *,
    explicit: str | None,
    env_name: str,
    default: str,
) -> str:
    return explicit or os.environ.get(env_name) or default


def _policy_space_source_url(
    installer: Installer,
    *,
    explicit: str | None,
    path: Path,
) -> str:
    configured = explicit or os.environ.get("MANAENV_POLICY_SPACE_URL")
    if configured:
        return configured
    if _is_git_checkout(path):
        return _git_output(installer, path, "remote", "get-url", "origin")
    return DEFAULT_POLICY_SPACE_URL


def _require_clean_checkout(
    installer: Installer,
    path: Path,
    label: str,
    *,
    ignored_prefixes: tuple[str, ...] = (),
) -> None:
    # Do not strip: porcelain status lines start with a significant space
    # (" M path"), and stripping shifts the path offset of the first line.
    dirty = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ).rstrip("\n")
    if ignored_prefixes:
        dirty = "\n".join(
            line
            for line in dirty.splitlines()
            if not any(line[3:].startswith(prefix) for prefix in ignored_prefixes)
        )
    if dirty:
        raise RuntimeError(
            f"{label} checkout is dirty; preserve or clean it before installation: {path}"
        )


def _ensure_checkout(
    installer: Installer,
    *,
    label: str,
    url: str,
    path: Path,
    branch: str,
    update: bool,
    ignored_prefixes: tuple[str, ...] = (),
) -> None:
    if path.is_dir() and not any(path.iterdir()):
        if not installer.dry_run:
            path.rmdir()
    if not path.exists():
        if not installer.dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
        installer.run(
            "git",
            "clone",
            "--branch",
            branch,
            "--single-branch",
            url,
            path,
        )
        return
    if not _is_git_checkout(path):
        raise RuntimeError(f"{label} target exists but is not a Git checkout: {path}")

    actual_url = _git_output(installer, path, "remote", "get-url", "origin")
    if actual_url != url:
        raise RuntimeError(f"{label} origin mismatch: expected {url}, got {actual_url}")
    try:
        actual_branch = _git_output(
            installer, path, "symbolic-ref", "--quiet", "--short", "HEAD"
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"{label} must be checked out on branch {branch}: {path}"
        ) from exc
    if actual_branch != branch:
        raise RuntimeError(
            f"{label} must be on {branch}, got {actual_branch}: {path}"
        )
    _require_clean_checkout(installer, path, label, ignored_prefixes=ignored_prefixes)

    if update:
        installer.run("git", "fetch", "origin", branch, cwd=path)
        installer.run(
            "git", "merge", "--ff-only", f"origin/{branch}", cwd=path
        )
        if not installer.dry_run:
            head = _git_output(installer, path, "rev-parse", "HEAD")
            remote = _git_output(
                installer, path, "rev-parse", f"origin/{branch}"
            )
            if head != remote:
                raise RuntimeError(
                    f"{label} did not reach origin/{branch}: {path}"
                )


def _ensure_isaaclab_submodule(
    installer: Installer,
    *,
    arena_dir: Path,
    isaaclab_dir: Path,
    isaaclab_url: str,
    update: bool,
) -> None:
    if not _is_git_checkout(arena_dir):
        if installer.dry_run:
            installer.run(
                "git",
                "-C",
                arena_dir,
                "-c",
                f"submodule.submodules/IsaacLab.url={isaaclab_url}",
                "submodule",
                "update",
                "--init",
                "--checkout",
                "--depth",
                "1",
                "--",
                "submodules/IsaacLab",
            )
            return
        raise RuntimeError(f"Arena checkout is unavailable: {arena_dir}")

    configured_branch = _git_output(
        installer,
        arena_dir,
        "config",
        "-f",
        ".gitmodules",
        "--get",
        "submodule.submodules/IsaacLab.branch",
    )
    if configured_branch != X2REAL_BRANCH:
        raise RuntimeError(
            "Arena's IsaacLab submodule must track x2real; "
            f".gitmodules contains {configured_branch!r}."
        )
    if _is_git_checkout(isaaclab_dir):
        _require_clean_checkout(
            installer,
            isaaclab_dir,
            "IsaacLab",
            ignored_prefixes=("newton/", *ISAACLAB_NEWTON_PIN_FILES),
        )

    installer.run(
        "git", "submodule", "sync", "--", "submodules/IsaacLab", cwd=arena_dir
    )
    installer.run(
        "git",
        "config",
        "submodule.submodules/IsaacLab.url",
        isaaclab_url,
        cwd=arena_dir,
    )
    installer.run(
        "git",
        "-c",
        f"submodule.submodules/IsaacLab.url={isaaclab_url}",
        "submodule",
        "update",
        "--init",
        "--checkout",
        "--depth",
        "1",
        "--",
        "submodules/IsaacLab",
        cwd=arena_dir,
    )
    if installer.dry_run:
        return

    installer.run("git", "remote", "set-url", "origin", isaaclab_url, cwd=isaaclab_dir)
    gitlink = _git_output(installer, arena_dir, "rev-parse", "HEAD:submodules/IsaacLab")
    head = _git_output(installer, isaaclab_dir, "rev-parse", "HEAD")
    if head != gitlink:
        raise RuntimeError(
            f"IsaacLab checkout {head} does not match Arena gitlink {gitlink}"
        )
    if update:
        installer.run("git", "fetch", "origin", X2REAL_BRANCH, cwd=isaaclab_dir)
        contained = subprocess.run(
            [
                "git",
                "-C",
                str(isaaclab_dir),
                "merge-base",
                "--is-ancestor",
                head,
                "FETCH_HEAD",
            ],
            check=False,
        )
        if contained.returncode != 0:
            raise RuntimeError(
                f"Arena's IsaacLab gitlink is not from origin/{X2REAL_BRANCH}: {head}"
            )


def _validate_isaacsim_gitlink(
    installer: Installer,
    *,
    isaaclab_dir: Path,
    isaacsim_url: str,
) -> None:
    if installer.dry_run:
        return
    if not _is_git_checkout(isaaclab_dir):
        raise RuntimeError(f"IsaacLab checkout is unavailable: {isaaclab_dir}")

    configured_url = _git_output(
        installer,
        isaaclab_dir,
        "config",
        "-f",
        ".gitmodules",
        "--get",
        "submodule.IsaacSim.url",
    )
    if configured_url != isaacsim_url:
        raise RuntimeError(
            "IsaacLab's IsaacSim gitlink URL mismatch: "
            f"expected {isaacsim_url}, got {configured_url}"
        )

    configured_branch = _git_output(
        installer,
        isaaclab_dir,
        "config",
        "-f",
        ".gitmodules",
        "--get",
        "submodule.IsaacSim.branch",
    )
    if configured_branch != X2REAL_BRANCH:
        raise RuntimeError(
            "IsaacLab's IsaacSim gitlink must track x2real; "
            f".gitmodules contains {configured_branch!r}."
        )


def _rewrite_internal_newton_pins(installer: Installer, isaaclab_dir: Path) -> None:
    """Point IsaacLab's Newton pins at the public fork instead of internal GitLab."""
    for relative in ISAACLAB_NEWTON_PIN_FILES:
        path = isaaclab_dir / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if INTERNAL_NEWTON_REF not in text:
            continue
        print(f"+ rewrite {INTERNAL_NEWTON_REF} -> {PUBLIC_NEWTON_REF} in {path}")
        if not installer.dry_run:
            path.write_text(text.replace(INTERNAL_NEWTON_REF, PUBLIC_NEWTON_REF), encoding="utf-8")


def _ensure_torchaudio(installer: Installer, python: Path) -> None:
    """Restore torchaudio when IsaacLab's torch reinstall dropped it.

    IsaacLab uninstalls torch/torchvision/torchaudio and reinstalls only the
    first two, while isaacsim-core requires torchaudio, so ``pip check`` fails.
    Install the build matching the installed torch's CUDA tag (cu128 on x86_64,
    cu130 on aarch64).
    """
    probe = subprocess.run(
        [str(python), "-c",
         "from importlib.metadata import version, PackageNotFoundError\n"
         "try:\n    version('torchaudio'); print('present')\n"
         "except PackageNotFoundError:\n    print(version('torch'))"],
        capture_output=True, text=True, check=False,
    )
    result = probe.stdout.strip()
    if probe.returncode != 0 or not result or result == "present":
        return
    base, _, cuda_tag = result.partition("+")
    command: list[object] = [python, "-m", "pip", "install", "--no-deps", f"torchaudio=={result}"]
    if cuda_tag:
        command += ["--index-url", f"{PYTORCH_INDEX_URL}/{cuda_tag}"]
    installer.run(*command)


def _align_resolver_drift(installer: Installer, python: Path) -> None:
    """Pin packages that drifted past Isaac Sim's requirements; drop orphans."""
    installer.run(python, "-m", "pip", "install", *RESOLVER_DRIFT_PINS)
    probe = (
        "import sys, re\n"
        "from importlib.metadata import distributions, PackageNotFoundError, version\n"
        "names = set(sys.argv[1:])\n"
        "needed = set()\n"
        "for d in distributions():\n"
        "    for r in d.requires or []:\n"
        "        if 'extra ==' in r: continue\n"
        "        n = re.split(r'[ <>=!~;\\[(]', r, 1)[0].lower().replace('_', '-')\n"
        "        if n in names: needed.add(n)\n"
        "for n in sorted(names - needed):\n"
        "    try:\n"
        "        version(n); print(n)\n"
        "    except PackageNotFoundError:\n"
        "        pass\n"
    )
    orphans = subprocess.run(
        [str(python), "-c", probe, *RESOLVER_DRIFT_ORPHANS],
        capture_output=True, text=True, check=False,
    ).stdout.split()
    if orphans:
        installer.run(python, "-m", "pip", "uninstall", "-y", *orphans)


def _nvidia_driver_major() -> int | None:
    """Lowest NVIDIA driver major version across GPUs, or None when unknown.

    ``MANAENV_NVIDIA_DRIVER_VERSION`` (e.g. ``570.211.01``) overrides the
    ``nvidia-smi`` query, for containers without it and for testing.
    """
    lines = [os.environ.get("MANAENV_NVIDIA_DRIVER_VERSION", "")]
    if not lines[0]:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, check=False, timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0:
            return None
        lines = result.stdout.splitlines()
    majors = [
        int(head)
        for head in (line.strip().split(".", 1)[0] for line in lines)
        if head.isdigit()
    ]
    return min(majors) if majors else None


def _warp_cu12_wheel(
    *, driver_major: int | None, machine: str, installed_warp: str | None
) -> tuple[str, str] | None:
    """Return ``(filename, sha256)`` of the CUDA 12 warp build to install, if any.

    ``None`` keeps the installed warp-lang: the driver is unknown or R580+,
    warp-lang is absent, or a CUDA 12 build is already installed. A pre-R580
    driver with a warp-lang version/architecture that has no verified CUDA 12
    wheel here only warns, so the rest of the installation is kept.
    """
    if driver_major is None or driver_major >= WARP_CUDA13_MIN_DRIVER_MAJOR:
        return None
    if installed_warp is None:
        return None
    base, _, local = installed_warp.partition("+")
    if local.startswith("cu12"):
        return None
    machine = {"amd64": "x86_64", "arm64": "aarch64"}.get(machine.lower(), machine.lower())
    wheel = WARP_CU12_WHEELS.get(machine)
    if base != WARP_CU12_VERSION or wheel is None:
        print(
            f"[WARNING] NVIDIA driver {driver_major} is older than R{WARP_CUDA13_MIN_DRIVER_MAJOR}, "
            f"but no verified CUDA 12 build of warp-lang {installed_warp} for {machine} is known. "
            "If Warp reports 'Insufficient CUDA driver version', install the matching "
            "'+cu12' wheel from https://github.com/NVIDIA/warp/releases with "
            "'pip install --no-deps --force-reinstall'.",
            file=sys.stderr,
        )
        return None
    return wheel


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _ensure_warp_matches_driver(installer: Installer, python: Path) -> None:
    """Swap PyPI's CUDA 13 warp-lang for NVIDIA's CUDA 12 build on pre-R580 drivers."""
    driver_major = _nvidia_driver_major()
    probe = subprocess.run(
        [str(python), "-c",
         "from importlib.metadata import version, PackageNotFoundError\n"
         "try:\n    print(version('warp-lang'))\n"
         "except PackageNotFoundError:\n    pass"],
        capture_output=True, text=True, check=False,
    )
    installed_warp = probe.stdout.strip() or None
    wheel = _warp_cu12_wheel(
        driver_major=driver_major, machine=platform.machine(), installed_warp=installed_warp
    )
    if wheel is None:
        return
    filename, expected_sha256 = wheel
    print(
        f"NVIDIA driver {driver_major} < R{WARP_CUDA13_MIN_DRIVER_MAJOR}: replace warp-lang "
        f"{installed_warp} (CUDA 13) with {filename}"
    )
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "manaenv" / "warp-cu12"
    wheel_path = cache / filename
    if not (wheel_path.is_file() and _sha256(wheel_path) == expected_sha256):
        url = WARP_CU12_RELEASE_URL.format(
            version=WARP_CU12_VERSION, filename=urllib.parse.quote(filename)
        )
        print(f"+ download {url} -> {wheel_path}")
        if not installer.dry_run:
            cache.mkdir(parents=True, exist_ok=True)
            partial = wheel_path.with_name(wheel_path.name + ".part")
            with urllib.request.urlopen(url, timeout=300) as response, partial.open("wb") as out:
                shutil.copyfileobj(response, out, 1 << 20)
            actual_sha256 = _sha256(partial)
            if actual_sha256 != expected_sha256:
                partial.unlink()
                raise RuntimeError(
                    f"{filename}: sha256 {actual_sha256} does not match the release digest "
                    f"{expected_sha256}"
                )
            partial.replace(wheel_path)
    installer.run(python, "-m", "pip", "install", "--no-deps", "--force-reinstall", wheel_path)


def _verify_python(python: Path) -> None:
    if not python.is_file() or not os.access(python, os.X_OK):
        raise RuntimeError(f"Python interpreter is not executable: {python}")
    version = subprocess.check_output(
        [
            str(python),
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ],
        text=True,
    ).strip()
    if version != "3.12":
        raise RuntimeError(f"ManaEnv requires Python 3.12; {python} reports {version}")


def _verify_prerequisites(python: Path, *, install_sources: bool) -> None:
    modules = (
        ("isaacsim",)
        if install_sources
        else (
            "isaacsim",
            "newton",
            "newton_usd_schemas",
            "isaaclab",
            "isaaclab_arena",
            "isaaclab_visualizers.kit",
            "policy_space_protocol",
        )
    )
    code = """
import importlib.util
import sys
from importlib.metadata import PackageNotFoundError, version

expected_isaacsim = sys.argv[1]
try:
    installed_isaacsim = version("isaacsim")
except PackageNotFoundError:
    raise SystemExit("[ERROR] Missing simulation prerequisite: isaacsim")
if installed_isaacsim != expected_isaacsim:
    raise SystemExit(
        f"[ERROR] Isaac Sim {expected_isaacsim} is required; found {installed_isaacsim}"
    )

missing = [name for name in sys.argv[2:] if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("[ERROR] Missing simulation prerequisites: " + ", ".join(missing))
"""
    subprocess.run([str(python), "-c", code, ISAACSIM_VERSION, *modules], check=True)


def _isaaclab_targets(isaaclab_dir: Path) -> list[Path]:
    source_root = isaaclab_dir / "source"
    if not source_root.is_dir():
        raise RuntimeError(f"IsaacLab source directory is missing: {source_root}")
    targets = sorted(
        path
        for path in source_root.iterdir()
        if path.is_dir()
        and ((path / "pyproject.toml").is_file() or (path / "setup.py").is_file())
    )
    if not targets:
        raise RuntimeError(
            f"No installable IsaacLab packages found under {source_root}"
        )
    return targets


def _install_editables(
    installer: Installer,
    *,
    python: Path,
    arena_dir: Path,
    isaaclab_dir: Path,
    policy_space_dir: Path,
    install_sources: bool,
) -> None:
    if install_sources:
        installer.run(
            python,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            NEWTON_PIP_SPEC,
            NEWTON_SCHEMAS_PIP_SPEC,
        )
        if installer.dry_run and not isaaclab_dir.is_dir():
            print(
                "+ "
                f"{shlex.quote(str(python))} -m pip install --no-deps --editable "
                "<IsaacLab source packages> --editable <Arena> "
                "--editable <Policy Space protocol>"
            )
        else:
            targets = _isaaclab_targets(isaaclab_dir)
            targets.append(arena_dir)
            targets.append(policy_space_dir / "packages" / "protocol")
            command: list[object] = [python, "-m", "pip", "install", "--no-deps"]
            for target in targets:
                command.extend(("--editable", target))
            installer.run(*command)
    installer.run(python, "-m", "pip", "install", "--editable", ROOT)
    _ensure_torchaudio(installer, python)
    _align_resolver_drift(installer, python)
    installer.run(python, "-P", ROOT / "manaenv/scripts/x2real_runtime_wheels.py", "--install")
    _ensure_warp_matches_driver(installer, python)


def _verify_install(
    python: Path,
    *,
    arena_dir: Path,
    isaaclab_dir: Path,
    policy_space_dir: Path,
    install_sources: bool,
) -> None:
    code = """
import importlib
import sys
from importlib.metadata import version
from pathlib import Path

installed_isaacsim = version("isaacsim")
if installed_isaacsim != sys.argv[1]:
    raise SystemExit(
        f"[ERROR] Isaac Sim {sys.argv[1]} is required; found {installed_isaacsim}"
    )
modules = (
    "isaacsim",
    "newton",
    "newton_usd_schemas",
    "isaaclab",
    "isaaclab_arena",
    "isaaclab_visualizers.kit",
    "policy_space_protocol",
    "manaenv",
)
loaded = {name: importlib.import_module(name) for name in modules}
if sys.argv[2] == "true":
    expected = {
        "isaaclab": Path(sys.argv[3]).resolve(),
        "isaaclab_arena": Path(sys.argv[4]).resolve(),
        "policy_space_protocol": Path(sys.argv[5]).resolve(),
        "manaenv": Path(sys.argv[6]).resolve(),
    }
    for name, root in expected.items():
        location = Path(loaded[name].__file__).resolve()
        if not location.is_relative_to(root):
            raise SystemExit(f"[ERROR] {name} resolved outside {root}: {location}")
print("ManaEnv publication source install: OK")
"""
    subprocess.run(
        [
            str(python),
            "-c",
            code,
            ISAACSIM_VERSION,
            str(install_sources).lower(),
            str(isaaclab_dir),
            str(arena_dir),
            str(policy_space_dir),
            str(ROOT),
        ],
        check=True,
    )
    subprocess.run(
        [str(python), "-P", str(ROOT / "manaenv/scripts/x2real_pip_check.py")], check=True
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="target Python 3.12 interpreter",
    )
    parser.add_argument(
        "--arena-url", help="override the public IsaacLab Arena x2real URL"
    )
    parser.add_argument(
        "--isaaclab-url", help="override the public IsaacLab x2real URL"
    )
    parser.add_argument(
        "--isaacsim-url", help="override the public Isaac Sim x2real URL"
    )
    parser.add_argument(
        "--policy-space-url",
        help="override the GitHub Policy Space main-branch URL",
    )
    parser.add_argument(
        "--skip-source-update",
        action="store_true",
        help="do not fetch existing source checkouts",
    )
    parser.add_argument(
        "--skip-sources",
        action="store_true",
        help="use already-installed simulation packages",
    )
    parser.add_argument(
        "--sources-only",
        action="store_true",
        help="prepare dependency source checkouts without pip installs",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="skip prerequisite and final import checks",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print mutating commands without running them",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.skip_sources and args.sources_only:
        raise RuntimeError("--skip-sources and --sources-only cannot be used together")

    python = args.python.expanduser().resolve()
    if not args.sources_only:
        _verify_python(python)
    installer = Installer(dry_run=args.dry_run)
    deps_root = (
        Path(os.environ.get("MANAENV_DEPS_ROOT", ROOT / "3rd")).expanduser().resolve()
    )
    arena_dir = deps_root / "isaaclabarena"
    isaaclab_dir = arena_dir / "submodules" / "IsaacLab"
    policy_space_dir = deps_root / "policy_space"

    if not args.skip_sources:
        isaacsim_url = _source_url(
            explicit=args.isaacsim_url,
            env_name="MANAENV_ISAACSIM_URL",
            default=DEFAULT_ISAACSIM_URL,
        )
        arena_url = _source_url(
            explicit=args.arena_url,
            env_name="MANAENV_ARENA_URL",
            default=DEFAULT_ARENA_URL,
        )
        isaaclab_url = _source_url(
            explicit=args.isaaclab_url,
            env_name="MANAENV_ISAACLAB_URL",
            default=DEFAULT_ISAACLAB_URL,
        )
        policy_space_url = _policy_space_source_url(
            installer,
            explicit=args.policy_space_url,
            path=policy_space_dir,
        )
        update = not args.skip_source_update
        _ensure_checkout(
            installer,
            label="Arena",
            url=arena_url,
            path=arena_dir,
            branch=X2REAL_BRANCH,
            update=update,
            ignored_prefixes=ARENA_IGNORED_DIRTY_PREFIXES,
        )
        _ensure_checkout(
            installer,
            label="Policy Space",
            url=policy_space_url,
            path=policy_space_dir,
            branch=POLICY_SPACE_BRANCH,
            update=update,
        )
        _ensure_isaaclab_submodule(
            installer,
            arena_dir=arena_dir,
            isaaclab_dir=isaaclab_dir,
            isaaclab_url=isaaclab_url,
            update=update,
        )
        _validate_isaacsim_gitlink(
            installer,
            isaaclab_dir=isaaclab_dir,
            isaacsim_url=isaacsim_url,
        )
        _rewrite_internal_newton_pins(installer, isaaclab_dir)

    if args.sources_only:
        print("Dependency source preparation complete.")
        return 0
    if not args.skip_verify and not args.dry_run:
        _verify_prerequisites(python, install_sources=not args.skip_sources)
    _install_editables(
        installer,
        python=python,
        arena_dir=arena_dir,
        isaaclab_dir=isaaclab_dir,
        policy_space_dir=policy_space_dir,
        install_sources=not args.skip_sources,
    )
    if not args.skip_verify and not args.dry_run:
        _verify_install(
            python,
            arena_dir=arena_dir,
            isaaclab_dir=isaaclab_dir,
            policy_space_dir=policy_space_dir,
            install_sources=not args.skip_sources,
        )
    print("ManaEnv publication install complete.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
