from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / (".venv-win" if os.name == "nt" else ".venv")

def venv_python() -> Path:
    """Return the venv python executable path for the current OS."""
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _get_python_version(exe: str) -> str:
    try:
        res = subprocess.run(
            [exe, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            capture_output=True,
            text=True,
            check=True
        )
        return res.stdout.strip()
    except Exception:
        return ""


def _remove_venv(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except OSError as exc:
        raise RuntimeError(
            f"Failed to remove broken venv at {path}. "
            "Delete it manually and re-run."
        ) from exc


def ensure_venv() -> None:
    """Create venv if missing."""
    if not VENV_DIR.exists():
        print(f"--- Creating virtualenv at {VENV_DIR} ---")
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)], cwd=str(ROOT))

    py = venv_python()
    if not py.exists():
        raise SystemExit(
            f"Venv python not found at:\n  {py}\n\n"
            "This usually happens when the venv creation failed or you are using a Python distribution "
            "(e.g., MSYS2) that behaves differently.\n"
            "Try running with a standard Windows Python from python.org, or delete the venv and re-run."
        )


def install_requirements() -> None:
    ensure_venv()
    py = venv_python()

    req = ROOT / "requirements.txt"  # keep it simple; avoid .lock on Windows unless per-version
    if not req.exists():
        raise SystemExit(f"Missing requirements.txt at {req}")

    print(f"--- Installing dependencies from {req.name} ---")
    subprocess.check_call([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], cwd=str(ROOT))
    subprocess.check_call([str(py), "-m", "pip", "install", "-r", str(req)], cwd=str(ROOT))


def run_pipeline() -> None:
    # Run for both levels if not specified
    levels = ["hour", "day"]
    env_level = os.getenv("BIKE_LEVEL")
    if env_level:
        levels = [env_level.strip().lower()]

    for level in levels:
        print(f"--- Running pipeline for level: {level} ---")
        env = os.environ.copy()
        env["BIKE_LEVEL"] = level
        subprocess.check_call([str(venv_python()), "src/pipeline.py"], cwd=ROOT, env=env)


def run_streamlit() -> None:
    print("--- Launching Streamlit app ---")
    # Using Popen or check_call? If it's the last thing, check_call is fine.
    # However, sometimes users want to run it in the background.
    # The user said "after that open the streamlit app", implying sequential.
    try:
        subprocess.check_call([str(venv_python()), "-m", "streamlit", "run", "app.py"], cwd=ROOT)
    except KeyboardInterrupt:
        print("\n--- Streamlit stopped by user ---")


def find_compatible_python() -> str:
    """Find a Python version compatible with Matplotlib (avoiding 3.14)."""
    current_version = sys.version_info[:2]
    if current_version != (3, 14):
        return sys.executable

    # If we are on 3.14, try to find a stable one
    candidates = ["python3.13", "python3.12", "python3.11", "python3.10", "python3.9", "python3", "python"]
    for cand in candidates:
        try:
            # Check version of candidate
            ver = _get_python_version(cand)
            if ver and ver != "3.14":
                return shutil.which(cand) or cand
        except Exception:
            continue

    return sys.executable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--install-only", action="store_true", help="Only setup venv and install requirements")
    args = parser.parse_args()

    # Re-spawn with compatible python if necessary
    compatible_exe = find_compatible_python()
    if compatible_exe != sys.executable:
        print(f"--- Switching from Python {_get_python_version(sys.executable)} to {compatible_exe} for stability ---")
        os.execv(compatible_exe, [compatible_exe] + sys.argv)

    ensure_venv()
    install_requirements()
    
    if not args.install_only:
        run_pipeline()
        run_streamlit()


if __name__ == "__main__":
    if sys.version_info < (3, 10):
        raise SystemExit("This project requires Python 3.10+. Please install Python 3.10 or newer.")
    main()
