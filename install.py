#!/usr/bin/env python3
"""
Ascend installer — cross-platform.

Run me with Python 3.9+ (the bootstrap scripts make sure you have it):

    Windows : double-click  install_windows.bat
    macOS   : double-click  install_macos.command
    Linux   : run           ./install_linux.sh

…or, if you already have Python on PATH, just:  python install.py

What I do, regardless of OS:
  1. check the Python version,
  2. create a private virtual environment (.venv) next to this file,
  3. install everything Ascend needs into it (numpy, scipy, soundfile,
     PySide6, imageio-ffmpeg) — nothing touches your system Python,
  4. check for ffmpeg (optional; a bundled build is used if it's missing),
  5. write a one-click launcher for your platform.
"""
import os
import sys
import shutil
import platform
import subprocess
from pathlib import Path

APP = Path(__file__).resolve().parent
VENV = APP / ".venv"
REQ = APP / "requirements.txt"
GUI = APP / "ascend_gui.py"
MIN_PY = (3, 9)
PKGS = ["numpy>=1.22", "scipy>=1.8", "soundfile>=0.12",
        "PySide6>=6.4", "imageio-ffmpeg>=0.4.9"]


def step(msg):
    print("  • " + msg, flush=True)


def fail(msg):
    print("\nInstallation stopped: " + msg + "\n", flush=True)
    sys.exit(1)


def venv_python():
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def create_venv():
    marker = VENV / ("Scripts" if os.name == "nt" else "bin")
    if marker.exists() and venv_python().exists():
        step("Virtual environment already present (.venv) — reusing it.")
        return
    step("Creating a private virtual environment (.venv) …")
    try:
        import venv as _venv
        _venv.EnvBuilder(with_pip=True, clear=False, upgrade_deps=False).create(str(VENV))
    except Exception as e:
        # Some distros ship venv without pip bootstrap; fall back to virtualenv.
        step(f"venv module failed ({e}); trying a plain python -m venv …")
        r = subprocess.run([sys.executable, "-m", "venv", str(VENV)])
        if r.returncode != 0:
            fail("could not create a virtual environment. On Debian/Ubuntu install "
                 "'python3-venv' (sudo apt-get install python3-venv) and re-run.")
    if not venv_python().exists():
        fail("the virtual environment was created without a Python executable.")


def pip_install():
    py = str(venv_python())
    step("Upgrading pip inside the environment …")
    subprocess.run([py, "-m", "pip", "install", "--upgrade", "pip"],
                   stdout=subprocess.DEVNULL)
    step("Installing dependencies — this can take a few minutes the first time …")
    if REQ.exists():
        args = [py, "-m", "pip", "install", "-r", str(REQ)]
    else:
        args = [py, "-m", "pip", "install", *PKGS]
    r = subprocess.run(args)
    if r.returncode != 0:
        fail("a dependency failed to install. Check your internet connection and "
             "re-run. If PySide6 fails on Linux, install your distro's Qt/OpenGL "
             "runtime libraries (e.g. libgl1, libxkbcommon0) and try again.")


def check_ffmpeg():
    if shutil.which("ffmpeg"):
        step("System ffmpeg found — full format support enabled.")
    else:
        step("No system ffmpeg on PATH; the bundled imageio-ffmpeg build will be "
             "used for compressed formats (MP3/AAC/M4A/AC-3/…).")


def write_launcher():
    py = venv_python()
    if os.name == "nt":
        runpy = VENV / "Scripts/pythonw.exe"
        if not runpy.exists():
            runpy = py
        launcher = APP / "Ascend.bat"
        launcher.write_text(
            "@echo off\r\n"
            'cd /d "%~dp0"\r\n'
            f'start "" "{runpy}" "%~dp0ascend_gui.py" %*\r\n',
            encoding="utf-8")
        name = "Ascend.bat"
    else:
        is_mac = sys.platform == "darwin"
        launcher = APP / ("Ascend.command" if is_mac else "Ascend.sh")
        launcher.write_text(
            "#!/bin/sh\n"
            'cd "$(dirname "$0")"\n'
            f'exec "{py}" "ascend_gui.py" "$@"\n',
            encoding="utf-8")
        os.chmod(launcher, 0o755)
        name = launcher.name
    step(f"Created launcher: {name}")
    return name


def main():
    print("\nAscend installer")
    print("================")
    if sys.version_info < MIN_PY:
        fail(f"Python {MIN_PY[0]}.{MIN_PY[1]}+ is required; this is "
             f"{platform.python_version()}. Please install a newer Python and re-run.")
    if not GUI.exists():
        fail(f"ascend_gui.py was not found next to this installer "
             f"(looked in {APP}). Keep all the files together and re-run.")
    step(f"Python {platform.python_version()} · {platform.system()} {platform.machine()}")
    create_venv()
    pip_install()
    check_ffmpeg()
    name = write_launcher()
    print("\nAll set!  Launch Ascend by ", end="")
    if os.name == "nt":
        print(f'double-clicking "{name}" in this folder.')
    elif sys.platform == "darwin":
        print(f'double-clicking "{name}" in this folder.\n'
              "(First time: right-click → Open, to get past Gatekeeper.)")
    else:
        print(f'running "./{name}" in this folder.')
    print()


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        fail(f"a command failed: {e}")
    except KeyboardInterrupt:
        fail("cancelled.")
