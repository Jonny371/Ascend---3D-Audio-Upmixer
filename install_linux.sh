#!/bin/sh
# Ascend - Linux installer.  Run:  ./install_linux.sh
cd "$(dirname "$0")" || exit 1
echo
echo " Ascend - Linux installer"
echo " ========================"
echo

# Detect the distro package manager and install Python + venv + libsndfile + ffmpeg
# if they are missing.  ffmpeg/libsndfile are best-effort; the app still runs with
# the bundled imageio-ffmpeg if ffmpeg is absent.
PM=""; PKGS=""
if command -v apt-get >/dev/null 2>&1; then
  PM="sudo apt-get install -y"; UPD="sudo apt-get update"
  PKGS="python3 python3-venv python3-pip libsndfile1 ffmpeg libgl1 libxkbcommon0"
elif command -v dnf >/dev/null 2>&1; then
  PM="sudo dnf install -y"; UPD=""
  PKGS="python3 python3-pip libsndfile ffmpeg mesa-libGL libxkbcommon"
elif command -v pacman >/dev/null 2>&1; then
  PM="sudo pacman -S --noconfirm --needed"; UPD=""
  PKGS="python python-pip libsndfile ffmpeg libglvnd libxkbcommon"
elif command -v zypper >/dev/null 2>&1; then
  PM="sudo zypper install -y"; UPD=""
  PKGS="python3 python3-pip libsndfile1 ffmpeg Mesa-libGL1 libxkbcommon0"
fi

if [ -n "$PM" ]; then
  echo " Installing system components (you may be asked for your password):"
  echo "   $PKGS"
  [ -n "$UPD" ] && $UPD
  $PM $PKGS || echo " (some packages may already be installed - continuing)"
else
  echo " Unknown package manager - please make sure python3, python3-venv, pip,"
  echo " and libsndfile are installed, then continue."
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo " Python 3 is still not available. Please install it and re-run."
  exit 1
fi

echo " Using: python3"
python3 install.py
