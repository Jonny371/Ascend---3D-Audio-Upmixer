#!/bin/sh
# Ascend - macOS installer.  Double-click this file (or: right-click -> Open).
cd "$(dirname "$0")" || exit 1
echo
echo " Ascend - macOS installer"
echo " ========================"
echo

if command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  echo " Python 3 was not found."
  if command -v brew >/dev/null 2>&1; then
    echo " Installing Python 3 with Homebrew ..."
    brew install python && PY=python3
  fi
  if [ -z "$PY" ]; then
    echo " Please install Python 3 from https://www.python.org/downloads/macos/"
    echo " (or install Homebrew from https://brew.sh and re-run), then open this file again."
    echo
    printf " Press Return to close. "; read _
    exit 1
  fi
fi

# ffmpeg is optional (a bundled build is used otherwise); install it if brew is here.
if ! command -v ffmpeg >/dev/null 2>&1 && command -v brew >/dev/null 2>&1; then
  echo " Installing ffmpeg with Homebrew (optional, for extra formats) ..."
  brew install ffmpeg || true
fi

echo " Using: $PY"
"$PY" install.py
echo
printf " Press Return to close. "; read _
