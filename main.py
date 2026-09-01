"""AdLens server entry point."""

import os
import shutil
import sys
from pathlib import Path


def _bootstrap_ffmpeg_path() -> None:
    """
    Ensure ffmpeg is on PATH.

    winget installs ffmpeg but requires a new shell session for PATH changes
    to take effect.  This function finds common winget/scoop install locations
    and adds them to os.environ["PATH"] at process start so the current
    process and all subprocesses can find ffmpeg without a shell restart.
    """
    if shutil.which("ffmpeg"):
        return  # Already on PATH

    # Common winget install locations on Windows
    candidates = [
        # winget default package location
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages",
        # Chocolatey
        Path("C:/ProgramData/chocolatey/bin"),
        # Scoop
        Path(os.environ.get("USERPROFILE", "")) / "scoop" / "shims",
        # Manual install
        Path("C:/ffmpeg/bin"),
        Path("C:/Program Files/ffmpeg/bin"),
    ]

    for base in candidates:
        if not base.exists():
            continue
        # winget installs into a versioned subdirectory — search recursively
        for ffmpeg_exe in base.rglob("ffmpeg.exe"):
            ffmpeg_dir = str(ffmpeg_exe.parent)
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
            print(f"[startup] Added FFmpeg to PATH: {ffmpeg_dir}", flush=True)
            return

    print(
        "[startup] WARNING: ffmpeg not found on PATH. "
        "Install with: winget install Gyan.FFmpeg (then restart shell, "
        "or set ADLENS_FFMPEG_DIR in .env)",
        file=sys.stderr,
    )


_bootstrap_ffmpeg_path()

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "adlens.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # reload=True interferes with PATH bootstrap
        log_level="info",
    )

