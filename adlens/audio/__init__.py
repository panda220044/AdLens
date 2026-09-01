"""Audio extraction from video files using ffmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path

from adlens.utils.logger import get_logger

logger = get_logger(__name__)


class AudioExtractor:
    """
    Extracts audio from a video file to a WAV file using ffmpeg.

    WAV is preferred over MP3 for local Whisper models (lossless,
    no decoding overhead).  The OpenAI Whisper API accepts MP3/WAV/M4A.

    The output file is placed alongside the video in the work directory.
    """

    def extract(self, video_path: str, output_dir: str | None = None) -> str:
        """
        Extract audio from *video_path* and return the WAV file path.

        Parameters
        ----------
        video_path:
            Absolute path to the source video file.
        output_dir:
            Directory for the output WAV.  Defaults to the same directory
            as *video_path*.
        """
        src = Path(video_path)
        dst_dir = Path(output_dir) if output_dir else src.parent
        dst = dst_dir / (src.stem + "_audio.wav")

        if dst.exists():
            logger.info("Audio already extracted: %s", dst)
            return str(dst)

        cmd = [
            "ffmpeg",
            "-y",                    # overwrite if exists
            "-i", str(src),
            "-vn",                   # no video
            "-acodec", "pcm_s16le",  # uncompressed PCM
            "-ar", "16000",          # 16 kHz — optimal for Whisper
            "-ac", "1",              # mono
            str(dst),
        ]

        logger.info("Extracting audio: %s → %s", src.name, dst.name)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg audio extraction failed:\n{result.stderr}"
            )

        logger.info("Audio extracted: %s (%.1f MB)", dst.name, dst.stat().st_size / 1e6)
        return str(dst)
