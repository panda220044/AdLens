"""
Video ingestion layer.

Provides an abstract VideoIngester and concrete implementations:
  - YouTubeIngester   — downloads via yt-dlp
  - LocalFileIngester — reads local files

Factory function `get_ingester(source)` selects the right implementation.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path

from adlens.models import Platform, VideoKind, VideoSource
from adlens.utils.config import settings
from adlens.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class VideoIngester(ABC):
    """
    Abstract base class for all video ingesters.

    Subclasses must implement `ingest()` which downloads/copies the video
    to the work directory and returns a VideoSource with a populated
    `local_path`.
    """

    @abstractmethod
    def ingest(self, source: str) -> VideoSource:
        """
        Ingest the video from *source* (URL or path).

        Returns a VideoSource with:
          - local_path set to the downloaded/resolved file
          - platform, kind, duration_s, title populated
        """
        ...

    @staticmethod
    def _probe_duration(path: str) -> float:
        """Use ffprobe to get video duration in seconds."""
        import subprocess
        import json

        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "quiet",
                    "-print_format", "json",
                    "-show_format",
                    path,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            data = json.loads(result.stdout)
            return float(data.get("format", {}).get("duration", 0))
        except Exception as exc:
            logger.warning("ffprobe failed: %s", exc)
            return 0.0


# ---------------------------------------------------------------------------
# YouTube ingester
# ---------------------------------------------------------------------------


class YouTubeIngester(VideoIngester):
    """
    Downloads a YouTube video (or Short) using yt-dlp.

    The video and audio are merged into a single .mp4 file in the
    work directory.  Audio is also extracted separately as a .wav
    for the ASR pipeline.
    """

    _YOUTUBE_SHORT_RE = re.compile(
        r"youtube\.com/shorts/|youtu\.be/[A-Za-z0-9_-]{11}"
    )
    _YOUTUBE_RE = re.compile(r"youtube\.com|youtu\.be")

    def ingest(self, source: str) -> VideoSource:
        """Download from YouTube and return a populated VideoSource."""
        try:
            import yt_dlp  # type: ignore
        except ImportError:
            raise RuntimeError(
                "yt-dlp is not installed. Run: pip install yt-dlp"
            )

        work_dir = settings.work_path
        is_short = bool(self._YOUTUBE_SHORT_RE.search(source))
        kind = VideoKind.short if is_short else VideoKind.vod

        # Output template — unique per URL to avoid collisions
        import hashlib
        url_hash = hashlib.md5(source.encode()).hexdigest()[:8]
        out_template = str(work_dir / f"yt_{url_hash}.%(ext)s")

        ydl_opts: dict = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": out_template,
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            # Extract subtitles for additional linguistic signals
            "writesubtitles": False,
            "postprocessors": [],
        }

        logger.info("Downloading: %s", source)
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(source, download=True)

        title = info.get("title", "")
        duration_s = float(info.get("duration") or 0)

        # Find the downloaded file
        video_path = str(work_dir / f"yt_{url_hash}.mp4")
        if not Path(video_path).exists():
            # yt-dlp may have chosen a different extension
            candidates = list(work_dir.glob(f"yt_{url_hash}.*"))
            if candidates:
                video_path = str(candidates[0])
            else:
                raise FileNotFoundError(
                    f"Downloaded file not found for hash {url_hash}"
                )

        if duration_s == 0:
            duration_s = self._probe_duration(video_path)

        logger.info("Downloaded → %s (%.1fs)", video_path, duration_s)

        return VideoSource(
            url=source,
            local_path=video_path,
            platform=Platform.youtube,
            kind=kind,
            duration_s=duration_s,
            title=title,
        )


# ---------------------------------------------------------------------------
# Local file ingester
# ---------------------------------------------------------------------------


class LocalFileIngester(VideoIngester):
    """
    Ingests a local video file.

    Used for:
      - Manually downloaded Instagram Reels
      - Any local .mp4 / .mov / .avi file

    Platform is set to Platform.instagram if the filename contains
    common Instagram export patterns, otherwise Platform.file.
    """

    _INSTAGRAM_PATTERNS = re.compile(
        r"instagram|reel|ig_|reels", re.IGNORECASE
    )

    def ingest(self, source: str) -> VideoSource:
        """Verify and describe a local video file."""
        path = Path(source).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Local video file not found: {path}")

        if not path.is_file():
            raise ValueError(f"Not a file: {path}")

        suffix = path.suffix.lower()
        if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}:
            raise ValueError(f"Unsupported video format: {suffix}")

        # Determine platform from filename heuristics
        platform = (
            Platform.instagram
            if self._INSTAGRAM_PATTERNS.search(path.stem)
            else Platform.file
        )

        # Instagram Reels are always short-form
        kind = (
            VideoKind.short
            if platform == Platform.instagram
            else VideoKind.vod
        )

        duration_s = self._probe_duration(str(path))

        logger.info("Local file: %s (%.1fs)", path, duration_s)

        return VideoSource(
            url="",
            local_path=str(path),
            platform=platform,
            kind=kind,
            duration_s=duration_s,
            title=path.stem,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


_YOUTUBE_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.|m\.)?(?:youtube\.com|youtu\.be)/", re.IGNORECASE
)


def get_ingester(source: str) -> VideoIngester:
    """
    Select the appropriate VideoIngester for *source*.

    Rules:
      - YouTube URL (youtube.com / youtu.be) → YouTubeIngester
      - Instagram URL → raise informative error (manual download required)
      - Anything else that looks like a path → LocalFileIngester
    """
    if _YOUTUBE_PATTERN.match(source):
        return YouTubeIngester()

    if "instagram.com" in source.lower():
        raise ValueError(
            "Instagram URLs are not automatically scraped (anti-bot policy). "
            "Please download the Reel manually and supply the local file path. "
            "See README.md §Instagram Reels."
        )

    # Treat as local file path
    return LocalFileIngester()
