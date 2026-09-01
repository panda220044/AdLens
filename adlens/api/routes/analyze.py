"""
Analysis routes.

POST /analyze       — analyze a video from a URL
POST /analyze/file  — analyze an uploaded local video file
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, HttpUrl, field_validator

from adlens.analyzer import AdLensAnalyzer
from adlens.utils.config import settings
from adlens.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

_analyzer = AdLensAnalyzer()


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class AnalyzeURLRequest(BaseModel):
    """Request body for URL-based analysis."""

    url: str

    @field_validator("url")
    @classmethod
    def _validate_url(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("url must not be empty")
        return v


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/analyze", response_model=None)
async def analyze_url(request: AnalyzeURLRequest) -> dict[str, Any]:
    """
    Analyze a video from a URL.

    Accepts YouTube (VOD and Shorts) URLs.
    Returns the full analysis result with detected ad segments.

    For Instagram Reels: supply the locally downloaded file via
    POST /analyze/file instead.
    """
    logger.info("POST /analyze — url=%s", request.url)

    try:
        result = await asyncio.to_thread(_analyzer.analyze, request.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Analysis failed for %s", request.url)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")

    return result.to_api_response()


@router.post("/analyze/file", response_model=None)
async def analyze_file(video: UploadFile = File(...)) -> dict[str, Any]:
    """
    Analyze a locally uploaded video file.

    Use this endpoint for:
    - Manually downloaded Instagram Reels
    - Any local .mp4 / .mov / .mkv file

    The file is saved to a temporary location in the work directory,
    analyzed, and the temp file is cleaned up afterward.
    """
    logger.info("POST /analyze/file — filename=%s", video.filename)

    # Validate extension
    suffix = Path(video.filename or "video.mp4").suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}",
        )

    # Save upload to work directory
    upload_dir = settings.work_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    tmp_path = upload_dir / (video.filename or "upload.mp4")

    try:
        with open(tmp_path, "wb") as f:
            content = await video.read()
            f.write(content)

        logger.info("Saved upload: %s (%.1f MB)", tmp_path, len(content) / 1e6)

        result = await asyncio.to_thread(_analyzer.analyze, str(tmp_path))
    except Exception as exc:
        logger.exception("Analysis failed for uploaded file %s", video.filename)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")

    return result.to_api_response()
