"""
AdLens Configuration System.

All configuration is driven by environment variables (never hardcoded).
Copy .env.example to .env and fill in values.

Usage:
    from adlens.utils.config import settings
    print(settings.openai_api_key)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    AdLens runtime settings.

    All values come from environment variables or a .env file.
    No secrets are ever hardcoded.
    """

    model_config = SettingsConfigDict(
        env_prefix="ADLENS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------
    # API Keys (no prefix — checked without ADLENS_ prefix)
    # ------------------------------------------------------------------
    openai_api_key: str = Field(
        default="",
        alias="OPENAI_API_KEY",
        description="OpenAI API key. Required for production ASR and vision.",
    )

    # ------------------------------------------------------------------
    # Provider selection
    # ------------------------------------------------------------------
    asr_provider: Literal["whisper_api", "faster_whisper", "free", "mock"] = Field(
        default="free",
        description=(
            "ASR backend. 'free' = 100% free speech recognition. "
            "'whisper_api' = OpenAI hosted (requires key). "
            "'faster_whisper' = local model. 'mock' = stub for dev/testing."
        ),
    )

    vision_provider: Literal["gpt4o_mini", "local", "mock"] = Field(
        default="local",
        description=(
            "Visual classification backend. "
            "'gpt4o_mini' = OpenAI GPT-4o-mini (requires key). "
            "'local' = 100% free local OCR + frame analysis. "
            "'mock' = deterministic stub."
        ),
    )

    # ------------------------------------------------------------------
    # Frame sampling
    # ------------------------------------------------------------------
    sampling_strategy: Literal["uniform", "adaptive", "keyframe"] = Field(
        default="adaptive",
        description=(
            "Frame sampling strategy. "
            "'uniform' = fixed FPS. "
            "'adaptive' = scene-boundary-aware + fill. "
            "'keyframe' = extract I-frames only."
        ),
    )

    sample_fps: float = Field(
        default=1.0,
        gt=0,
        description="Frames-per-second for uniform sampling strategy.",
    )

    max_frames_per_video: int = Field(
        default=300,
        gt=0,
        description="Hard cap on frames sampled per video (cost guard).",
    )

    # ------------------------------------------------------------------
    # Scene detection
    # ------------------------------------------------------------------
    scene_threshold: float = Field(
        default=27.0,
        gt=0,
        description="PySceneDetect ContentDetector threshold (lower = more sensitive).",
    )

    # ------------------------------------------------------------------
    # Signal fusion weights (must sum to ~1.0, validated below)
    # ------------------------------------------------------------------
    fusion_audio_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    fusion_visual_weight: float = Field(default=0.30, ge=0.0, le=1.0)
    fusion_ocr_weight: float = Field(default=0.20, ge=0.0, le=1.0)
    fusion_scene_weight: float = Field(default=0.15, ge=0.0, le=1.0)

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------
    ad_score_threshold: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Minimum fused score for a window to be considered an ad.",
    )

    min_segment_duration_s: float = Field(
        default=1.0,
        gt=0,
        description="Minimum segment duration in seconds (filters noise).",
    )

    max_gap_s: float = Field(
        default=3.0,
        gt=0,
        description="Maximum silence gap to bridge when merging adjacent segments.",
    )

    # ------------------------------------------------------------------
    # Working directory
    # ------------------------------------------------------------------
    work_dir: str = Field(
        default="./workdir",
        description="Directory for temporary files (downloads, frames, audio).",
    )

    # ------------------------------------------------------------------
    # Vision model detail level
    # ------------------------------------------------------------------
    vision_detail: Literal["low", "high"] = Field(
        default="low",
        description="GPT-4o-mini image detail level. 'low' is ~10x cheaper.",
    )

    vision_batch_size: int = Field(
        default=5,
        gt=0,
        description="Number of frames sent per vision API call.",
    )

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------
    @field_validator("openai_api_key", mode="before")
    @classmethod
    def _coerce_openai_key(cls, v: str) -> str:
        # Also accept from standard env var name
        return v or os.environ.get("OPENAI_API_KEY", "")

    @property
    def work_path(self) -> Path:
        p = Path(self.work_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key and self.openai_api_key.startswith("sk-"))

    @property
    def effective_asr_provider(self) -> str:
        """Fall back gracefully if no API key is available."""
        if self.asr_provider == "whisper_api" and not self.has_openai_key:
            return "faster_whisper"
        return self.asr_provider

    @property
    def effective_vision_provider(self) -> str:
        """Fall back gracefully if no API key is available."""
        if self.vision_provider == "gpt4o_mini" and not self.has_openai_key:
            return "local"
        return self.vision_provider

    @property
    def fusion_weights(self) -> dict[str, float]:
        total = (
            self.fusion_audio_weight
            + self.fusion_visual_weight
            + self.fusion_ocr_weight
            + self.fusion_scene_weight
        )
        if total == 0:
            return {"audio": 0.25, "visual": 0.25, "ocr": 0.25, "scene": 0.25}
        return {
            "audio": self.fusion_audio_weight / total,
            "visual": self.fusion_visual_weight / total,
            "ocr": self.fusion_ocr_weight / total,
            "scene": self.fusion_scene_weight / total,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached after first call)."""
    return Settings()


# Convenience alias
settings = get_settings()
