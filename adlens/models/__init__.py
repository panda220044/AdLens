"""Shared Pydantic data models for AdLens."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Platform(str, Enum):
    """Video source platform."""

    youtube = "youtube"
    instagram = "instagram"
    file = "file"
    other = "other"


class VideoKind(str, Enum):
    """Type of video content."""

    vod = "vod"          # Standard long-form video-on-demand
    short = "short"      # YouTube Shorts / Instagram Reels
    live = "live"        # Live stream (Tier 2 — not yet implemented)


class AdType(str, Enum):
    """
    Advertisement taxonomy.

    Definitions
    -----------
    preroll
        Advertisement that plays before the main content begins.
        On YouTube this is typically platform-inserted.
        For creator-controlled content, marks a hosted ad at t=0.

    midroll_sponsor_read
        A host-delivered sponsorship read within the main content.
        The host speaks directly to camera or continues their format
        while promoting a product/service.  Visual content may remain
        static (see ambiguity ruling #6 in DESIGN.md).

    product_placement
        A product or brand appears visibly within the content itself
        without an explicit announcement — e.g., prominently featured
        hardware, app, or service integrated into the video narrative.

    self_promo
        The creator promotes their own channel, merchandise, Patreon,
        newsletter, or other owned property.  Does NOT include ambient
        logo wear (ruling #2).

    affiliate
        A segment where the creator promotes a product via an affiliate
        link or discount code, earning a commission on sales.

    platform_inserted
        Advertisement inserted by the hosting platform (YouTube Ads).
        These do NOT appear in downloaded video files and cannot be
        detected from the video stream.  Documented in stats only.

    bumper
        Very short (≥ 1.0 s) animated or card-based "sponsored by"
        identifier.  Typically appears at the start or end of creator-
        produced segments.

    other
        Advertising content that does not fit neatly into the above
        categories.
    """

    preroll = "preroll"
    midroll_sponsor_read = "midroll_sponsor_read"
    product_placement = "product_placement"
    self_promo = "self_promo"
    affiliate = "affiliate"
    platform_inserted = "platform_inserted"
    bumper = "bumper"
    other = "other"


# ---------------------------------------------------------------------------
# Low-level evidence primitives
# ---------------------------------------------------------------------------


class TranscriptWord(BaseModel):
    """A single word from the ASR output with its timestamp."""

    word: str
    start_s: float
    end_s: float
    probability: float = 1.0


class TranscriptSpan(BaseModel):
    """A segment of the transcript (sentence/phrase level)."""

    start_s: float
    end_s: float
    text: str
    words: list[TranscriptWord] = Field(default_factory=list)


class LinguisticEvidence(BaseModel):
    """A linguistic signal detected in the transcript."""

    start_s: float
    end_s: float
    text: str
    signal_type: str       # e.g. "sponsor_disclosure", "discount_code", "cta"
    matched_pattern: str   # the exact pattern/keyword that triggered
    score: float = Field(ge=0.0, le=1.0)


class FrameSample(BaseModel):
    """Metadata for a sampled video frame."""

    timestamp_s: float
    frame_index: int
    path: str              # absolute path to saved JPEG
    width: int = 0
    height: int = 0


class OCRResult(BaseModel):
    """Text extracted from a single frame via OCR."""

    timestamp_s: float
    raw_texts: list[str] = Field(default_factory=list)
    ad_related_texts: list[str] = Field(default_factory=list)
    ad_score: float = Field(default=0.0, ge=0.0, le=1.0)


class VisualClassification(BaseModel):
    """Visual model classification result for a single frame."""

    timestamp_s: float
    is_ad: bool
    ad_type_hint: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    brand_hint: str = ""
    reasoning: str = ""


class SceneBoundary(BaseModel):
    """A detected scene change / cut point."""

    timestamp_s: float
    score: float = 0.0     # scene change score (higher = more abrupt)


# ---------------------------------------------------------------------------
# Fused evidence windows
# ---------------------------------------------------------------------------


class TimeWindow(BaseModel):
    """A temporal window with aggregated evidence scores."""

    start_s: float
    end_s: float
    audio_score: float = Field(default=0.0, ge=0.0, le=1.0)
    visual_score: float = Field(default=0.0, ge=0.0, le=1.0)
    ocr_score: float = Field(default=0.0, ge=0.0, le=1.0)
    scene_score: float = Field(default=0.0, ge=0.0, le=1.0)
    fused_score: float = Field(default=0.0, ge=0.0, le=1.0)
    linguistic_signals: list[str] = Field(default_factory=list)
    ocr_texts: list[str] = Field(default_factory=list)
    visual_hints: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Output contract models
# ---------------------------------------------------------------------------


class SegmentEvidence(BaseModel):
    """Evidence bundle for a detected advertisement segment."""

    frame_timestamps: list[float] = Field(default_factory=list)
    transcript_span: str = ""
    signals_used: list[str] = Field(default_factory=list)
    linguistic_details: list[LinguisticEvidence] = Field(default_factory=list)
    ocr_texts: list[str] = Field(default_factory=list)
    visual_classifications: list[VisualClassification] = Field(default_factory=list)


class AdSegment(BaseModel):
    """A detected advertisement segment — the core output unit."""

    id: str = Field(default_factory=lambda: f"seg_{uuid.uuid4().hex[:6]}")
    start_s: float
    end_s: float
    ad_type: AdType
    confidence: float = Field(ge=0.0, le=1.0)
    brand: str = ""
    description: str = ""
    evidence: SegmentEvidence = Field(default_factory=SegmentEvidence)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


class VideoSource(BaseModel):
    """Metadata about the video being analyzed."""

    url: str = ""
    local_path: str = ""
    platform: Platform = Platform.other
    kind: VideoKind = VideoKind.vod
    duration_s: float = 0.0
    title: str = ""
    processed_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class ProcessingStats(BaseModel):
    """Processing statistics — mandatory in every response."""

    wall_clock_s: float = 0.0
    estimated_cost_usd: float = 0.0
    frames_sampled: int = 0
    model_calls: int = 0
    audio_duration_s: float = 0.0
    asr_provider: str = ""
    vision_provider: str = ""
    sampling_strategy: str = ""


class AnalysisResult(BaseModel):
    """Top-level analysis result returned by the API."""

    source: VideoSource
    segments: list[AdSegment] = Field(default_factory=list)
    stats: ProcessingStats = Field(default_factory=ProcessingStats)

    def to_api_response(self) -> dict[str, Any]:
        """Serialize to the canonical API response format."""
        return {
            "source": {
                "url": self.source.url,
                "platform": self.source.platform.value,
                "kind": self.source.kind.value,
                "duration_s": self.source.duration_s,
                "processed_at": self.source.processed_at,
            },
            "segments": [
                {
                    "id": seg.id,
                    "start_s": seg.start_s,
                    "end_s": seg.end_s,
                    "ad_type": seg.ad_type.value,
                    "confidence": seg.confidence,
                    "brand": seg.brand,
                    "description": seg.description,
                    "evidence": {
                        "frame_timestamps": seg.evidence.frame_timestamps,
                        "transcript_span": seg.evidence.transcript_span,
                        "signals_used": seg.evidence.signals_used,
                    },
                }
                for seg in self.segments
            ],
            "stats": {
                "wall_clock_s": self.stats.wall_clock_s,
                "estimated_cost_usd": self.stats.estimated_cost_usd,
                "frames_sampled": self.stats.frames_sampled,
                "model_calls": self.stats.model_calls,
            },
        }
