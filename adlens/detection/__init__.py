"""
Signal fusion layer.

Combines ASR/linguistic, OCR, visual, and scene-change evidence into
a single fused score per time window.

Key design choices:
  - Evidence is aggregated into fixed-width non-overlapping windows
  - Each signal contributes a score 0.0–1.0 to its window
  - Final score = weighted average (weights sum to 1.0)
  - Weights are configurable via environment variables
  - All evidence is preserved for explainability, not discarded

The fusion logic is intentionally transparent — no black box model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from adlens.models import (
    LinguisticEvidence,
    OCRResult,
    SceneBoundary,
    TimeWindow,
    VisualClassification,
)
from adlens.utils.logger import get_logger

if TYPE_CHECKING:
    from adlens.utils.config import Settings

logger = get_logger(__name__)


class EvidenceAggregator:
    """
    Aggregates multi-modal evidence into fixed-width time windows.

    Window width is chosen based on video duration:
      - Short videos (< 120s): 2-second windows
      - Long videos (≥ 120s): 5-second windows

    Each window accumulates the maximum signal score from its inputs
    (max-pooling, not averaging) to avoid diluting short bursts.
    """

    def __init__(self, window_s: float | None = None) -> None:
        self._window_s = window_s  # None = auto-select

    def build_windows(
        self,
        duration_s: float,
        linguistic_evidence: list[LinguisticEvidence],
        ocr_results: list[OCRResult],
        visual_results: list[VisualClassification],
        scene_boundaries: list[SceneBoundary],
    ) -> list[TimeWindow]:
        """Create time windows and fill them with evidence scores."""
        if duration_s <= 0:
            return []

        # Auto-select window width
        window_s = self._window_s or (2.0 if duration_s < 120 else 5.0)

        # Create windows
        windows: list[TimeWindow] = []
        t = 0.0
        while t < duration_s:
            w_end = min(t + window_s, duration_s)
            windows.append(TimeWindow(start_s=round(t, 3), end_s=round(w_end, 3)))
            t = w_end

        logger.info(
            "Fusion: %d windows (%.1fs each) for %.1fs video",
            len(windows), window_s, duration_s
        )

        # Fill audio scores from linguistic evidence
        for ev in linguistic_evidence:
            for w in self._overlapping(windows, ev.start_s, ev.end_s):
                w.audio_score = max(w.audio_score, ev.score)
                if ev.signal_type not in w.linguistic_signals:
                    w.linguistic_signals.append(ev.signal_type)

        # Fill OCR scores
        for ocr in ocr_results:
            for w in self._overlapping(windows, ocr.timestamp_s, ocr.timestamp_s):
                w.ocr_score = max(w.ocr_score, ocr.ad_score)
                w.ocr_texts.extend(ocr.ad_related_texts)

        # Fill visual scores
        for vis in visual_results:
            for w in self._overlapping(windows, vis.timestamp_s, vis.timestamp_s):
                score = vis.confidence if vis.is_ad else 0.0
                w.visual_score = max(w.visual_score, score)
                if vis.ad_type_hint and vis.ad_type_hint not in w.visual_hints:
                    w.visual_hints.append(vis.ad_type_hint)

        # Fill scene scores — mark windows containing a boundary
        for boundary in scene_boundaries:
            for w in self._overlapping(windows, boundary.timestamp_s, boundary.timestamp_s):
                w.scene_score = max(w.scene_score, boundary.score)

        return windows

    @staticmethod
    def _overlapping(
        windows: list[TimeWindow], start_s: float, end_s: float
    ) -> list[TimeWindow]:
        """Return all windows that overlap with [start_s, end_s]."""
        return [
            w for w in windows
            if w.start_s < end_s + 0.001 and w.end_s > start_s - 0.001
        ]


class AdScorer:
    """
    Computes a fused advertisement probability score for each time window.

    Formula
    -------
    fused = w_audio * audio_score
           + w_visual * visual_score
           + w_ocr * ocr_score
           + w_scene * scene_score

    Weights are normalized to sum to 1.0 (handled by settings.fusion_weights).

    Scene signal is treated as a multiplier boost, not an independent
    indicator.  A scene boundary alone does not flag an ad; it amplifies
    adjacent evidence.
    """

    def __init__(self, settings: "Settings") -> None:
        self._weights = settings.fusion_weights

    def score(self, windows: list[TimeWindow]) -> list[TimeWindow]:
        """Compute and store `fused_score` for every window."""
        w_a = self._weights["audio"]
        w_v = self._weights["visual"]
        w_o = self._weights["ocr"]
        w_s = self._weights["scene"]

        for window in windows:
            # Base weighted sum
            base = (
                w_a * window.audio_score
                + w_v * window.visual_score
                + w_o * window.ocr_score
            )
            # Boost visual-only poster/banner ads: if visual or OCR score is strong, boost base
            if window.visual_score >= 0.50 or window.ocr_score >= 0.50:
                strong_vis_ocr = max(window.visual_score, window.ocr_score)
                base = max(base, strong_vis_ocr * 0.85)

            # Scene is a small additive boost when other signals are present
            scene_boost = w_s * window.scene_score * (1.0 if base > 0.1 else 0.2)
            window.fused_score = round(min(base + scene_boost, 1.0), 4)

        return windows
