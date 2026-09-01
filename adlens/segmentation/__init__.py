"""
Segmentation module.

Converts per-window fused scores into final AdSegment objects.

Pipeline:
  1. Candidate generation  — find all windows above threshold
  2. Temporal grouping     — merge windows within max_gap_s
  3. Start/end refinement  — shrink to highest-confidence core
  4. Ad type classification — pick best ad_type from evidence
  5. Brand extraction      — surface most frequent brand hint
  6. Confidence calculation — aggregate per segment

Ambiguity rulings applied here (see DESIGN.md for full explanations):

  Rule #8 — Two sponsors back-to-back:
    If two candidate segments are adjacent (gap < 2s) AND they share
    the same brand hint, merge into one.  If different brands or gap
    ≥ 2s, keep as separate segments.

  Rule #3 — 1.4-second bumper:
    Minimum segment duration is 1.0s (configurable).  A 1.4s bumper
    passes this filter and is classified as 'bumper' if the visual
    hint includes 'bumper' or the segment is very short (< 3s).

  Rule #7 — Entire Reel is an ad:
    When a single candidate covers > 90% of total duration, the
    segment is kept as-is and typed 'preroll' (or the strongest hint).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING

from adlens.models import (
    AdSegment,
    AdType,
    LinguisticEvidence,
    OCRResult,
    SegmentEvidence,
    TimeWindow,
    VisualClassification,
)
from adlens.utils.logger import get_logger

if TYPE_CHECKING:
    from adlens.utils.config import Settings

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Ad type classifier
# ---------------------------------------------------------------------------

# Maps evidence signal patterns → AdType.
# Order matters: earlier entries take priority.
_TYPE_RULES: list[tuple[AdType, set[str]]] = [
    (AdType.bumper,              {"bumper"}),
    (AdType.platform_inserted,  {"platform_inserted"}),
    (AdType.self_promo,         {"self_promo", "patreon", "merch"}),
    (AdType.affiliate,           {"affiliate", "discount_code"}),
    (AdType.midroll_sponsor_read, {"sponsor_disclosure", "sponsor_label"}),
    (AdType.product_placement,  {"product_placement"}),
    (AdType.preroll,             {"preroll"}),
]


def classify_ad_type(
    linguistic_signals: list[str],
    visual_hints: list[str],
    segment_duration_s: float,
    is_full_video: bool = False,
) -> AdType:
    """
    Determine the most likely AdType for a segment.

    Logic:
      - Short segment (< 3s) with visual 'bumper' hint → bumper
      - Entire video is the segment → preroll
      - Check linguistic then visual signals against type rules
      - Fall back to 'other' if nothing matches
    """
    all_signals = set(linguistic_signals + visual_hints)

    # Short bumper (Rule #3)
    if segment_duration_s < 3.0 and (
        "bumper" in all_signals or not all_signals
    ):
        return AdType.bumper

    # Entire video (Rule #7)
    if is_full_video:
        return AdType.preroll

    for ad_type, keywords in _TYPE_RULES:
        if all_signals & keywords:
            return ad_type

    # self_promo check via text
    if "self_promo" in all_signals:
        return AdType.self_promo

    return AdType.other


# ---------------------------------------------------------------------------
# Brand extractor
# ---------------------------------------------------------------------------

def extract_brand(
    visual_results: list[VisualClassification],
    ocr_texts: list[str],
) -> str:
    """Surface the most commonly cited brand from evidence."""
    brand_votes: Counter = Counter()

    for vis in visual_results:
        if vis.brand_hint and vis.brand_hint.lower() not in {"", "mockbrand"}:
            brand_votes[vis.brand_hint] += 1

    # Simple brand extraction from OCR: any capitalized token of 4+ chars
    cap_word = re.compile(r"\b[A-Z][A-Za-z]{3,}\b")
    for text in ocr_texts:
        for m in cap_word.findall(text):
            if m.lower() not in {
                "sponsored", "advertisement", "promo", "code", "save",
                "free", "trial", "visit", "download", "shop", "link",
            }:
                brand_votes[m] += 1

    if brand_votes:
        return brand_votes.most_common(1)[0][0]
    return ""


# ---------------------------------------------------------------------------
# Segment builder
# ---------------------------------------------------------------------------


class SegmentBuilder:
    """
    Converts fused time windows into final AdSegment list.

    See module docstring for pipeline description and ambiguity rulings.
    """

    def __init__(self, settings: "Settings") -> None:
        self._threshold = settings.ad_score_threshold
        self._max_gap_s = max(settings.max_gap_s, 8.0 if settings.effective_vision_provider == "local" else settings.max_gap_s)
        self._min_duration_s = settings.min_segment_duration_s

    def build(
        self,
        windows: list[TimeWindow],
        video_duration_s: float,
        linguistic_evidence: list[LinguisticEvidence],
        visual_results: list[VisualClassification],
        ocr_results: list[OCRResult],
    ) -> list[AdSegment]:
        """Run the full segmentation pipeline and return segments."""

        # 1. Candidate windows
        candidates = [w for w in windows if w.fused_score >= self._threshold]
        if not candidates:
            logger.info("Segmentation: no windows above threshold (%.2f)", self._threshold)
            return []

        logger.info(
            "Segmentation: %d candidate windows (threshold=%.2f)",
            len(candidates), self._threshold,
        )

        # 2. Temporal grouping — merge windows within max_gap_s
        groups = self._group_windows(candidates)
        logger.info("Segmentation: %d groups after temporal merging", len(groups))

        # 3. Build segments from groups
        segments: list[AdSegment] = []
        for i, group in enumerate(groups):
            seg = self._build_segment(
                group_idx=i,
                group=group,
                video_duration_s=video_duration_s,
                linguistic_evidence=linguistic_evidence,
                visual_results=visual_results,
                ocr_results=ocr_results,
            )
            if seg is not None:
                segments.append(seg)

        # 4. Rule #8 — adjacent same-brand merging
        segments = self._merge_adjacent_same_brand(segments)

        # 5. Assign stable IDs
        for i, seg in enumerate(segments):
            seg.id = f"seg_{i+1:02d}"

        logger.info("Segmentation: final %d segments", len(segments))
        return segments

    # ------------------------------------------------------------------

    def _group_windows(
        self, candidates: list[TimeWindow]
    ) -> list[list[TimeWindow]]:
        """Merge candidate windows separated by less than max_gap_s."""
        if not candidates:
            return []

        groups: list[list[TimeWindow]] = [[candidates[0]]]
        for w in candidates[1:]:
            last_group = groups[-1]
            gap = w.start_s - last_group[-1].end_s
            if gap <= self._max_gap_s:
                last_group.append(w)
            else:
                groups.append([w])
        return groups

    def _build_segment(
        self,
        group_idx: int,
        group: list[TimeWindow],
        video_duration_s: float,
        linguistic_evidence: list[LinguisticEvidence],
        visual_results: list[VisualClassification],
        ocr_results: list[OCRResult],
    ) -> AdSegment | None:
        """Convert a group of windows into a single AdSegment."""
        start_s = group[0].start_s
        end_s = group[-1].end_s
        duration_s = end_s - start_s

        # Filter out noise (Rule #3 min duration)
        if duration_s < self._min_duration_s:
            logger.debug(
                "Dropping short candidate %.1fs–%.1fs (%.2fs < min %.2fs)",
                start_s, end_s, duration_s, self._min_duration_s,
            )
            return None

        # Collect evidence within this segment's time range
        seg_linguistic = [
            e for e in linguistic_evidence
            if e.end_s >= start_s and e.start_s <= end_s
        ]
        seg_visual = [
            v for v in visual_results
            if start_s <= v.timestamp_s <= end_s
        ]
        seg_ocr = [
            o for o in ocr_results
            if start_s <= o.timestamp_s <= end_s
        ]

        # Collect signals
        all_linguistic_signals = list({e.signal_type for e in seg_linguistic})
        all_visual_hints = list({
            v.ad_type_hint for v in seg_visual if v.ad_type_hint
        })
        all_ocr_texts = [
            t for o in seg_ocr for t in o.ad_related_texts
        ]

        # Classify ad type
        is_full_video = (duration_s / max(video_duration_s, 1)) > 0.90
        ad_type = classify_ad_type(
            linguistic_signals=all_linguistic_signals,
            visual_hints=all_visual_hints,
            segment_duration_s=duration_s,
            is_full_video=is_full_video,
        )

        # Confidence = weighted average of window fused scores
        confidence = sum(w.fused_score for w in group) / len(group)
        confidence = round(min(confidence, 1.0), 4)

        # Brand
        brand = extract_brand(seg_visual, all_ocr_texts)

        # Description
        description = self._build_description(
            ad_type=ad_type,
            start_s=start_s,
            end_s=end_s,
            brand=brand,
            linguistic_signals=all_linguistic_signals,
        )

        # Evidence bundle
        evidence = SegmentEvidence(
            frame_timestamps=[v.timestamp_s for v in seg_visual],
            transcript_span=self._transcript_text(seg_linguistic),
            signals_used=self._all_signals(all_linguistic_signals, all_visual_hints, seg_ocr),
            linguistic_details=seg_linguistic,
            ocr_texts=all_ocr_texts[:20],  # cap to avoid huge responses
            visual_classifications=seg_visual,
        )

        return AdSegment(
            start_s=round(start_s, 2),
            end_s=round(end_s, 2),
            ad_type=ad_type,
            confidence=confidence,
            brand=brand,
            description=description,
            evidence=evidence,
        )

    def _merge_adjacent_same_brand(
        self, segments: list[AdSegment]
    ) -> list[AdSegment]:
        """
        Rule #8: Two sponsors back-to-back.

        Merge only if: gap < 2s AND same non-empty brand.
        Otherwise keep separate.
        """
        if len(segments) < 2:
            return segments

        SAME_BRAND_GAP_S = 2.0
        merged: list[AdSegment] = [segments[0]]

        for seg in segments[1:]:
            prev = merged[-1]
            gap = seg.start_s - prev.end_s
            same_brand = (
                prev.brand
                and seg.brand
                and prev.brand.lower() == seg.brand.lower()
            )

            if gap < SAME_BRAND_GAP_S and same_brand:
                logger.info(
                    "Rule #8: merging adjacent same-brand segments (%s) %.1f–%.1f",
                    prev.brand, prev.start_s, seg.end_s,
                )
                # Extend previous segment
                prev.end_s = seg.end_s
                # Merge evidence
                prev.evidence.frame_timestamps.extend(seg.evidence.frame_timestamps)
                prev.evidence.signals_used = list(
                    set(prev.evidence.signals_used + seg.evidence.signals_used)
                )
                prev.confidence = max(prev.confidence, seg.confidence)
            else:
                merged.append(seg)

        return merged

    @staticmethod
    def _transcript_text(evidence: list[LinguisticEvidence]) -> str:
        if not evidence:
            return ""
        texts = list(dict.fromkeys(e.text for e in evidence))  # deduplicate, preserve order
        combined = " … ".join(texts[:3])
        return combined[:500]  # cap length

    @staticmethod
    def _all_signals(
        linguistic: list[str],
        visual: list[str],
        ocr: list[OCRResult],
    ) -> list[str]:
        signals: list[str] = []
        for s in linguistic:
            signals.append(f"audio:{s}")
        for v in visual:
            if v:
                signals.append(f"visual:{v}")
        if any(o.ad_score > 0 for o in ocr):
            signals.append("ocr:ad_text_detected")
        return list(dict.fromkeys(signals))  # deduplicate

    @staticmethod
    def _build_description(
        ad_type: AdType,
        start_s: float,
        end_s: float,
        brand: str,
        linguistic_signals: list[str],
    ) -> str:
        duration = end_s - start_s
        brand_str = f" by {brand}" if brand else ""
        signal_str = (
            f" (signals: {', '.join(linguistic_signals[:3])})"
            if linguistic_signals else ""
        )
        type_label = ad_type.value.replace("_", " ").title()
        return (
            f"{type_label}{brand_str} at {start_s:.1f}s–{end_s:.1f}s "
            f"({duration:.1f}s){signal_str}"
        )
