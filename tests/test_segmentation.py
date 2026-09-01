"""
Tests for the segmentation module.

Covers: ad type classification, brand extraction, temporal grouping,
confidence calculation, and Rule #8 adjacent-brand merging.
"""

from __future__ import annotations

import pytest

from adlens.models import AdType, TimeWindow
from adlens.segmentation import SegmentBuilder, classify_ad_type, extract_brand
from adlens.models import VisualClassification


# ---------------------------------------------------------------------------
# Ad type classification tests
# ---------------------------------------------------------------------------


class TestClassifyAdType:
    def test_bumper_short_duration(self):
        # Short segment with no signals → bumper
        result = classify_ad_type([], [], segment_duration_s=1.5)
        assert result == AdType.bumper

    def test_bumper_with_hint(self):
        result = classify_ad_type([], ["bumper"], segment_duration_s=1.4)
        assert result == AdType.bumper

    def test_full_video_is_preroll(self):
        result = classify_ad_type([], [], segment_duration_s=30.0, is_full_video=True)
        assert result == AdType.preroll

    def test_sponsor_disclosure_is_midroll(self):
        result = classify_ad_type(
            ["sponsor_disclosure"], [], segment_duration_s=90.0
        )
        assert result == AdType.midroll_sponsor_read

    def test_discount_code_is_affiliate(self):
        result = classify_ad_type(
            ["discount_code"], [], segment_duration_s=45.0
        )
        assert result == AdType.affiliate

    def test_self_promo(self):
        result = classify_ad_type(
            ["self_promo"], [], segment_duration_s=20.0
        )
        assert result == AdType.self_promo

    def test_no_signals_is_other(self):
        result = classify_ad_type([], [], segment_duration_s=60.0)
        # > 3s with no signals → other (not bumper)
        assert result == AdType.other


# ---------------------------------------------------------------------------
# Temporal grouping tests
# ---------------------------------------------------------------------------


def make_settings():
    """Create settings with test-friendly values."""
    from adlens.utils.config import Settings
    return Settings(
        ad_score_threshold=0.4,
        max_gap_s=3.0,
        min_segment_duration_s=1.0,
    )


def make_window(start, end, score):
    return TimeWindow(start_s=start, end_s=end, fused_score=score)


class TestSegmentBuilder:
    def setup_method(self):
        self.builder = SegmentBuilder(make_settings())

    def test_group_windows_no_gap(self):
        windows = [
            make_window(0, 5, 0.8),
            make_window(5, 10, 0.8),
        ]
        groups = self.builder._group_windows(windows)
        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_group_windows_gap_exceeds_max(self):
        windows = [
            make_window(0, 5, 0.8),
            make_window(15, 20, 0.8),  # gap = 10s > max_gap_s
        ]
        groups = self.builder._group_windows(windows)
        assert len(groups) == 2

    def test_group_windows_gap_within_max(self):
        windows = [
            make_window(0, 5, 0.8),
            make_window(7, 12, 0.8),  # gap = 2s < max_gap_s=3
        ]
        groups = self.builder._group_windows(windows)
        assert len(groups) == 1

    def test_no_candidates_returns_empty(self):
        segments = self.builder.build(
            windows=[make_window(0, 10, 0.1)],  # below threshold
            video_duration_s=60.0,
            linguistic_evidence=[],
            visual_results=[],
            ocr_results=[],
        )
        assert segments == []

    def test_rule8_same_brand_merges(self):
        """Two adjacent same-brand segments within 2s should merge."""
        from adlens.models import AdSegment, SegmentEvidence

        seg1 = AdSegment(
            id="seg_01", start_s=10.0, end_s=50.0,
            ad_type=AdType.midroll_sponsor_read, confidence=0.8, brand="Acme",
            evidence=SegmentEvidence(signals_used=["audio:sponsor_disclosure"]),
        )
        seg2 = AdSegment(
            id="seg_02", start_s=51.0, end_s=90.0,  # gap=1s, same brand
            ad_type=AdType.midroll_sponsor_read, confidence=0.75, brand="Acme",
            evidence=SegmentEvidence(signals_used=["audio:discount_code"]),
        )
        merged = self.builder._merge_adjacent_same_brand([seg1, seg2])
        assert len(merged) == 1
        assert merged[0].end_s == 90.0
        assert "audio:sponsor_disclosure" in merged[0].evidence.signals_used
        assert "audio:discount_code" in merged[0].evidence.signals_used

    def test_rule8_different_brand_no_merge(self):
        """Two adjacent different-brand segments should NOT merge."""
        from adlens.models import AdSegment, SegmentEvidence

        seg1 = AdSegment(
            id="seg_01", start_s=10.0, end_s=50.0,
            ad_type=AdType.midroll_sponsor_read, confidence=0.8, brand="Acme",
            evidence=SegmentEvidence(),
        )
        seg2 = AdSegment(
            id="seg_02", start_s=51.0, end_s=90.0,  # gap=1s, DIFFERENT brand
            ad_type=AdType.midroll_sponsor_read, confidence=0.75, brand="Globex",
            evidence=SegmentEvidence(),
        )
        merged = self.builder._merge_adjacent_same_brand([seg1, seg2])
        assert len(merged) == 2

    def test_rule8_large_gap_no_merge(self):
        """Same brand but gap > 2s should NOT merge."""
        from adlens.models import AdSegment, SegmentEvidence

        seg1 = AdSegment(
            id="seg_01", start_s=10.0, end_s=50.0,
            ad_type=AdType.midroll_sponsor_read, confidence=0.8, brand="Acme",
            evidence=SegmentEvidence(),
        )
        seg2 = AdSegment(
            id="seg_02", start_s=55.0, end_s=90.0,  # gap=5s > 2s threshold
            ad_type=AdType.midroll_sponsor_read, confidence=0.75, brand="Acme",
            evidence=SegmentEvidence(),
        )
        merged = self.builder._merge_adjacent_same_brand([seg1, seg2])
        assert len(merged) == 2
