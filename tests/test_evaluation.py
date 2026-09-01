"""
Tests for the AdLens evaluation framework.

Tests non-obvious logic: IoU calculation, segment matching, metric
computation, and edge cases (no predictions, no ground truth, etc.)
"""

from __future__ import annotations

import pytest

from adlens.evaluation import (
    GTSegment,
    PredSegment,
    AggregateResult,
    evaluate_aggregate,
    evaluate_video,
    match_segments,
    segment_iou,
)


# ---------------------------------------------------------------------------
# IoU tests
# ---------------------------------------------------------------------------


class TestSegmentIoU:
    def test_perfect_overlap(self):
        assert segment_iou(0, 10, 0, 10) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert segment_iou(0, 5, 10, 20) == pytest.approx(0.0)

    def test_partial_overlap(self):
        # [0–10] and [5–15] — intersection=5, union=15
        iou = segment_iou(0, 10, 5, 15)
        assert iou == pytest.approx(5 / 15)

    def test_contained(self):
        # [2–8] fully inside [0–10] — intersection=6, union=10
        iou = segment_iou(2, 8, 0, 10)
        assert iou == pytest.approx(6 / 10)

    def test_touching_boundary(self):
        # Segments that touch at a single point — no overlap
        iou = segment_iou(0, 5, 5, 10)
        assert iou == pytest.approx(0.0)

    def test_zero_duration(self):
        assert segment_iou(5, 5, 0, 10) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Matching tests
# ---------------------------------------------------------------------------


def _pred(start, end, ad_type="midroll_sponsor_read", conf=0.8):
    return PredSegment(start_s=start, end_s=end, ad_type=ad_type, confidence=conf)


def _gt(start, end, ad_type="midroll_sponsor_read"):
    return GTSegment(start_s=start, end_s=end, ad_type=ad_type)


class TestMatchSegments:
    def test_exact_match(self):
        preds = [_pred(10, 60)]
        gts = [_gt(10, 60)]
        matches, unmatched_gt, unmatched_pred = match_segments(preds, gts)
        assert len(matches) == 1
        assert matches[0].iou == pytest.approx(1.0)
        assert len(unmatched_gt) == 0
        assert len(unmatched_pred) == 0

    def test_no_overlap_both_unmatched(self):
        preds = [_pred(0, 10)]
        gts = [_gt(50, 100)]
        matches, unmatched_gt, unmatched_pred = match_segments(preds, gts)
        assert len(matches) == 0
        assert len(unmatched_gt) == 1
        assert len(unmatched_pred) == 1

    def test_one_pred_two_gt_greedy(self):
        # One prediction overlaps with both GTs — should match the higher IoU
        preds = [_pred(0, 20)]
        gts = [_gt(0, 20), _gt(10, 30)]  # first is perfect match
        matches, unmatched_gt, unmatched_pred = match_segments(preds, gts)
        assert len(matches) == 1
        assert matches[0].iou == pytest.approx(1.0)

    def test_below_threshold_no_match(self):
        # IoU = 2/18 ≈ 0.11 — below default 0.5 threshold
        preds = [_pred(0, 10)]
        gts = [_gt(8, 20)]
        matches, unmatched_gt, unmatched_pred = match_segments(
            preds, gts, iou_threshold=0.5
        )
        assert len(matches) == 0

    def test_empty_predictions(self):
        gts = [_gt(0, 30)]
        matches, unmatched_gt, unmatched_pred = match_segments([], gts)
        assert matches == []
        assert len(unmatched_gt) == 1
        assert len(unmatched_pred) == 0

    def test_empty_ground_truth(self):
        preds = [_pred(0, 30)]
        matches, unmatched_gt, unmatched_pred = match_segments(preds, [])
        assert matches == []
        assert len(unmatched_gt) == 0
        assert len(unmatched_pred) == 1


# ---------------------------------------------------------------------------
# Per-video evaluation tests
# ---------------------------------------------------------------------------


class TestEvaluateVideo:
    def test_perfect(self):
        vr = evaluate_video(
            "vid1", "",
            predictions=[_pred(10, 60)],
            ground_truths=[_gt(10, 60)],
        )
        assert vr.precision == pytest.approx(1.0)
        assert vr.recall == pytest.approx(1.0)
        assert vr.f1 == pytest.approx(1.0)
        assert vr.mean_iou == pytest.approx(1.0)

    def test_all_false_positives(self):
        vr = evaluate_video(
            "vid1", "",
            predictions=[_pred(0, 30), _pred(60, 90)],
            ground_truths=[],
        )
        assert vr.precision == pytest.approx(0.0)
        assert vr.recall == pytest.approx(0.0)  # no GT means 0 recall
        assert vr.f1 == pytest.approx(0.0)
        assert vr.false_positives == 2

    def test_all_false_negatives(self):
        vr = evaluate_video(
            "vid1", "",
            predictions=[],
            ground_truths=[_gt(30, 90)],
        )
        assert vr.recall == pytest.approx(0.0)
        assert vr.false_negatives == 1

    def test_partial(self):
        # 1 TP, 1 FP, 1 FN
        vr = evaluate_video(
            "vid1", "",
            predictions=[_pred(0, 30), _pred(200, 230)],  # second is FP
            ground_truths=[_gt(0, 30), _gt(100, 130)],    # second is FN
        )
        assert vr.true_positives == 1
        assert vr.false_positives == 1
        assert vr.false_negatives == 1
        assert vr.precision == pytest.approx(0.5)
        assert vr.recall == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Aggregate evaluation tests
# ---------------------------------------------------------------------------


class TestEvaluateAggregate:
    def test_empty_input(self):
        agg = evaluate_aggregate([])
        assert agg.n_videos == 0

    def test_macro_vs_micro(self):
        # Video A: P=1.0 R=1.0, Video B: P=0.0 R=0.0
        vr_a = evaluate_video("a", "", [_pred(0, 30)], [_gt(0, 30)])
        vr_b = evaluate_video("b", "", [_pred(0, 30)], [_gt(100, 130)])
        agg = evaluate_aggregate([vr_a, vr_b])

        # Macro: average of per-video metrics
        assert agg.macro_precision == pytest.approx(0.5)
        assert agg.macro_recall == pytest.approx(0.5)

        # Micro: aggregate TP/FP/FN counts
        assert agg.micro_precision == pytest.approx(0.5)  # 1 TP / (1 TP + 1 FP)
        assert agg.micro_recall == pytest.approx(0.5)     # 1 TP / (1 TP + 1 FN)
