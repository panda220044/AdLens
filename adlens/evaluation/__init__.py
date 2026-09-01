"""
Evaluation framework for AdLens.

Computes:
  - Precision, Recall, F1 (at segment level)
  - Per-segment IoU (Intersection over Union)
  - Boundary error (start/end RMSE)
  - Per-video breakdowns
  - Failure analysis

Ground truth format (ground_truth.json):
{
  "videos": [
    {
      "video_id": "yt_ujFWRFYLGjY",
      "url": "https://www.youtube.com/watch?v=ujFWRFYLGjY",
      "local_path": "",
      "segments": [
        {
          "start_s": 120.5,
          "end_s": 180.0,
          "ad_type": "midroll_sponsor_read",
          "brand": "ExampleBrand",
          "notes": "Clear sponsor read with discount code"
        }
      ]
    }
  ]
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adlens.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class GTSegment:
    """Ground truth segment."""
    start_s: float
    end_s: float
    ad_type: str
    brand: str = ""
    notes: str = ""

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class PredSegment:
    """Predicted segment from AdLens."""
    start_s: float
    end_s: float
    ad_type: str
    confidence: float
    brand: str = ""


@dataclass
class SegmentMatch:
    """Result of matching one prediction to one ground truth."""
    pred: PredSegment
    gt: GTSegment
    iou: float
    start_error_s: float
    end_error_s: float
    type_match: bool


@dataclass
class VideoResult:
    """Evaluation results for a single video."""
    video_id: str
    url: str
    n_gt: int
    n_pred: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    mean_iou: float
    mean_start_error_s: float
    mean_end_error_s: float
    matches: list[SegmentMatch] = field(default_factory=list)
    unmatched_gt: list[GTSegment] = field(default_factory=list)
    unmatched_pred: list[PredSegment] = field(default_factory=list)


@dataclass
class AggregateResult:
    """Aggregate metrics across all videos."""
    n_videos: int
    macro_precision: float
    macro_recall: float
    macro_f1: float
    micro_precision: float
    micro_recall: float
    micro_f1: float
    mean_iou: float
    mean_start_error_s: float
    mean_end_error_s: float
    per_video: list[VideoResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# IoU computation
# ---------------------------------------------------------------------------


def segment_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Compute Intersection over Union for two time segments."""
    intersection = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 0:
        return 0.0
    return intersection / union


# ---------------------------------------------------------------------------
# Matching: greedy by IoU (Hungarian would be overkill here)
# ---------------------------------------------------------------------------

IOU_THRESHOLD = 0.50  # minimum IoU to count as a true positive


def match_segments(
    predictions: list[PredSegment],
    ground_truths: list[GTSegment],
    iou_threshold: float = IOU_THRESHOLD,
) -> tuple[list[SegmentMatch], list[GTSegment], list[PredSegment]]:
    """
    Greedy matching of predictions to ground truths by IoU.

    Each prediction and ground truth can be matched at most once.
    Returns: (matches, unmatched_gt, unmatched_pred)
    """
    if not predictions or not ground_truths:
        return [], list(ground_truths), list(predictions)

    # Build all candidate pairs sorted by IoU descending
    candidates: list[tuple[float, int, int]] = []
    for pi, pred in enumerate(predictions):
        for gi, gt in enumerate(ground_truths):
            iou = segment_iou(pred.start_s, pred.end_s, gt.start_s, gt.end_s)
            if iou >= iou_threshold:
                candidates.append((iou, pi, gi))

    candidates.sort(reverse=True, key=lambda x: x[0])

    matched_preds: set[int] = set()
    matched_gts: set[int] = set()
    matches: list[SegmentMatch] = []

    for iou, pi, gi in candidates:
        if pi in matched_preds or gi in matched_gts:
            continue
        pred = predictions[pi]
        gt = ground_truths[gi]
        matches.append(
            SegmentMatch(
                pred=pred,
                gt=gt,
                iou=iou,
                start_error_s=abs(pred.start_s - gt.start_s),
                end_error_s=abs(pred.end_s - gt.end_s),
                type_match=pred.ad_type == gt.ad_type,
            )
        )
        matched_preds.add(pi)
        matched_gts.add(gi)

    unmatched_gt = [ground_truths[i] for i in range(len(ground_truths)) if i not in matched_gts]
    unmatched_pred = [predictions[i] for i in range(len(predictions)) if i not in matched_preds]

    return matches, unmatched_gt, unmatched_pred


# ---------------------------------------------------------------------------
# Per-video evaluation
# ---------------------------------------------------------------------------


def evaluate_video(
    video_id: str,
    url: str,
    predictions: list[PredSegment],
    ground_truths: list[GTSegment],
) -> VideoResult:
    """Evaluate predictions vs. ground truth for a single video."""
    matches, unmatched_gt, unmatched_pred = match_segments(predictions, ground_truths)

    tp = len(matches)
    fp = len(unmatched_pred)
    fn = len(unmatched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    mean_iou = sum(m.iou for m in matches) / len(matches) if matches else 0.0
    mean_start_err = (
        sum(m.start_error_s for m in matches) / len(matches) if matches else 0.0
    )
    mean_end_err = (
        sum(m.end_error_s for m in matches) / len(matches) if matches else 0.0
    )

    return VideoResult(
        video_id=video_id,
        url=url,
        n_gt=len(ground_truths),
        n_pred=len(predictions),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        mean_iou=round(mean_iou, 4),
        mean_start_error_s=round(mean_start_err, 2),
        mean_end_error_s=round(mean_end_err, 2),
        matches=matches,
        unmatched_gt=unmatched_gt,
        unmatched_pred=unmatched_pred,
    )


# ---------------------------------------------------------------------------
# Aggregate evaluation
# ---------------------------------------------------------------------------


def evaluate_aggregate(video_results: list[VideoResult]) -> AggregateResult:
    """Compute macro and micro aggregate metrics."""
    n = len(video_results)
    if n == 0:
        return AggregateResult(n_videos=0, macro_precision=0, macro_recall=0,
                                macro_f1=0, micro_precision=0, micro_recall=0,
                                micro_f1=0, mean_iou=0, mean_start_error_s=0,
                                mean_end_error_s=0)

    macro_p = sum(v.precision for v in video_results) / n
    macro_r = sum(v.recall for v in video_results) / n
    macro_f1 = 2 * macro_p * macro_r / (macro_p + macro_r) if (macro_p + macro_r) else 0

    total_tp = sum(v.true_positives for v in video_results)
    total_fp = sum(v.false_positives for v in video_results)
    total_fn = sum(v.false_negatives for v in video_results)

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0

    # IoU and boundary errors — only from matched pairs
    all_ious = [m.iou for v in video_results for m in v.matches]
    all_start_errs = [m.start_error_s for v in video_results for m in v.matches]
    all_end_errs = [m.end_error_s for v in video_results for m in v.matches]

    mean_iou = sum(all_ious) / len(all_ious) if all_ious else 0.0
    mean_start = sum(all_start_errs) / len(all_start_errs) if all_start_errs else 0.0
    mean_end = sum(all_end_errs) / len(all_end_errs) if all_end_errs else 0.0

    return AggregateResult(
        n_videos=n,
        macro_precision=round(macro_p, 4),
        macro_recall=round(macro_r, 4),
        macro_f1=round(macro_f1, 4),
        micro_precision=round(micro_p, 4),
        micro_recall=round(micro_r, 4),
        micro_f1=round(micro_f1, 4),
        mean_iou=round(mean_iou, 4),
        mean_start_error_s=round(mean_start, 2),
        mean_end_error_s=round(mean_end, 2),
        per_video=video_results,
    )


# ---------------------------------------------------------------------------
# Ground truth loader
# ---------------------------------------------------------------------------


def load_ground_truth(path: str) -> dict[str, list[GTSegment]]:
    """
    Load ground_truth.json and return a mapping of video_id → segments.

    Raises FileNotFoundError if the file doesn't exist.
    Raises ValueError if the format is invalid.
    """
    gt_path = Path(path)
    if not gt_path.exists():
        raise FileNotFoundError(
            f"Ground truth file not found: {gt_path}\n"
            "Run: python -m adlens.evaluation.annotate to create it."
        )

    with open(gt_path) as f:
        data = json.load(f)

    result: dict[str, list[GTSegment]] = {}
    for video in data.get("videos", []):
        vid_id = video.get("video_id", "")
        segs = [
            GTSegment(
                start_s=float(s["start_s"]),
                end_s=float(s["end_s"]),
                ad_type=s.get("ad_type", "other"),
                brand=s.get("brand", ""),
                notes=s.get("notes", ""),
            )
            for s in video.get("segments", [])
        ]
        result[vid_id] = segs

    return result


# ---------------------------------------------------------------------------
# Prediction loader
# ---------------------------------------------------------------------------


def load_predictions(path: str) -> dict[str, list[PredSegment]]:
    """
    Load a saved predictions file (JSON in API response format).

    Expects a list of AnalysisResult API responses:
    [
      { "source": {..., "video_id": "..."}, "segments": [...] }
    ]
    """
    pred_path = Path(path)
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")

    with open(pred_path) as f:
        data = json.load(f)

    result: dict[str, list[PredSegment]] = {}
    if isinstance(data, list):
        items = data
    else:
        items = [data]

    for item in items:
        source = item.get("source", {})
        vid_id = source.get("video_id", source.get("url", "unknown"))
        segs = [
            PredSegment(
                start_s=float(s["start_s"]),
                end_s=float(s["end_s"]),
                ad_type=s.get("ad_type", "other"),
                confidence=float(s.get("confidence", 0.0)),
                brand=s.get("brand", ""),
            )
            for s in item.get("segments", [])
        ]
        result[vid_id] = segs

    return result


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_report(aggregate: AggregateResult) -> str:
    """Format evaluation results as a human-readable text report."""
    lines = [
        "=" * 70,
        "AdLens Evaluation Report",
        "=" * 70,
        f"Videos evaluated: {aggregate.n_videos}",
        "",
        "Aggregate Metrics",
        "-" * 40,
        f"  Macro Precision : {aggregate.macro_precision:.4f}",
        f"  Macro Recall    : {aggregate.macro_recall:.4f}",
        f"  Macro F1        : {aggregate.macro_f1:.4f}",
        f"  Micro Precision : {aggregate.micro_precision:.4f}",
        f"  Micro Recall    : {aggregate.micro_recall:.4f}",
        f"  Micro F1        : {aggregate.micro_f1:.4f}",
        f"  Mean IoU        : {aggregate.mean_iou:.4f}",
        f"  Mean Start Err  : {aggregate.mean_start_error_s:.2f}s",
        f"  Mean End Err    : {aggregate.mean_end_error_s:.2f}s",
        "",
        "Per-Video Results",
        "-" * 40,
    ]

    for vr in aggregate.per_video:
        lines.append(f"\n  [{vr.video_id}]")
        lines.append(f"    GT={vr.n_gt}  Pred={vr.n_pred}  "
                     f"TP={vr.true_positives}  FP={vr.false_positives}  "
                     f"FN={vr.false_negatives}")
        lines.append(f"    P={vr.precision:.3f}  R={vr.recall:.3f}  "
                     f"F1={vr.f1:.3f}  IoU={vr.mean_iou:.3f}")

        if vr.unmatched_gt:
            lines.append("    MISSED (false negatives):")
            for gt in vr.unmatched_gt:
                lines.append(
                    f"      GT  {gt.start_s:.1f}–{gt.end_s:.1f}s  "
                    f"{gt.ad_type}  {gt.notes[:60]}"
                )

        if vr.unmatched_pred:
            lines.append("    FALSE POSITIVES:")
            for pred in vr.unmatched_pred:
                lines.append(
                    f"      PRED {pred.start_s:.1f}–{pred.end_s:.1f}s  "
                    f"{pred.ad_type}  conf={pred.confidence:.2f}"
                )

    lines.append("=" * 70)
    return "\n".join(lines)
