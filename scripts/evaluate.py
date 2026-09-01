"""
Evaluation CLI script for AdLens.

Usage:
    python scripts/evaluate.py \\
        --predictions results/all_results.json \\
        --ground-truth ground_truth.json \\
        --output eval_report.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from adlens.evaluation import (
    evaluate_aggregate,
    evaluate_video,
    format_report,
    load_ground_truth,
    load_predictions,
)
from adlens.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="AdLens evaluation script")
    parser.add_argument(
        "--predictions",
        default="results/all_results.json",
        help="Path to predictions JSON (all_results.json from run_test_set.py)",
    )
    parser.add_argument(
        "--ground-truth",
        default="ground_truth.json",
        help="Path to ground_truth.json",
    )
    parser.add_argument(
        "--output",
        default="eval_report.txt",
        help="Output path for text report",
    )
    parser.add_argument(
        "--json-output",
        default="eval_results.json",
        help="Output path for JSON results",
    )
    args = parser.parse_args()

    logger.info("Loading ground truth: %s", args.ground_truth)
    try:
        gt_by_vid = load_ground_truth(args.ground_truth)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    logger.info("Loading predictions: %s", args.predictions)
    try:
        pred_by_vid = load_predictions(args.predictions)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    # Evaluate per video
    video_results = []
    all_vid_ids = set(gt_by_vid) | set(pred_by_vid)

    for vid_id in sorted(all_vid_ids):
        gt_segs = gt_by_vid.get(vid_id, [])
        pred_segs = pred_by_vid.get(vid_id, [])
        vr = evaluate_video(vid_id, url="", predictions=pred_segs, ground_truths=gt_segs)
        video_results.append(vr)
        logger.info(
            "%s: P=%.3f R=%.3f F1=%.3f (TP=%d FP=%d FN=%d)",
            vid_id, vr.precision, vr.recall, vr.f1,
            vr.true_positives, vr.false_positives, vr.false_negatives,
        )

    agg = evaluate_aggregate(video_results)
    report = format_report(agg)

    print(report)

    # Save text report
    with open(args.output, "w") as f:
        f.write(report)
    logger.info("Report saved: %s", args.output)

    # Save JSON results (for programmatic access / CI)
    import dataclasses

    def dc_to_dict(obj):
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        return obj

    json_data = {
        "aggregate": {
            "n_videos": agg.n_videos,
            "macro_precision": agg.macro_precision,
            "macro_recall": agg.macro_recall,
            "macro_f1": agg.macro_f1,
            "micro_precision": agg.micro_precision,
            "micro_recall": agg.micro_recall,
            "micro_f1": agg.micro_f1,
            "mean_iou": agg.mean_iou,
            "mean_start_error_s": agg.mean_start_error_s,
            "mean_end_error_s": agg.mean_end_error_s,
        },
        "per_video": [
            {
                "video_id": vr.video_id,
                "n_gt": vr.n_gt,
                "n_pred": vr.n_pred,
                "true_positives": vr.true_positives,
                "false_positives": vr.false_positives,
                "false_negatives": vr.false_negatives,
                "precision": vr.precision,
                "recall": vr.recall,
                "f1": vr.f1,
                "mean_iou": vr.mean_iou,
                "mean_start_error_s": vr.mean_start_error_s,
                "mean_end_error_s": vr.mean_end_error_s,
            }
            for vr in agg.per_video
        ],
    }

    with open(args.json_output, "w") as f:
        json.dump(json_data, f, indent=2)
    logger.info("JSON results saved: %s", args.json_output)


if __name__ == "__main__":
    main()
