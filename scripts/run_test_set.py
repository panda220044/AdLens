"""
CLI script to run AdLens analysis on the five assignment test videos.

Usage:
    python scripts/run_test_set.py [--output results/]

Skips Instagram Reels if local_path is not configured in test_data.json.
Saves each result as a JSON file in the output directory.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from adlens.analyzer import AdLensAnalyzer
from adlens.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AdLens on all test videos")
    parser.add_argument("--output", default="results", help="Output directory for results")
    parser.add_argument(
        "--test-data", default="test_data.json", help="Path to test_data.json"
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.test_data) as f:
        test_data = json.load(f)

    analyzer = AdLensAnalyzer()
    all_results = []
    t_total_start = time.monotonic()

    for video in test_data["videos"]:
        vid_id = video["video_id"]
        url = video.get("url", "")
        local_path = video.get("local_path", "")
        platform = video.get("platform", "")

        # Determine what to analyze
        if platform == "instagram":
            if not local_path or not Path(local_path).exists():
                logger.warning(
                    "Skipping %s — Instagram Reel requires manual download. "
                    "Set local_path in test_data.json.",
                    vid_id,
                )
                continue
            source = local_path
        else:
            source = url

        logger.info("\n%s\nAnalyzing: %s (%s)\n%s", "="*60, vid_id, source, "="*60)

        try:
            t_start = time.monotonic()
            result = analyzer.analyze(source)
            elapsed = time.monotonic() - t_start

            response = result.to_api_response()
            # Add video_id for evaluation lookup
            response["source"]["video_id"] = vid_id

            # Save individual result
            out_file = out_dir / f"{vid_id}.json"
            with open(out_file, "w") as f:
                json.dump(response, f, indent=2)

            logger.info(
                "✓ %s — %d segments | %.1fs | $%.4f",
                vid_id,
                len(response["segments"]),
                elapsed,
                response["stats"]["estimated_cost_usd"],
            )
            all_results.append(response)

        except Exception as exc:
            logger.error("✗ %s failed: %s", vid_id, exc)

    # Save combined results for evaluation
    combined_file = out_dir / "all_results.json"
    with open(combined_file, "w") as f:
        json.dump(all_results, f, indent=2)

    total_elapsed = time.monotonic() - t_total_start
    total_cost = sum(r["stats"]["estimated_cost_usd"] for r in all_results)

    print(f"\n{'='*60}")
    print(f"Test set complete")
    print(f"  Videos processed : {len(all_results)}")
    print(f"  Total wall clock : {total_elapsed:.1f}s")
    print(f"  Total cost       : ${total_cost:.4f}")
    print(f"  Results saved to : {out_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
