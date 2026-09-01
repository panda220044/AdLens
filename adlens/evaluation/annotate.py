"""
Ground truth annotation CLI for AdLens.

Interactive command-line tool for recording ground truth segments
from the five assignment test videos.

Usage:
    python -m adlens.evaluation.annotate [--output ground_truth.json]

The tool:
  1. Lists the five test videos from test_data.json
  2. For each video, prompts the user to enter segments one by one
  3. Saves results to ground_truth.json

Do NOT run this tool and claim the results are complete without
actually watching the videos and inspecting the content.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

AD_TYPES = [
    "preroll",
    "midroll_sponsor_read",
    "product_placement",
    "self_promo",
    "affiliate",
    "platform_inserted",
    "bumper",
    "other",
]

TEST_DATA_PATH = Path(__file__).parent.parent.parent / "test_data.json"
DEFAULT_OUTPUT = Path(__file__).parent.parent.parent / "ground_truth.json"


def load_test_data() -> list[dict]:
    if not TEST_DATA_PATH.exists():
        print(f"ERROR: test_data.json not found at {TEST_DATA_PATH}")
        sys.exit(1)
    with open(TEST_DATA_PATH) as f:
        data = json.load(f)
    return data.get("videos", [])


def prompt_segments(video: dict) -> list[dict]:
    """Interactively collect segment annotations for one video."""
    vid_id = video["video_id"]
    url = video.get("url") or video.get("local_path", "")

    print(f"\n{'='*60}")
    print(f"Video: {vid_id}")
    print(f"URL/Path: {url}")
    print(f"Platform: {video.get('platform', '?')}")
    print(f"{'='*60}")
    print("Enter segments one by one. Press ENTER with no input to finish.")
    print()

    segments = []
    seg_num = 1

    while True:
        print(f"  Segment {seg_num}:")
        start = input("    start_s (or ENTER to finish): ").strip()
        if not start:
            break

        try:
            start_s = float(start)
        except ValueError:
            print("    Invalid number, skipping.")
            continue

        end = input("    end_s: ").strip()
        try:
            end_s = float(end)
        except ValueError:
            print("    Invalid number, skipping.")
            continue

        if end_s <= start_s:
            print("    end_s must be > start_s, skipping.")
            continue

        print(f"    Ad type options: {', '.join(AD_TYPES)}")
        ad_type = input("    ad_type: ").strip().lower()
        if ad_type not in AD_TYPES:
            print(f"    Unknown type '{ad_type}', using 'other'.")
            ad_type = "other"

        brand = input("    brand (optional): ").strip()
        notes = input("    notes (optional): ").strip()

        segments.append(
            {
                "start_s": start_s,
                "end_s": end_s,
                "ad_type": ad_type,
                "brand": brand,
                "notes": notes,
            }
        )
        print(f"    ✓ Segment recorded: {start_s}–{end_s}s [{ad_type}]")
        seg_num += 1

    return segments


def run(output_path: Path = DEFAULT_OUTPUT) -> None:
    """Main annotation workflow."""
    videos = load_test_data()

    # Load existing ground truth if it exists (to allow incremental annotation)
    existing: dict = {"videos": []}
    if output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        print(f"Loaded existing ground truth: {output_path}")
        print(f"  {len(existing['videos'])} videos already annotated.")

    existing_ids = {v["video_id"] for v in existing["videos"]}

    print("\nAdLens Ground Truth Annotation Tool")
    print("=====================================")
    print("Watch each video carefully before annotating.")
    print("Only annotate videos you have actually watched.")
    print()

    for video in videos:
        vid_id = video["video_id"]

        if vid_id in existing_ids:
            skip = input(
                f"\n'{vid_id}' already annotated. Re-annotate? [y/N]: "
            ).strip().lower()
            if skip != "y":
                continue
            # Remove existing entry
            existing["videos"] = [
                v for v in existing["videos"] if v["video_id"] != vid_id
            ]

        segments = prompt_segments(video)

        existing["videos"].append(
            {
                "video_id": vid_id,
                "url": video.get("url", ""),
                "local_path": video.get("local_path", ""),
                "platform": video.get("platform", ""),
                "annotated_by": "manual",
                "segments": segments,
            }
        )

        # Save after each video so partial work is preserved
        with open(output_path, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"\nSaved → {output_path} ({len(segments)} segments for {vid_id})")

    print(f"\n✓ Annotation complete. Ground truth saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AdLens ground truth annotation tool")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output path for ground_truth.json",
    )
    args = parser.parse_args()
    run(Path(args.output))
