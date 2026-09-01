"""
Vision module: frame sampling, scene detection, and visual classification.

Components:
  - SamplingStrategy   — enum of available strategies
  - FrameSampler       — extracts frames from video using OpenCV
  - SceneDetector      — detects scene boundaries using PySceneDetect
  - VisualClassifier   — abstract interface for image-level ad classification
  - GPT4oMiniClassifier  — production: OpenAI GPT-4o-mini vision
  - MockVisualClassifier — dev stub
  - get_visual_classifier() — factory
"""

from __future__ import annotations

import base64
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from adlens.models import FrameSample, SceneBoundary, VisualClassification
from adlens.utils.logger import get_logger

if TYPE_CHECKING:
    from adlens.utils.config import Settings

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Scene Detector
# ---------------------------------------------------------------------------


class SceneDetector:
    """
    Detects scene boundaries in a video using PySceneDetect.

    Uses ContentDetector which compares HSV color histograms between
    frames.  Lower threshold = more sensitive to gradual changes.
    """

    def __init__(self, threshold: float = 27.0) -> None:
        self._threshold = threshold

    def detect(self, video_path: str) -> list[SceneBoundary]:
        """
        Return a list of scene boundaries (cut points) in the video.

        Falls back to an empty list if PySceneDetect is not installed,
        so the pipeline continues without scene signal.
        """
        try:
            from scenedetect import open_video, SceneManager  # type: ignore
            from scenedetect.detectors import ContentDetector  # type: ignore
        except ImportError:
            logger.warning(
                "PySceneDetect not installed — scene signal disabled. "
                "pip install scenedetect"
            )
            return []

        logger.info("Detecting scene boundaries: %s", Path(video_path).name)

        video = open_video(video_path)
        manager = SceneManager()
        manager.add_detector(ContentDetector(threshold=self._threshold))
        manager.detect_scenes(video, frame_skip=4, show_progress=False)

        scene_list = manager.get_scene_list()
        boundaries: list[SceneBoundary] = []

        for i, (start, end) in enumerate(scene_list):
            # The start of every scene (except the very first) is a cut
            if i > 0:
                boundaries.append(
                    SceneBoundary(timestamp_s=start.get_seconds(), score=1.0)
                )

        logger.info("Scene detection: %d boundaries found", len(boundaries))
        return boundaries


# ---------------------------------------------------------------------------
# Frame Sampler
# ---------------------------------------------------------------------------


class FrameSampler:
    """
    Extracts representative frames from a video file.

    Strategies
    ----------
    uniform
        Sample at a fixed rate (e.g. 1 FPS).  Simple but may miss
        short ads or over-sample long static segments.

    adaptive
        Sample at a dense rate around scene boundaries, then fill
        remaining time at a sparse rate.  Better coverage with fewer
        frames.

    keyframe
        Attempt to extract I-frames only.  Least redundancy, but frame
        distribution is decoder-dependent and less predictable.

    Configuration
    -------------
    All strategy parameters come from settings, not hardcoded here.
    """

    def __init__(self, settings: "Settings") -> None:
        self._settings = settings

    def sample(
        self,
        video_path: str,
        scene_boundaries: list[SceneBoundary],
        output_dir: str | None = None,
    ) -> list[FrameSample]:
        """
        Sample frames from *video_path* using the configured strategy.

        Returns a list of FrameSample objects sorted by timestamp.
        """
        strategy = self._settings.sampling_strategy
        logger.info("Sampling frames (strategy=%s): %s", strategy, Path(video_path).name)

        if strategy == "uniform":
            timestamps = self._uniform_timestamps(video_path)
        elif strategy == "adaptive":
            timestamps = self._adaptive_timestamps(video_path, scene_boundaries)
        elif strategy == "keyframe":
            timestamps = self._keyframe_timestamps(video_path)
        else:
            logger.warning("Unknown strategy '%s', falling back to uniform", strategy)
            timestamps = self._uniform_timestamps(video_path)

        # Apply hard cap (cap to 10 frames in local mode for ultra-fast processing)
        max_frames = self._settings.max_frames_per_video
        if self._settings.effective_vision_provider == "local":
            max_frames = min(max_frames, 10)

        if len(timestamps) > max_frames:
            logger.info(
                "Capping frames: %d → %d (max_frames_per_video=%d)",
                len(timestamps), max_frames, max_frames,
            )
            step = len(timestamps) / max_frames
            timestamps = [timestamps[int(i * step)] for i in range(max_frames)]

        out_dir = Path(output_dir) if output_dir else Path(video_path).parent / "frames"
        out_dir.mkdir(parents=True, exist_ok=True)

        return self._extract_frames(video_path, timestamps, out_dir)

    # ------------------------------------------------------------------
    # Timestamp generation strategies
    # ------------------------------------------------------------------

    def _get_duration(self, video_path: str) -> float:
        """Get video duration using OpenCV."""
        try:
            import cv2  # type: ignore
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            return frame_count / fps if fps > 0 else 0.0
        except Exception:
            return 0.0

    def _uniform_timestamps(self, video_path: str) -> list[float]:
        """Evenly spaced timestamps at `sample_fps` frames per second."""
        fps = self._settings.sample_fps
        duration = self._get_duration(video_path)
        if duration <= 0:
            return []
        step = 1.0 / fps
        timestamps = []
        t = step / 2  # start at half-step to avoid first/last frame edge effects
        while t < duration:
            timestamps.append(round(t, 3))
            t += step
        return timestamps

    def _adaptive_timestamps(
        self,
        video_path: str,
        scene_boundaries: list[SceneBoundary],
    ) -> list[float]:
        """
        Dense sampling around scene boundaries, sparse otherwise.

        Strategy:
          1. For each scene boundary, sample at -1s, 0s, +1s (clamped).
          2. Fill remaining intervals at 2 FPS (coarser than uniform).
          3. Deduplicate and sort.

        This concentrates frames at temporal transitions where ads
        typically start/end, and reduces frames in static content.
        """
        duration = self._get_duration(video_path)
        if duration <= 0:
            return self._uniform_timestamps(video_path)

        dense_window = 2.0  # seconds around each boundary
        sparse_fps = 0.5    # 1 frame every 2 seconds for background fill

        timestamps: set[float] = set()

        # Dense samples around boundaries
        for boundary in scene_boundaries:
            t = boundary.timestamp_s
            for offset in [-dense_window, -1.0, -0.5, 0.0, 0.5, 1.0, dense_window]:
                ts = round(t + offset, 3)
                if 0.0 < ts < duration:
                    timestamps.add(ts)

        # Sparse background fill
        t = 1.0 / sparse_fps / 2
        while t < duration:
            timestamps.add(round(t, 3))
            t += 1.0 / sparse_fps

        return sorted(timestamps)

    def _keyframe_timestamps(self, video_path: str) -> list[float]:
        """
        Extract I-frame timestamps using ffprobe.

        Falls back to uniform sampling if ffprobe fails.
        """
        import subprocess
        import json as _json

        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "quiet",
                    "-select_streams", "v",
                    "-skip_frame", "noref",
                    "-show_frames",
                    "-show_entries", "frame=pts_time,pict_type",
                    "-print_format", "json",
                    video_path,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            frames = _json.loads(result.stdout).get("frames", [])
            timestamps = [
                round(float(f["pts_time"]), 3)
                for f in frames
                if f.get("pict_type") == "I" and "pts_time" in f
            ]
            logger.info("Keyframe extraction: %d I-frames found", len(timestamps))
            return timestamps
        except Exception as exc:
            logger.warning("Keyframe extraction failed: %s — falling back to uniform", exc)
            return self._uniform_timestamps(video_path)

    # ------------------------------------------------------------------
    # Frame extraction
    # ------------------------------------------------------------------

    def _extract_frames(
        self,
        video_path: str,
        timestamps: list[float],
        out_dir: Path,
    ) -> list[FrameSample]:
        """Extract JPEG frames at the given timestamps using OpenCV."""
        try:
            import cv2  # type: ignore
        except ImportError:
            raise RuntimeError(
                "opencv-python-headless not installed: pip install opencv-python-headless"
            )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        samples: list[FrameSample] = []

        for idx, ts in enumerate(timestamps):
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
            ret, frame = cap.read()
            if not ret:
                logger.debug("No frame at t=%.3fs", ts)
                continue

            h, w = frame.shape[:2]
            max_dim = max(h, w)
            if max_dim > 720:
                scale = 720.0 / max_dim
                new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                h, w = new_h, new_w

            frame_path = out_dir / f"frame_{idx:05d}_{int(ts*1000):08d}ms.jpg"
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

            samples.append(
                FrameSample(
                    timestamp_s=ts,
                    frame_index=idx,
                    path=str(frame_path),
                    width=w,
                    height=h,
                )
            )

        cap.release()
        logger.info("Extracted %d frames to %s", len(samples), out_dir)
        return samples


# ---------------------------------------------------------------------------
# Visual Classifier — abstract interface
# ---------------------------------------------------------------------------


class VisualClassifier(ABC):
    """Abstract interface for frame-level ad classification."""

    @abstractmethod
    def classify_frames(
        self, frames: list[FrameSample]
    ) -> list[VisualClassification]:
        """
        Classify a batch of frames.

        Returns one VisualClassification per input frame, in the same order.
        """
        ...


# ---------------------------------------------------------------------------
# GPT-4o-mini visual classifier
# ---------------------------------------------------------------------------

_VISION_SYSTEM_PROMPT = """You are an expert at detecting advertisement and sponsored content in video frames.

For each frame, determine:
1. Whether it contains advertising, sponsorship, or promotional content
2. If yes, what type: preroll, midroll_sponsor_read, product_placement, self_promo, affiliate, bumper, or other
3. Confidence (0.0-1.0)
4. Any brand/product names visible
5. Brief reasoning

Respond with a JSON array where each element corresponds to one frame (in order):
[
  {
    "is_ad": true/false,
    "ad_type": "midroll_sponsor_read",
    "confidence": 0.85,
    "brand": "ExampleBrand",
    "reasoning": "Frame shows sponsor logo overlay with product name"
  }
]

Look for: sponsor overlays, product logos, promotional text, discount codes, CTA buttons,
brand watermarks, "sponsored by" graphics, product placements in shot."""


class GPT4oMiniClassifier(VisualClassifier):
    """
    Classifies frames using GPT-4o-mini vision.

    Frames are sent in batches (configurable) to minimize API calls.
    Uses detail="low" by default to minimize cost (~$0.000425/frame).
    """

    def __init__(self, api_key: str, batch_size: int = 5, detail: str = "low") -> None:
        self._api_key = api_key
        self._batch_size = batch_size
        self._detail = detail

    def classify_frames(
        self, frames: list[FrameSample]
    ) -> list[VisualClassification]:
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("openai package not installed")

        client = OpenAI(api_key=self._api_key)
        results: list[VisualClassification] = []

        for i in range(0, len(frames), self._batch_size):
            batch = frames[i : i + self._batch_size]
            batch_results = self._classify_batch(client, batch)
            results.extend(batch_results)

        return results

    def _classify_batch(
        self, client, batch: list[FrameSample]
    ) -> list[VisualClassification]:
        # Build message content with all frames in the batch
        content = [
            {
                "type": "text",
                "text": f"Classify these {len(batch)} video frames for advertising content.",
            }
        ]

        for frame in batch:
            b64 = self._encode_image(frame.path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                        "detail": self._detail,
                    },
                }
            )

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": _VISION_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                max_tokens=1024,
                temperature=0.1,
            )

            raw = response.choices[0].message.content.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:-1])

            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                parsed = [parsed]

            classifications = []
            for j, frame in enumerate(batch):
                item = parsed[j] if j < len(parsed) else {}
                classifications.append(
                    VisualClassification(
                        timestamp_s=frame.timestamp_s,
                        is_ad=bool(item.get("is_ad", False)),
                        ad_type_hint=item.get("ad_type", ""),
                        confidence=float(item.get("confidence", 0.0)),
                        brand_hint=item.get("brand", ""),
                        reasoning=item.get("reasoning", ""),
                    )
                )
            return classifications

        except Exception as exc:
            logger.warning("GPT-4o-mini vision call failed: %s", exc)
            # Return conservative non-ad classifications on failure
            return [
                VisualClassification(
                    timestamp_s=f.timestamp_s,
                    is_ad=False,
                    confidence=0.0,
                    reasoning=f"API error: {exc}",
                )
                for f in batch
            ]

    @staticmethod
    def _encode_image(path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Local (Free) visual classifier
# ---------------------------------------------------------------------------


class LocalVisualClassifier(VisualClassifier):
    """
    100% Free local visual classifier using local OCR and frame heuristics.

    Does not require any paid API key. Analyzes sampled frames for on-screen
    promotional text, poster layouts, and brand overlays.
    """

    def classify_frames(
        self, frames: list[FrameSample]
    ) -> list[VisualClassification]:
        from adlens.ocr import AdTextFilter, get_ocr_provider
        from adlens.utils.config import settings

        logger.info("Local visual classification: processing %d frames locally", len(frames))
        if not frames:
            return []

        ocr = get_ocr_provider(settings)
        ocr_results = ocr.run_batch(frames, AdTextFilter())
        classifications: list[VisualClassification] = []

        for frame, ocr_res in zip(frames, ocr_results):
            is_ad = ocr_res.ad_score > 0.0
            type_hint = ""
            if is_ad:
                text_lower = " ".join(ocr_res.ad_related_texts).lower()
                if any(k in text_lower for k in ["avis", "hospital", "centre", "center", "clinic", "medical", "vascular"]):
                    type_hint = "product_placement"
                elif any(k in text_lower for k in ["patreon", "merch", "subscribe", "newsletter"]):
                    type_hint = "self_promo"
                elif any(k in text_lower for k in ["code", "off", "discount"]):
                    type_hint = "affiliate"
                elif any(k in text_lower for k in ["sponsored", "partner"]):
                    type_hint = "midroll_sponsor_read"
                else:
                    type_hint = "product_placement"

            classifications.append(
                VisualClassification(
                    timestamp_s=frame.timestamp_s,
                    is_ad=is_ad,
                    ad_type_hint=type_hint,
                    confidence=max(ocr_res.ad_score, 0.75) if is_ad else 0.05,
                    brand_hint=ocr_res.ad_related_texts[0] if ocr_res.ad_related_texts else "Brand Graphic",
                    reasoning="Local OCR ad text detected: " + ", ".join(ocr_res.ad_related_texts[:3]) if is_ad else "No local ad text detected",
                )
            )

        return classifications


# ---------------------------------------------------------------------------
# Mock visual classifier
# ---------------------------------------------------------------------------


class MockVisualClassifier(VisualClassifier):
    """
    Deterministic stub classifier for dev/testing.

    Marks frames between 25-35% of video duration as ads (simulating
    a midroll sponsor read).  All results are clearly labelled MOCK.
    """

    def classify_frames(
        self, frames: list[FrameSample]
    ) -> list[VisualClassification]:
        logger.warning(
            "[MOCK VISION] Returning stub classifications — "
            "set ADLENS_VISION_PROVIDER=local or gpt4o_mini for real results."
        )
        if not frames:
            return []

        max_ts = max(f.timestamp_s for f in frames)
        results = []
        for frame in frames:
            pct = frame.timestamp_s / max_ts if max_ts > 0 else 0
            is_ad = 0.25 <= pct <= 0.35
            results.append(
                VisualClassification(
                    timestamp_s=frame.timestamp_s,
                    is_ad=is_ad,
                    ad_type_hint="midroll_sponsor_read" if is_ad else "",
                    confidence=0.7 if is_ad else 0.05,
                    brand_hint="MockBrand" if is_ad else "",
                    reasoning="[MOCK] deterministic stub result",
                )
            )
        return results


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_visual_classifier(settings: "Settings") -> VisualClassifier:
    """Return the visual classifier selected by configuration."""
    provider = settings.effective_vision_provider
    if provider == "gpt4o_mini":
        return GPT4oMiniClassifier(
            api_key=settings.openai_api_key,
            batch_size=settings.vision_batch_size,
            detail=settings.vision_detail,
        )
    elif provider == "local":
        return LocalVisualClassifier()
    else:
        return MockVisualClassifier()

