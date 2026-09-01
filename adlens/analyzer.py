"""
Core analysis orchestrator for AdLens.

Ties together all pipeline stages:
  1. Video ingestion
  2. Audio extraction + ASR + linguistic analysis
  3. Scene detection
  4. Frame sampling
  5. OCR
  6. Visual classification
  7. Signal fusion
  8. Segmentation

Produces an AnalysisResult with full evidence.
"""

from __future__ import annotations

import time
from pathlib import Path

from adlens.asr import LinguisticAnalyzer, get_asr_provider
from adlens.audio import AudioExtractor
from adlens.detection import AdScorer, EvidenceAggregator
from adlens.ingestion import get_ingester
from adlens.models import AnalysisResult, ProcessingStats, VideoSource
from adlens.ocr import get_ocr_provider
from adlens.segmentation import SegmentBuilder
from adlens.utils.config import Settings, get_settings
from adlens.utils.cost_tracker import CostTracker
from adlens.utils.logger import get_logger
from adlens.vision import FrameSampler, SceneDetector, get_visual_classifier

logger = get_logger(__name__)


class AdLensAnalyzer:
    """
    Main analysis orchestrator.

    All configuration comes from Settings (environment variables).
    All providers are injected via factories so they can be replaced.

    Usage::

        analyzer = AdLensAnalyzer()
        result = analyzer.analyze("https://youtube.com/watch?v=...")
        print(result.to_api_response())
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def analyze(self, source: str) -> AnalysisResult:
        """
        Run the full analysis pipeline on *source* (URL or local path).

        Parameters
        ----------
        source:
            YouTube URL, YouTube Shorts URL, or local video file path.
            Instagram URLs raise ValueError — supply local file instead.
        """
        tracker = CostTracker()
        s = self._settings
        t_start = time.monotonic()

        logger.info("=" * 60)
        logger.info("AdLens analysis starting: %s", source)
        logger.info("ASR provider   : %s", s.effective_asr_provider)
        logger.info("Vision provider: %s", s.effective_vision_provider)
        logger.info("Sampling       : %s @ %.1f fps", s.sampling_strategy, s.sample_fps)

        # ------------------------------------------------------------------
        # Stage 1: Ingestion
        # ------------------------------------------------------------------
        logger.info("--- Stage 1: Ingestion ---")
        ingester = get_ingester(source)
        video_source: VideoSource = ingester.ingest(source)
        logger.info(
            "Ingested: platform=%s kind=%s duration=%.1fs",
            video_source.platform.value,
            video_source.kind.value,
            video_source.duration_s,
        )

        work_dir = str(s.work_path)

        # ------------------------------------------------------------------
        # Stage 2: Audio extraction + ASR + linguistic analysis
        # ------------------------------------------------------------------
        logger.info("--- Stage 2: ASR + Linguistic Analysis ---")
        asr_provider = get_asr_provider(s)
        audio_extractor = AudioExtractor()
        linguistic_analyzer = LinguisticAnalyzer()

        try:
            audio_path = audio_extractor.extract(
                video_source.local_path, output_dir=work_dir
            )
            transcript = asr_provider.transcribe(audio_path)
            tracker.record_whisper(video_source.duration_s)
        except Exception as exc:
            logger.error("ASR failed: %s — continuing without audio signal", exc)
            transcript = []

        linguistic_evidence = linguistic_analyzer.analyze(transcript)
        logger.info(
            "Linguistic: %d transcript spans, %d ad signals",
            len(transcript), len(linguistic_evidence),
        )

        # ------------------------------------------------------------------
        # Stage 3: Scene detection
        # ------------------------------------------------------------------
        logger.info("--- Stage 3: Scene Detection ---")
        scene_detector = SceneDetector(threshold=s.scene_threshold)
        try:
            scene_boundaries = scene_detector.detect(video_source.local_path)
        except Exception as exc:
            logger.warning("Scene detection failed: %s", exc)
            scene_boundaries = []

        logger.info("Scene detection: %d boundaries", len(scene_boundaries))

        # ------------------------------------------------------------------
        # Stage 4: Frame sampling
        # ------------------------------------------------------------------
        logger.info("--- Stage 4: Frame Sampling ---")
        frame_sampler = FrameSampler(s)
        frame_dir = str(s.work_path / "frames" / Path(video_source.local_path).stem)

        try:
            frames = frame_sampler.sample(
                video_source.local_path,
                scene_boundaries=scene_boundaries,
                output_dir=frame_dir,
            )
        except Exception as exc:
            logger.error("Frame sampling failed: %s — continuing without frames", exc)
            frames = []

        tracker.record_frames(len(frames))
        logger.info("Sampled %d frames", len(frames))

        # ------------------------------------------------------------------
        # Stage 5: OCR
        # ------------------------------------------------------------------
        logger.info("--- Stage 5: OCR ---")
        ocr_provider = get_ocr_provider(s)
        try:
            ocr_target_frames = frames
            if s.effective_vision_provider == "local" and len(frames) > 8:
                step = max(len(frames) // 8, 1)
                ocr_target_frames = [frames[i] for i in range(0, len(frames), step)][:8]
            ocr_results = ocr_provider.run_batch(ocr_target_frames)
        except Exception as exc:
            logger.error("OCR failed: %s — continuing without OCR signal", exc)
            ocr_results = []

        # ------------------------------------------------------------------
        # Stage 6: Visual classification
        # ------------------------------------------------------------------
        logger.info("--- Stage 6: Visual Classification ---")
        visual_classifier = get_visual_classifier(s)

        # Only run vision model on frames that have some prior signal
        # (ad_score from OCR > 0 OR near a scene boundary) to reduce cost
        candidate_frames = self._select_candidate_frames(
            frames, ocr_results, scene_boundaries
        )
        logger.info(
            "Vision: %d/%d frames selected for classification",
            len(candidate_frames), len(frames),
        )

        try:
            visual_results = visual_classifier.classify_frames(candidate_frames)
            if candidate_frames:
                tracker.record_vision_call(
                    n_images=len(candidate_frames), detail=s.vision_detail
                )
        except Exception as exc:
            logger.error("Visual classification failed: %s", exc)
            visual_results = []

        # ------------------------------------------------------------------
        # Stage 7: Signal fusion
        # ------------------------------------------------------------------
        logger.info("--- Stage 7: Signal Fusion ---")
        aggregator = EvidenceAggregator()
        scorer = AdScorer(s)

        windows = aggregator.build_windows(
            duration_s=video_source.duration_s,
            linguistic_evidence=linguistic_evidence,
            ocr_results=ocr_results,
            visual_results=visual_results,
            scene_boundaries=scene_boundaries,
        )
        windows = scorer.score(windows)

        scored_count = sum(1 for w in windows if w.fused_score >= s.ad_score_threshold)
        logger.info(
            "Fusion: %d windows scored, %d above threshold (%.2f)",
            len(windows), scored_count, s.ad_score_threshold,
        )

        # ------------------------------------------------------------------
        # Stage 8: Segmentation
        # ------------------------------------------------------------------
        logger.info("--- Stage 8: Segmentation ---")
        builder = SegmentBuilder(s)
        segments = builder.build(
            windows=windows,
            video_duration_s=video_source.duration_s,
            linguistic_evidence=linguistic_evidence,
            visual_results=visual_results,
            ocr_results=ocr_results,
        )

        logger.info("Segmentation: %d final segments", len(segments))
        for seg in segments:
            logger.info(
                "  %s  %.1f–%.1fs  type=%-25s conf=%.2f  brand=%s",
                seg.id, seg.start_s, seg.end_s,
                seg.ad_type.value, seg.confidence, seg.brand or "—",
            )

        # ------------------------------------------------------------------
        # Assemble result
        # ------------------------------------------------------------------
        wall_s = time.monotonic() - t_start
        stats = ProcessingStats(
            wall_clock_s=round(wall_s, 2),
            estimated_cost_usd=round(tracker.total_cost_usd, 6),
            frames_sampled=tracker.frames_sampled,
            model_calls=tracker.model_calls,
            audio_duration_s=video_source.duration_s,
            asr_provider=s.effective_asr_provider,
            vision_provider=s.effective_vision_provider,
            sampling_strategy=s.sampling_strategy,
        )

        logger.info(
            "Analysis complete in %.1fs | cost=$%.4f | %d model calls | %d frames",
            wall_s, tracker.total_cost_usd, tracker.model_calls, tracker.frames_sampled,
        )
        logger.info("=" * 60)

        return AnalysisResult(
            source=video_source,
            segments=segments,
            stats=stats,
        )

    def _select_candidate_frames(
        self,
        frames: list[FrameSample],
        ocr_results: list,
        scene_boundaries: list,
    ) -> list[FrameSample]:
        if not frames:
            return []

        # If 8 or fewer frames, use all
        if len(frames) <= 8:
            return frames

        # In local CPU vision mode, sample 8 evenly distributed frames to stay within HTTP timeout
        if self._settings.effective_vision_provider == "local":
            step = max(len(frames) // 8, 1)
            selected = [frames[i] for i in range(0, len(frames), step)][:8]
            return selected

        ocr_by_ts = {r.timestamp_s: r for r in ocr_results}
        boundary_times = {b.timestamp_s for b in scene_boundaries}

        def is_candidate(frame) -> bool:
            ts = frame.timestamp_s
            ocr = ocr_by_ts.get(ts)
            if ocr and ocr.ad_score > 0.1:
                return True
            if any(abs(ts - b) <= 2.0 for b in boundary_times):
                return True
            if ts <= 10.0:
                return True
            return False

        candidates = [f for f in frames if is_candidate(f)]
        min_count = max(len(frames) // 10, 5)
        if len(candidates) < min_count:
            step = len(frames) // min_count
            for i in range(0, len(frames), step):
                if frames[i] not in candidates:
                    candidates.append(frames[i])
            candidates.sort(key=lambda f: f.timestamp_s)

        return candidates
