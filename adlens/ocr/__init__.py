"""
OCR module: extract and filter text from video frames.

Provides:
  - OCRProvider (abstract)
  - EasyOCRProvider — local deep-learning OCR (no system install needed)
  - MockOCRProvider — dev stub
  - AdTextFilter    — classifies extracted text as ad-related or not
  - get_ocr_provider() — factory
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from adlens.models import FrameSample, OCRResult
from adlens.utils.logger import get_logger

if TYPE_CHECKING:
    from adlens.utils.config import Settings

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Ad-related text patterns (for OCR filtering)
# ---------------------------------------------------------------------------

# These patterns flag text as advertising-related.
# Organized so each match can be explained in evidence output.
AD_TEXT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("sponsor_label",   re.compile(r"\bsponsored\b|\bsponsoring\b", re.I)),
    ("discount_code",   re.compile(r"\b(?:code|coupon|promo)\s*:?\s*[A-Z0-9]{3,}\b", re.I)),
    ("cta",             re.compile(r"\bswipe\s+up\b|\blink\s+in\s+bio\b|\bshop\s+now\b|\bbuy\s+now\b", re.I)),
    ("discount_pct",    re.compile(r"\b\d{1,2}\s*%\s+off\b|\bsave\s+\$?\d+\b", re.I)),
    ("free_trial",      re.compile(r"\bfree\s+trial\b|\bfree\s+\d+\s+days?\b", re.I)),
    ("affiliate_url",   re.compile(r"https?://[^\s]+(?:ref|aff|partner|click)[^\s]*", re.I)),
    ("ad_label",        re.compile(r"^\s*(?:ad|advertisement|promoted|paid\s+partnership)\s*$", re.I)),
    ("download_cta",    re.compile(r"\bdownload\s+(?:now|free|today)\b|\bget\s+the\s+app\b", re.I)),
    ("price_point",     re.compile(r"\bonly\s+\$\d+|\bjust\s+\$\d+|\b\d+\s*-\s*rupees?\b", re.I)),
    ("patreon",         re.compile(r"\bpatreon\.com\b|\bpatreon\b", re.I)),
    ("merch",           re.compile(r"\bmerch\b|\bshop\b.*\blink\b", re.I)),
    ("brand_hospital",  re.compile(r"\b(?:hospitals?|vascular|clinic|centre|center|health\s*care|medical|diagnostics|nursing)\b", re.I)),
    ("contact_info",    re.compile(r"\b\d{5}[\s.-]?\d{5}\b|\b\d{3}[\s.-]?\d{3}[\s.-]?\d{4}\b|\bcontact\b|\bcall\s+now\b", re.I)),
    ("brand_org",       re.compile(r"\b(?:avis|idream|unit\s+of|pvt\s+ltd|limited|ltd|inc|corp)\b", re.I)),
    ("self_promo_cta",  re.compile(r"\b(?:subscribe|follow|visit|website|check\s+out)\b", re.I)),
]


class AdTextFilter:
    """Classifies extracted OCR text as ad-related or not."""

    def score(self, texts: list[str]) -> tuple[list[str], float]:
        """
        Filter *texts* to ad-related items and compute a score.

        Returns
        -------
        ad_texts : list of matching strings
        score    : 0.0–1.0 aggregate ad likelihood
        """
        ad_texts: list[str] = []
        matched_patterns: set[str] = set()

        for text in texts:
            for label, pattern in AD_TEXT_PATTERNS:
                if pattern.search(text):
                    ad_texts.append(text)
                    matched_patterns.add(label)
                    break  # each text string counted once

        score = 0.0
        if ad_texts:
            # 1 category matched = 0.50 score, >= 2 categories matched = 0.85 score
            score = 0.50 if len(matched_patterns) == 1 else 0.85

        return ad_texts, score


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class OCRProvider(ABC):
    """Abstract base for OCR backends."""

    @abstractmethod
    def extract(self, frame: FrameSample) -> list[str]:
        """Return a list of text strings found in the frame."""
        ...

    def run_batch(
        self,
        frames: list[FrameSample],
        text_filter: AdTextFilter | None = None,
    ) -> list[OCRResult]:
        """
        Run OCR on all frames and apply optional ad-text filtering.

        Returns one OCRResult per frame.
        """
        filter_ = text_filter or AdTextFilter()
        results: list[OCRResult] = []

        for frame in frames:
            try:
                texts = self.extract(frame)
            except Exception as exc:
                logger.debug("OCR failed for frame %.3fs: %s", frame.timestamp_s, exc)
                texts = []

            ad_texts, score = filter_.score(texts)
            results.append(
                OCRResult(
                    timestamp_s=frame.timestamp_s,
                    raw_texts=texts,
                    ad_related_texts=ad_texts,
                    ad_score=score,
                )
            )

        ad_count = sum(1 for r in results if r.ad_score > 0)
        logger.info(
            "OCR: processed %d frames, %d with ad text", len(frames), ad_count
        )
        return results


# ---------------------------------------------------------------------------
# EasyOCR provider
# ---------------------------------------------------------------------------


class EasyOCRProvider(OCRProvider):
    """
    OCR using EasyOCR (deep-learning, no system Tesseract needed).

    Languages default to English only.  Add more via the constructor
    if non-English ads are expected.

    Model weights (~100–300 MB) are downloaded on first use and cached
    in the user's home directory.
    """

    def __init__(self, languages: list[str] | None = None) -> None:
        self._languages = languages or ["en"]
        self._reader = None
        self._cache: dict[str, list[str]] = {}

    def _get_reader(self):
        if self._reader is None:
            try:
                import easyocr  # type: ignore
            except ImportError:
                raise RuntimeError(
                    "easyocr not installed: pip install easyocr"
                )
            logger.info("Loading EasyOCR model (first run downloads weights)…")
            import torch
            torch.set_num_threads(1)
            self._reader = easyocr.Reader(self._languages, gpu=False, quantize=False, verbose=False)
        return self._reader

    def extract(self, frame: FrameSample) -> list[str]:
        if frame.path in self._cache:
            return self._cache[frame.path]

        reader = self._get_reader()
        img_input: Any = frame.path
        try:
            import cv2
            img = cv2.imread(frame.path)
            if img is not None:
                h, w = img.shape[:2]
                max_dim = max(h, w)
                if max_dim > 640:
                    scale = 640.0 / max_dim
                    new_w, new_h = int(w * scale), int(h * scale)
                    img_input = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except Exception:
            pass

        results = reader.readtext(img_input, canvas_size=320, detail=0, paragraph=False)
        res = [str(r).strip() for r in results if r and str(r).strip()]
        self._cache[frame.path] = res
        return res


# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------


class MockOCRProvider(OCRProvider):
    """
    Deterministic stub OCR for development.

    Returns fake "SPONSORED" / "USE CODE MOCK20" labels at the same
    25-35% range used by MockVisualClassifier.
    """

    def extract(self, frame: FrameSample) -> list[str]:
        logger.debug("[MOCK OCR] t=%.3f", frame.timestamp_s)
        # Determine if this frame is in the mock ad zone
        # (cannot know duration here; use raw timestamp heuristic)
        if 30.0 <= frame.timestamp_s <= 90.0:
            return ["SPONSORED", "USE CODE MOCK20", "Visit mockbrand.com"]
        return []


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


_shared_easyocr_provider: EasyOCRProvider | None = None


def get_ocr_provider(settings: "Settings") -> OCRProvider:
    """Return the OCR provider appropriate for the current settings."""
    global _shared_easyocr_provider
    if settings.effective_vision_provider in ("gpt4o_mini", "local") or settings.asr_provider in ("faster_whisper", "free"):
        if _shared_easyocr_provider is None:
            _shared_easyocr_provider = EasyOCRProvider()
        return _shared_easyocr_provider
    else:
        return MockOCRProvider()
