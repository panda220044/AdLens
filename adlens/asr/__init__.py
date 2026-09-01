"""
ASR (Automatic Speech Recognition) module.

Provides:
  - ASRProvider     — abstract interface
  - WhisperAPIProvider  — OpenAI hosted Whisper
  - FasterWhisperProvider — local faster-whisper
  - MockASRProvider — deterministic stub for testing without API keys
  - LinguisticAnalyzer  — scans transcript for ad-related signals

Factory: `get_asr_provider(settings) → ASRProvider`
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from adlens.models import LinguisticEvidence, TranscriptSpan, TranscriptWord
from adlens.utils.logger import get_logger

if TYPE_CHECKING:
    from adlens.utils.config import Settings

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class ASRProvider(ABC):
    """
    Abstract base for all ASR backends.

    Implement `transcribe()` to return word-timestamped transcript spans.
    The interface is intentionally minimal so backends are swappable.
    """

    @abstractmethod
    def transcribe(self, audio_path: str) -> list[TranscriptSpan]:
        """
        Transcribe *audio_path* and return timestamped segments.

        Each TranscriptSpan covers one logical sentence/phrase with
        word-level timestamps where the backend supports them.
        """
        ...


# ---------------------------------------------------------------------------
# OpenAI Whisper API provider
# ---------------------------------------------------------------------------


class WhisperAPIProvider(ASRProvider):
    """
    Transcribes audio using the OpenAI Whisper API.

    Cost: $0.006 / minute of audio.
    Returns segment-level + word-level timestamps (verbose_json).
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def transcribe(self, audio_path: str) -> list[TranscriptSpan]:
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("openai package not installed: pip install openai")

        client = OpenAI(api_key=self._api_key)
        path = Path(audio_path)

        logger.info("Whisper API transcribing: %s", path.name)

        with open(path, "rb") as f:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["segment", "word"],
            )

        spans: list[TranscriptSpan] = []
        for seg in (response.segments or []):
            words = [
                TranscriptWord(
                    word=w.word,
                    start_s=w.start,
                    end_s=w.end,
                    probability=getattr(w, "probability", 1.0),
                )
                for w in (seg.words or [])
            ]
            spans.append(
                TranscriptSpan(
                    start_s=seg.start,
                    end_s=seg.end,
                    text=seg.text.strip(),
                    words=words,
                )
            )

        logger.info("Whisper API: %d segments transcribed", len(spans))
        return spans


# ---------------------------------------------------------------------------
# Local faster-whisper provider
# ---------------------------------------------------------------------------


class FasterWhisperProvider(ASRProvider):
    """
    Transcribes audio using the local faster-whisper library.

    No API cost.  Requires downloading model weights on first run (~500 MB
    for 'medium', ~150 MB for 'small').

    Model is configurable via the ADLENS_FASTER_WHISPER_MODEL env var.
    Defaults to 'small' for speed; use 'medium' or 'large-v3' for accuracy.
    """

    def __init__(self, model_size: str = "small") -> None:
        self._model_size = model_size
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel  # type: ignore
            except ImportError:
                raise RuntimeError(
                    "faster-whisper not installed: pip install faster-whisper"
                )
            logger.info("Loading faster-whisper model: %s", self._model_size)
            self._model = WhisperModel(
                self._model_size,
                device="cpu",
                compute_type="int8",
                cpu_threads=1,
            )
        return self._model

    def transcribe(self, audio_path: str) -> list[TranscriptSpan]:
        model = self._load_model()
        logger.info("faster-whisper transcribing: %s", Path(audio_path).name)

        segments_iter, info = model.transcribe(
            audio_path,
            word_timestamps=True,
            vad_filter=False,
            language=None,  # auto-detect
        )

        spans: list[TranscriptSpan] = []
        for seg in segments_iter:
            words = []
            if seg.words:
                words = [
                    TranscriptWord(
                        word=w.word,
                        start_s=w.start,
                        end_s=w.end,
                        probability=w.probability,
                    )
                    for w in seg.words
                ]
            spans.append(
                TranscriptSpan(
                    start_s=seg.start,
                    end_s=seg.end,
                    text=seg.text.strip(),
                    words=words,
                )
            )

        logger.info("faster-whisper: %d segments", len(spans))
        return spans


# ---------------------------------------------------------------------------
# Free speech recognition provider (No API key needed)
# ---------------------------------------------------------------------------


class FreeSpeechRecognitionProvider(ASRProvider):
    """
    100% Free speech recognition provider using SpeechRecognition.

    No API key or GPU required. Transcribes audio instantly.
    """

    def transcribe(self, audio_path: str) -> list[TranscriptSpan]:
        try:
            import speech_recognition as sr
        except ImportError:
            raise RuntimeError("SpeechRecognition not installed: pip install SpeechRecognition")

        logger.info("Free ASR transcribing: %s", Path(audio_path).name)
        r = sr.Recognizer()
        try:
            with sr.AudioFile(audio_path) as source:
                audio = r.record(source)
            text = r.recognize_google(audio)
            logger.info("Free ASR transcribed text: '%s'", text)
            if text:
                return [TranscriptSpan(start_s=0.0, end_s=60.0, text=text)]
        except Exception as exc:
            logger.warning("Free ASR transcription error: %s", exc)

        return []


# ---------------------------------------------------------------------------
# Mock provider (for dev/testing without API keys)
# ---------------------------------------------------------------------------


class MockASRProvider(ASRProvider):
    """
    Returns a minimal stub transcript for testing.

    Inserts a fake midroll sponsor segment at 30% through the video
    to allow downstream pipeline testing.

    DO NOT use in production.  Clearly labelled as mock in logs.
    """

    def transcribe(self, audio_path: str) -> list[TranscriptSpan]:
        logger.warning(
            "[MOCK ASR] Returning stub transcript — not real transcription. "
            "Set ADLENS_ASR_PROVIDER=free, whisper_api, or faster_whisper for real results."
        )

        # Estimate a basic duration from filename (can't read audio without ffprobe)
        return [
            TranscriptSpan(
                start_s=0.0,
                end_s=5.0,
                text="Welcome back to the channel everyone.",
            ),
            TranscriptSpan(
                start_s=30.0,
                end_s=90.0,
                text=(
                    "This video is sponsored by ExampleBrand. "
                    "Use code MOCK20 for twenty percent off your first order. "
                    "Link in the description below. "
                    "Now back to the topic."
                ),
            ),
            TranscriptSpan(
                start_s=90.0,
                end_s=120.0,
                text="Alright so as I was saying earlier...",
            ),
        ]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_asr_provider(settings: "Settings") -> ASRProvider:
    """Return the ASR provider selected by configuration."""
    provider = settings.effective_asr_provider
    if provider == "whisper_api":
        return WhisperAPIProvider(api_key=settings.openai_api_key)
    elif provider == "faster_whisper":
        return FasterWhisperProvider(model_size="tiny")
    elif provider == "free" or provider == "google_free":
        return FreeSpeechRecognitionProvider()
    else:
        return FreeSpeechRecognitionProvider()



# ---------------------------------------------------------------------------
# Linguistic pattern analysis
# ---------------------------------------------------------------------------

# Keywords and patterns that indicate sponsored/ad content in speech.
# Organized by signal category for explainability.
LINGUISTIC_PATTERNS: dict[str, list[re.Pattern]] = {
    "sponsor_disclosure": [
        re.compile(r"\bsponsored\s+by\b", re.I),
        re.compile(r"\bbrought\s+to\s+you\s+by\b", re.I),
        re.compile(r"\bthis\s+(video|episode|podcast)\s+is\s+sponsored\b", re.I),
        re.compile(r"\bpartner(?:ed|ship)?\s+with\b", re.I),
        re.compile(r"\bin\s+partnership\s+with\b", re.I),
        re.compile(r"\bpaid\s+(promotion|partnership|collaboration)\b", re.I),
        re.compile(r"\baffiliate\b", re.I),
    ],
    "discount_code": [
        re.compile(r"\buse\s+(?:my\s+)?(?:code|coupon|promo)\b", re.I),
        re.compile(r"\bpromo\s+code\b", re.I),
        re.compile(r"\bdiscount\s+code\b", re.I),
        re.compile(r"\b(?:save|get)\s+\d+\s*(?:percent|%)\s+off\b", re.I),
        re.compile(r"\bexclusive\s+(?:discount|offer|deal)\b", re.I),
    ],
    "cta": [
        re.compile(r"\blink\s+in\s+(?:the\s+)?(?:bio|description|comments)\b", re.I),
        re.compile(r"\bcheck\s+(?:it|them|this)\s+out\b", re.I),
        re.compile(r"\bhead\s+(?:over\s+)?to\b", re.I),
        re.compile(r"\bvisit\s+(?:the\s+)?(?:website|site|link)\b", re.I),
        re.compile(r"\bdownload\s+(?:the\s+)?(?:app|extension|plugin)\b", re.I),
        re.compile(r"\bsign\s+up\s+(?:for\s+)?(?:free|today|now)\b", re.I),
        re.compile(r"\bfirst\s+\d+\s+(?:people|customers|viewers)\b", re.I),
    ],
    "product_mention": [
        re.compile(r"\bfree\s+trial\b", re.I),
        re.compile(r"\bsubscription\b", re.I),
        re.compile(r"\bget\s+started\s+(?:for\s+free)?\b", re.I),
        re.compile(r"\blimited\s+(?:time\s+)?offer\b", re.I),
        re.compile(r"\bmonth[s]?\s+free\b", re.I),
    ],
    "self_promo": [
        re.compile(r"\bjoin\s+(?:my\s+)?patreon\b", re.I),
        re.compile(r"\bmy\s+merch\b", re.I),
        re.compile(r"\bmerch\s+(?:store|shop|link)\b", re.I),
        re.compile(r"\bbecome\s+a\s+(?:member|patron)\b", re.I),
        re.compile(r"\bmembership\b", re.I),
        re.compile(r"\bnewsletter\b", re.I),
        re.compile(r"\bsubscribe\s+(?:to\s+)?(?:my|the|this)\b", re.I),
    ],
}

# Confidence score assigned to each category
CATEGORY_SCORES: dict[str, float] = {
    "sponsor_disclosure": 0.9,
    "discount_code": 0.85,
    "cta": 0.6,
    "product_mention": 0.55,
    "self_promo": 0.75,
}


class LinguisticAnalyzer:
    """
    Scans timestamped transcript spans for advertising-related signals.

    Returns a list of LinguisticEvidence objects, one per match.
    Evidence is kept at the span level so timestamps remain meaningful.
    """

    def analyze(self, spans: list[TranscriptSpan]) -> list[LinguisticEvidence]:
        """Scan all spans and return detected linguistic evidence."""
        evidence: list[LinguisticEvidence] = []

        for span in spans:
            for category, patterns in LINGUISTIC_PATTERNS.items():
                for pattern in patterns:
                    match = pattern.search(span.text)
                    if match:
                        evidence.append(
                            LinguisticEvidence(
                                start_s=span.start_s,
                                end_s=span.end_s,
                                text=span.text,
                                signal_type=category,
                                matched_pattern=pattern.pattern,
                                score=CATEGORY_SCORES.get(category, 0.5),
                            )
                        )
                        break  # one match per category per span is enough

        if evidence:
            logger.info(
                "Linguistic analysis: %d signals found in %d spans",
                len(evidence), len(spans),
            )
        else:
            logger.info("Linguistic analysis: no ad signals found")

        return evidence
