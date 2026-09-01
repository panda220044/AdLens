"""
Cost and latency tracker for AdLens.

Tracks all API calls and estimates USD cost based on known pricing.
Updated whenever pricing changes — make model configurable in config.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock


# ---------------------------------------------------------------------------
# Pricing table (USD)  —  update when OpenAI pricing changes
# ---------------------------------------------------------------------------

PRICING: dict[str, float] = {
    # Whisper API: per second of audio
    "whisper_api_per_second": 0.006 / 60.0,
    # GPT-4o-mini input tokens (per token)
    "gpt4o_mini_input_per_token": 0.15 / 1_000_000,
    # GPT-4o-mini output tokens (per token)
    "gpt4o_mini_output_per_token": 0.60 / 1_000_000,
    # Approximate tokens per image at detail=low
    "gpt4o_mini_image_low_tokens": 2833,
    # Approximate tokens per image at detail=high (1080p, ~6 tiles)
    "gpt4o_mini_image_high_tokens": 36835,
}


@dataclass
class CostTracker:
    """
    Thread-safe tracker for API cost and call counts.

    Usage::

        tracker = CostTracker()
        tracker.record_whisper(duration_s=120.0)
        tracker.record_vision_call(n_images=5, detail="low")
        print(tracker.total_cost_usd)
    """

    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _model_calls: int = field(default=0, init=False)
    _frames_sampled: int = field(default=0, init=False)
    _whisper_seconds: float = field(default=0.0, init=False)
    _vision_tokens: int = field(default=0, init=False)
    _text_input_tokens: int = field(default=0, init=False)
    _text_output_tokens: int = field(default=0, init=False)
    _start_time: float = field(default_factory=time.monotonic, init=False)

    # ------------------------------------------------------------------

    def record_whisper(self, duration_s: float) -> None:
        """Record a Whisper API transcription."""
        with self._lock:
            self._whisper_seconds += duration_s
            self._model_calls += 1

    def record_vision_call(self, n_images: int, detail: str = "low") -> None:
        """Record a GPT-4o-mini vision call with n_images images."""
        key = f"gpt4o_mini_image_{detail}_tokens"
        tokens_per_image = PRICING.get(key, PRICING["gpt4o_mini_image_low_tokens"])
        with self._lock:
            self._vision_tokens += n_images * int(tokens_per_image)
            self._model_calls += 1

    def record_text_call(self, input_tokens: int, output_tokens: int) -> None:
        """Record a GPT-4o-mini text-only call."""
        with self._lock:
            self._text_input_tokens += input_tokens
            self._text_output_tokens += output_tokens
            self._model_calls += 1

    def record_frames(self, n: int) -> None:
        """Record sampled frames (not a model call)."""
        with self._lock:
            self._frames_sampled += n

    # ------------------------------------------------------------------

    @property
    def model_calls(self) -> int:
        return self._model_calls

    @property
    def frames_sampled(self) -> int:
        return self._frames_sampled

    @property
    def wall_clock_s(self) -> float:
        return time.monotonic() - self._start_time

    @property
    def total_cost_usd(self) -> float:
        whisper_cost = self._whisper_seconds * PRICING["whisper_api_per_second"]
        vision_cost = (
            self._vision_tokens * PRICING["gpt4o_mini_input_per_token"]
        )
        text_cost = (
            self._text_input_tokens * PRICING["gpt4o_mini_input_per_token"]
            + self._text_output_tokens * PRICING["gpt4o_mini_output_per_token"]
        )
        return whisper_cost + vision_cost + text_cost

    def to_dict(self) -> dict[str, float | int]:
        return {
            "wall_clock_s": round(self.wall_clock_s, 2),
            "estimated_cost_usd": round(self.total_cost_usd, 6),
            "frames_sampled": self._frames_sampled,
            "model_calls": self._model_calls,
        }
