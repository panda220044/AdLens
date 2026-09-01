"""Tests for linguistic pattern analysis."""

from __future__ import annotations

import pytest

from adlens.asr import LinguisticAnalyzer
from adlens.models import TranscriptSpan


def make_span(text, start=0.0, end=10.0):
    return TranscriptSpan(start_s=start, end_s=end, text=text)


class TestLinguisticAnalyzer:
    def setup_method(self):
        self.analyzer = LinguisticAnalyzer()

    def test_sponsor_disclosure(self):
        spans = [make_span("This video is sponsored by TechCorp.")]
        evidence = self.analyzer.analyze(spans)
        types = [e.signal_type for e in evidence]
        assert "sponsor_disclosure" in types

    def test_brought_to_you(self):
        spans = [make_span("Brought to you by our partner Example.")]
        evidence = self.analyzer.analyze(spans)
        types = [e.signal_type for e in evidence]
        assert "sponsor_disclosure" in types

    def test_discount_code(self):
        spans = [make_span("Use code SAVE20 for twenty percent off.")]
        evidence = self.analyzer.analyze(spans)
        types = [e.signal_type for e in evidence]
        assert "discount_code" in types

    def test_cta_link_in_description(self):
        spans = [make_span("Check the link in the description below.")]
        evidence = self.analyzer.analyze(spans)
        types = [e.signal_type for e in evidence]
        assert "cta" in types

    def test_patreon_is_self_promo(self):
        spans = [make_span("If you like this content, join my Patreon.")]
        evidence = self.analyzer.analyze(spans)
        types = [e.signal_type for e in evidence]
        assert "self_promo" in types

    def test_no_ad_content(self):
        spans = [make_span("Today we're going to talk about machine learning.")]
        evidence = self.analyzer.analyze(spans)
        assert evidence == []

    def test_multiple_signals_in_one_span(self):
        spans = [
            make_span(
                "This episode is sponsored by Acme. Use code ACME10 to save. "
                "Link in the description."
            )
        ]
        evidence = self.analyzer.analyze(spans)
        types = {e.signal_type for e in evidence}
        # Should detect at least two distinct signal categories
        assert len(types) >= 2

    def test_high_score_for_explicit_disclosure(self):
        spans = [make_span("This video is sponsored by ExampleBrand.")]
        evidence = self.analyzer.analyze(spans)
        disclosure = next(
            (e for e in evidence if e.signal_type == "sponsor_disclosure"), None
        )
        assert disclosure is not None
        assert disclosure.score >= 0.8

    def test_timestamps_preserved(self):
        spans = [make_span("Sponsored by Acme.", start=45.0, end=90.0)]
        evidence = self.analyzer.analyze(spans)
        assert any(e.start_s == 45.0 and e.end_s == 90.0 for e in evidence)
