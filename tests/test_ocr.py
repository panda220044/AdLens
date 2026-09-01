"""Tests for OCR ad text filtering."""

from __future__ import annotations

import pytest

from adlens.ocr import AdTextFilter


class TestAdTextFilter:
    def setup_method(self):
        self.filter = AdTextFilter()

    def test_sponsored_label(self):
        ad_texts, score = self.filter.score(["SPONSORED"])
        assert "SPONSORED" in ad_texts
        assert score > 0

    def test_discount_code(self):
        ad_texts, score = self.filter.score(["Use code: SAVE20"])
        assert len(ad_texts) > 0
        assert score >= 0.3

    def test_cta_swipe_up(self):
        ad_texts, score = self.filter.score(["SWIPE UP to shop"])
        assert len(ad_texts) > 0

    def test_free_trial(self):
        ad_texts, score = self.filter.score(["Start your free trial today"])
        assert len(ad_texts) > 0

    def test_normal_text_no_match(self):
        texts = ["Hello everyone", "Today's topic", "Machine learning basics"]
        ad_texts, score = self.filter.score(texts)
        assert ad_texts == []
        assert score == 0.0

    def test_multiple_signals_higher_score(self):
        texts = ["SPONSORED", "Use code DEAL50", "FREE TRIAL", "Shop Now"]
        _, score_multi = self.filter.score(texts)
        _, score_single = self.filter.score(["SPONSORED"])
        assert score_multi >= score_single

    def test_score_bounded(self):
        # Score must always be in [0, 1]
        many_texts = ["SPONSORED", "50% OFF", "Free Trial", "Shop Now",
                      "Use code X", "Download Now", "Visit site", "Link in bio"]
        _, score = self.filter.score(many_texts)
        assert 0.0 <= score <= 1.0
