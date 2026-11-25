"""Tests for reference extraction regex patterns and parsers."""

import pytest

from conference_pipeline.models import OcrResult, TranscriptChunk
from conference_pipeline.reference_extractor import ExtractionPatterns, ReferenceExtractor


class TestExtractionPatterns:
    """Tests for regex patterns used in reference extraction."""

    def setup_method(self):
        """Set up test patterns."""
        self.patterns = ExtractionPatterns()

    # URL patterns
    def test_url_pattern_https(self):
        """Test HTTPS URL extraction."""
        text = "Check out https://arxiv.org/abs/2301.12345 for details"
        matches = list(self.patterns.url.finditer(text))
        assert len(matches) == 1
        assert matches[0].group(0) == "https://arxiv.org/abs/2301.12345"

    def test_url_pattern_http(self):
        """Test HTTP URL extraction."""
        text = "Link: http://example.com/paper.pdf"
        matches = list(self.patterns.url.finditer(text))
        assert len(matches) == 1
        assert matches[0].group(0) == "http://example.com/paper.pdf"

    def test_url_pattern_multiple(self):
        """Test multiple URLs in one text."""
        text = "See https://arxiv.org/abs/123 and http://doi.org/10.1234"
        matches = list(self.patterns.url.finditer(text))
        assert len(matches) == 2

    def test_url_pattern_with_punctuation(self):
        """Test URL followed by punctuation."""
        text = "Visit https://example.com/paper."
        matches = list(self.patterns.url.finditer(text))
        assert len(matches) == 1
        # Note: The pattern may include the period, which is cleaned up later
        assert "https://example.com/paper" in matches[0].group(0)

    # arXiv patterns - new format
    def test_arxiv_new_format_basic(self):
        """Test new arXiv ID format (YYMM.NNNNN)."""
        text = "arXiv:2301.12345"
        matches = list(self.patterns.arxiv_new.finditer(text))
        assert len(matches) == 1
        assert matches[0].group(1) == "2301.12345"

    def test_arxiv_new_format_with_version(self):
        """Test new arXiv ID with version number."""
        text = "See 2301.12345v2 for updates"
        matches = list(self.patterns.arxiv_new.finditer(text))
        assert len(matches) == 1
        assert matches[0].group(1) == "2301.12345v2"

    def test_arxiv_new_format_five_digits(self):
        """Test new arXiv ID with 5-digit number."""
        text = "arxiv: 2301.123456"
        matches = list(self.patterns.arxiv_new.finditer(text))
        assert len(matches) == 1
        assert "2301.12345" in matches[0].group(1)

    def test_arxiv_new_format_in_url(self):
        """Test extracting arXiv ID from URL."""
        text = "https://arxiv.org/abs/2301.12345"
        matches = list(self.patterns.arxiv_url.finditer(text))
        assert len(matches) == 1
        assert matches[0].group(1) == "2301.12345"

    # arXiv patterns - old format
    def test_arxiv_old_format(self):
        """Test old arXiv ID format (category/YYYYNNN)."""
        text = "arxiv:cs.LG/0701234"
        matches = list(self.patterns.arxiv_old.finditer(text))
        assert len(matches) == 1
        assert matches[0].group(1) == "cs.LG/0701234"

    def test_arxiv_old_format_hep(self):
        """Test old arXiv format with hep category."""
        text = "See arxiv/hep-th/9901001"
        matches = list(self.patterns.arxiv_old.finditer(text))
        assert len(matches) == 1
        assert "hep-th/9901001" in matches[0].group(1)

    # DOI patterns
    def test_doi_basic(self):
        """Test basic DOI extraction."""
        text = "doi:10.1234/example.123"
        matches = list(self.patterns.doi.finditer(text))
        assert len(matches) == 1
        assert "10.1234/example.123" in matches[0].group(0)

    def test_doi_with_prefix(self):
        """Test DOI with 'doi:' prefix."""
        text = "DOI: 10.1038/nature12373"
        matches = list(self.patterns.doi.finditer(text))
        assert len(matches) == 1
        assert "10.1038/nature12373" in matches[0].group(0)

    def test_doi_in_url(self):
        """Test DOI in URL format."""
        text = "https://doi.org/10.1145/1234567.1234568"
        matches = list(self.patterns.doi.finditer(text))
        assert len(matches) == 1
        assert "10.1145/1234567.1234568" in matches[0].group(0)

    def test_doi_complex(self):
        """Test DOI with complex suffix."""
        text = "10.1000/xyz123-456.789"
        matches = list(self.patterns.doi.finditer(text))
        assert len(matches) == 1

    # Citation patterns
    def test_citation_etal_basic(self):
        """Test 'Author et al., Year' pattern."""
        text = "According to Smith et al., 2023"
        matches = list(self.patterns.citation_etal.finditer(text))
        assert len(matches) == 1
        assert matches[0].group(1) == "Smith"
        assert matches[0].group(2) == "2023"

    def test_citation_etal_parentheses(self):
        """Test citation with year in parentheses."""
        text = "This work (Johnson et al. (2022))"
        matches = list(self.patterns.citation_etal.finditer(text))
        assert len(matches) == 1
        assert matches[0].group(1) == "Johnson"
        assert matches[0].group(2) == "2022"

    def test_citation_multi_author(self):
        """Test multi-author citation."""
        text = "Smith and Jones, 2021"
        matches = list(self.patterns.citation_multi.finditer(text))
        assert len(matches) == 1
        assert "Smith" in matches[0].group(1)

    def test_citation_three_authors(self):
        """Test three-author citation with '&'."""
        text = "Smith & Jones & Brown, 2020"
        matches = list(self.patterns.citation_multi.finditer(text))
        assert len(matches) == 1

    # Venue patterns
    def test_venue_neurips(self):
        """Test NeurIPS venue pattern."""
        text = "Published at NeurIPS 2023"
        matches = list(self.patterns.venue.finditer(text))
        assert len(matches) == 1
        assert matches[0].group(1).lower() == "neurips"
        assert matches[0].group(2) == "2023"

    def test_venue_icml(self):
        """Test ICML venue pattern."""
        text = "Accepted to ICML 2024"
        matches = list(self.patterns.venue.finditer(text))
        assert len(matches) == 1
        assert matches[0].group(1).lower() == "icml"
        assert matches[0].group(2) == "2024"

    def test_venue_multiple(self):
        """Test multiple venue mentions."""
        text = "Presented at NeurIPS 2022 and ICLR 2023"
        matches = list(self.patterns.venue.finditer(text))
        assert len(matches) == 2


class TestReferenceExtractor:
    """Tests for the ReferenceExtractor class."""

    def setup_method(self):
        """Set up test extractor."""
        self.extractor = ReferenceExtractor()

    def test_extract_from_ocr_with_arxiv(self):
        """Test extraction from OCR with arXiv ID."""
        ocr = OcrResult(
            presentation_id="test-123",
            slide_index=1,
            image_path="test.jpg",
            raw_text="This builds on arxiv:2301.12345 work",
        )
        candidates = self.extractor.extract_from_ocr(ocr)
        assert len(candidates) >= 1
        arxiv_ids = [c.arxiv_id for c in candidates if c.arxiv_id]
        assert "2301.12345" in arxiv_ids

    def test_extract_from_ocr_with_doi(self):
        """Test extraction from OCR with DOI."""
        ocr = OcrResult(
            presentation_id="test-123",
            slide_index=2,
            image_path="test.jpg",
            raw_text="See DOI: 10.1234/example.567",
        )
        candidates = self.extractor.extract_from_ocr(ocr)
        assert len(candidates) >= 1
        dois = [c.doi for c in candidates if c.doi]
        assert any("10.1234" in d for d in dois)

    def test_extract_from_ocr_with_url(self):
        """Test extraction from OCR with URL."""
        ocr = OcrResult(
            presentation_id="test-123",
            slide_index=3,
            image_path="test.jpg",
            raw_text="More info: https://arxiv.org/abs/2301.99999",
        )
        candidates = self.extractor.extract_from_ocr(ocr)
        assert len(candidates) >= 1
        urls = [c.detected_url for c in candidates if c.detected_url]
        assert any("arxiv.org" in u for u in urls)

    def test_extract_from_ocr_empty_text(self):
        """Test extraction from empty OCR text."""
        ocr = OcrResult(
            presentation_id="test-123",
            slide_index=1,
            image_path="test.jpg",
            raw_text="",
        )
        candidates = self.extractor.extract_from_ocr(ocr)
        assert len(candidates) == 0

    def test_extract_from_transcript(self):
        """Test extraction from transcript chunks."""
        chunks = [
            TranscriptChunk(
                index=0,
                audio_path="test.mp3",
                text="We use the method from Smith et al., 2022",
            ),
            TranscriptChunk(
                index=1,
                audio_path="test.mp3",
                text="You can find it on arxiv 2301.55555",
            ),
        ]
        candidates = self.extractor.extract_from_transcript(chunks, "test-123")
        assert len(candidates) >= 2

    def test_deduplication(self):
        """Test that duplicate candidates are removed."""
        ocr = OcrResult(
            presentation_id="test-123",
            slide_index=1,
            image_path="test.jpg",
            raw_text="arxiv:2301.12345 and again arxiv:2301.12345",
        )
        candidates = self.extractor.extract_from_ocr(ocr)
        arxiv_ids = [c.arxiv_id for c in candidates if c.arxiv_id]
        # Should be deduplicated
        assert arxiv_ids.count("2301.12345") == 1

    def test_irrelevant_url_filtering(self):
        """Test that irrelevant URLs are filtered out."""
        ocr = OcrResult(
            presentation_id="test-123",
            slide_index=1,
            image_path="test.jpg",
            raw_text="Check twitter.com/user and https://arxiv.org/abs/123",
        )
        candidates = self.extractor.extract_from_ocr(ocr)
        urls = [c.detected_url for c in candidates if c.detected_url]
        # Twitter should be filtered
        assert not any("twitter.com" in u for u in urls)
        # arXiv should remain
        assert any("arxiv.org" in u for u in urls)

    def test_context_extraction(self):
        """Test that surrounding context is captured."""
        ocr = OcrResult(
            presentation_id="test-123",
            slide_index=1,
            image_path="test.jpg",
            raw_text="Based on the seminal work arxiv:2301.12345 which introduced...",
        )
        candidates = self.extractor.extract_from_ocr(ocr)
        assert len(candidates) >= 1
        # Context should include surrounding text
        assert "seminal" in candidates[0].raw_text or "introduced" in candidates[0].raw_text

    def test_merge_candidates(self):
        """Test merging candidates from multiple sources."""
        ocr_candidates = self.extractor.extract_from_ocr(
            OcrResult(
                presentation_id="test-123",
                slide_index=1,
                image_path="test.jpg",
                raw_text="arxiv:2301.11111",
            )
        )
        transcript_candidates = self.extractor.extract_from_transcript(
            [
                TranscriptChunk(
                    index=0,
                    audio_path="test.mp3",
                    text="arxiv 2301.22222",
                )
            ],
            "test-123",
        )
        merged = self.extractor.merge_candidates(ocr_candidates, transcript_candidates)
        arxiv_ids = [c.arxiv_id for c in merged if c.arxiv_id]
        assert "2301.11111" in arxiv_ids
        assert "2301.22222" in arxiv_ids


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def setup_method(self):
        """Set up test extractor."""
        self.extractor = ReferenceExtractor()

    def test_malformed_arxiv(self):
        """Test handling of malformed arXiv IDs."""
        ocr = OcrResult(
            presentation_id="test-123",
            slide_index=1,
            image_path="test.jpg",
            raw_text="arxiv:not-a-real-id",
        )
        candidates = self.extractor.extract_from_ocr(ocr)
        # Should not crash, may or may not find matches
        assert isinstance(candidates, list)

    def test_unicode_text(self):
        """Test handling of Unicode text."""
        ocr = OcrResult(
            presentation_id="test-123",
            slide_index=1,
            image_path="test.jpg",
            raw_text="Müller et al., 2023 used arxiv:2301.12345",
        )
        candidates = self.extractor.extract_from_ocr(ocr)
        assert len(candidates) >= 1

    def test_very_long_text(self):
        """Test handling of very long text."""
        long_text = "arxiv:2301.12345 " + "x" * 10000 + " doi:10.1234/test"
        ocr = OcrResult(
            presentation_id="test-123",
            slide_index=1,
            image_path="test.jpg",
            raw_text=long_text,
        )
        candidates = self.extractor.extract_from_ocr(ocr)
        # Should find both references despite long text
        assert len(candidates) >= 2

    def test_special_characters(self):
        """Test handling of special characters in text."""
        ocr = OcrResult(
            presentation_id="test-123",
            slide_index=1,
            image_path="test.jpg",
            raw_text="See [arXiv:2301.12345] and <doi:10.1234/test>",
        )
        candidates = self.extractor.extract_from_ocr(ocr)
        assert len(candidates) >= 1

    def test_newlines_in_text(self):
        """Test handling of newlines in text."""
        ocr = OcrResult(
            presentation_id="test-123",
            slide_index=1,
            image_path="test.jpg",
            raw_text="Line 1\narxiv:2301.12345\nLine 3",
        )
        candidates = self.extractor.extract_from_ocr(ocr)
        assert len(candidates) >= 1
