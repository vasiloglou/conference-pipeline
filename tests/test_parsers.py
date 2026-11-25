"""Tests for HTML parsers and scrapers."""

import pytest
import re

from bs4 import BeautifulSoup

from conference_pipeline.models import Presentation, PresentationStatus


class TestSlidesLiveIdExtraction:
    """Tests for SlidesLive ID extraction patterns."""

    def test_extract_slideslive_id_from_iframe(self):
        """Test extracting SlidesLive ID from iframe src."""
        html = """
        <html>
        <body>
            <iframe src="https://slideslive.com/embed/38943210"></iframe>
        </body>
        </html>
        """
        pattern = r"slideslive\.com/(?:embed/)?(?:presentation/)?(\d+)"
        match = re.search(pattern, html)
        assert match is not None
        assert match.group(1) == "38943210"

    def test_extract_slideslive_id_from_link(self):
        """Test extracting SlidesLive ID from link."""
        html = """
        <html>
        <body>
            <a href="https://slideslive.com/38943210">Watch Video</a>
        </body>
        </html>
        """
        pattern = r"slideslive\.com/(?:embed/)?(?:presentation/)?(\d+)"
        match = re.search(pattern, html)
        assert match is not None
        assert match.group(1) == "38943210"

    def test_extract_slideslive_id_from_data_attr(self):
        """Test extracting SlidesLive ID from data attribute."""
        html = """
        <html>
        <body>
            <div data-slideslive-id="38943210">Video Player</div>
        </body>
        </html>
        """
        soup = BeautifulSoup(html, "lxml")
        elem = soup.find(attrs={"data-slideslive-id": True})
        assert elem is not None
        assert elem["data-slideslive-id"] == "38943210"

    def test_extract_multiple_formats(self):
        """Test extracting SlidesLive IDs from various formats."""
        test_cases = [
            ("slideslive.com/38943210", "38943210"),
            ("slideslive.com/embed/38943210", "38943210"),
            ("slideslive.com/embed/presentation/38943210", "38943210"),
            ("https://slideslive.com/38943210/my-talk-title", "38943210"),
        ]

        pattern = r"slideslive\.com/(?:embed/)?(?:presentation/)?(\d+)"
        for url, expected_id in test_cases:
            match = re.search(pattern, url)
            assert match is not None, f"Failed to match: {url}"
            assert match.group(1) == expected_id


class TestScheduleScraper:
    """Tests for schedule scraper HTML parsing."""

    def test_extract_presentation_links_from_schedule(self):
        """Test extracting presentation links from schedule HTML."""
        html = """
        <html>
        <body>
            <div class="schedule">
                <a href="/virtual/2025/paper/12345">Paper 1</a>
                <a href="/virtual/2025/paper/12346">Paper 2</a>
                <a href="/virtual/2025/poster/67890">Poster 1</a>
            </div>
        </body>
        </html>
        """
        pattern = r'/virtual/\d+/(?:paper|poster|workshop)/\d+'
        matches = re.findall(pattern, html)
        assert len(matches) >= 2

    def test_extract_session_blocks(self):
        """Test extracting session blocks from schedule."""
        html = """
        <html>
        <body>
            <div class="session-block">
                <h3>Session A: Machine Learning</h3>
                <div class="papers">
                    <a href="/paper/1">Paper 1</a>
                    <a href="/paper/2">Paper 2</a>
                </div>
            </div>
            <div class="session-block">
                <h3>Session B: Computer Vision</h3>
                <div class="papers">
                    <a href="/paper/3">Paper 3</a>
                </div>
            </div>
        </body>
        </html>
        """
        soup = BeautifulSoup(html, "lxml")
        sessions = soup.find_all("div", class_="session-block")
        assert len(sessions) == 2


class TestURLPatterns:
    """Tests for URL pattern matching in scrapers."""

    def test_neurips_url_pattern(self):
        """Test NeurIPS URL pattern matching."""
        pattern = r'https://neurips\.cc/virtual/(\d+)/(?:paper|poster|workshop|session)/(\d+)'

        valid_urls = [
            "https://neurips.cc/virtual/2025/paper/12345",
            "https://neurips.cc/virtual/2024/poster/67890",
            "https://neurips.cc/virtual/2025/workshop/11111",
        ]

        for url in valid_urls:
            match = re.match(pattern, url)
            assert match is not None, f"Should match: {url}"
            assert match.group(1).isdigit()  # Year
            assert match.group(2).isdigit()  # ID

    def test_slideslive_url_pattern(self):
        """Test SlidesLive URL pattern matching."""
        pattern = r'slideslive\.com/(?:embed/)?(\d+)'

        valid_urls = [
            "https://slideslive.com/38943210",
            "https://slideslive.com/embed/38943210",
            "https://www.slideslive.com/38943210/presentation-title",
        ]

        for url in valid_urls:
            match = re.search(pattern, url)
            assert match is not None, f"Should match: {url}"
            assert match.group(1).isdigit()

    def test_arxiv_pdf_url_pattern(self):
        """Test arXiv PDF URL pattern matching."""
        pattern = r'arxiv\.org/pdf/(\d{4}\.\d{4,5})'

        valid_urls = [
            "https://arxiv.org/pdf/2301.12345",
            "https://arxiv.org/pdf/2301.12345v2.pdf",
            "https://export.arxiv.org/pdf/2301.12345",
        ]

        for url in valid_urls:
            match = re.search(pattern, url)
            assert match is not None, f"Should match: {url}"

    def test_doi_url_pattern(self):
        """Test DOI URL pattern matching."""
        pattern = r'doi\.org/(10\.\d{4,}/[^\s]+)'

        valid_urls = [
            "https://doi.org/10.1234/abc.123",
            "https://doi.org/10.1038/nature12373",
            "https://dx.doi.org/10.1145/1234567.1234568",
        ]

        for url in valid_urls:
            match = re.search(pattern, url)
            assert match is not None, f"Should match: {url}"


class TestHTMLEdgeCases:
    """Tests for edge cases in HTML parsing."""

    def test_malformed_html(self):
        """Test handling of malformed HTML."""
        malformed_html = """
        <html>
        <body>
            <div>Unclosed div
            <p>Unclosed paragraph
            <a href="test">Unclosed link
        """
        # Should not raise exception
        soup = BeautifulSoup(malformed_html, "lxml")
        assert soup is not None

    def test_html_with_scripts(self):
        """Test handling of HTML with script tags."""
        html = """
        <html>
        <body>
            <script>
                var data = {"video_id": "38943210"};
            </script>
            <h1>Test Paper</h1>
        </body>
        </html>
        """
        soup = BeautifulSoup(html, "lxml")
        h1 = soup.find("h1")
        assert h1 is not None
        assert h1.text == "Test Paper"

    def test_html_with_special_entities(self):
        """Test handling of HTML entities."""
        html = """
        <html>
        <body>
            <h1>Learning &amp; Understanding: A &lt;Framework&gt;</h1>
            <div class="authors">O'Brien &amp; M&uuml;ller</div>
        </body>
        </html>
        """
        soup = BeautifulSoup(html, "lxml")
        h1 = soup.find("h1")
        assert h1 is not None
        # HTML entities should be decoded
        assert "&amp;" not in h1.text
        assert "&" in h1.text or "Learning" in h1.text

    def test_empty_body(self):
        """Test handling of empty body."""
        html = "<html><head></head><body></body></html>"
        soup = BeautifulSoup(html, "lxml")
        body = soup.find("body")
        assert body is not None
        assert body.text.strip() == ""

    def test_deeply_nested_html(self):
        """Test handling of deeply nested HTML."""
        nested = "<div>" * 50 + "<h1>Deep Title</h1>" + "</div>" * 50
        html = f"<html><body>{nested}</body></html>"
        soup = BeautifulSoup(html, "lxml")
        h1 = soup.find("h1")
        assert h1 is not None
        assert h1.text == "Deep Title"


class TestPresentationModel:
    """Tests for Presentation model status tracking."""

    def test_presentation_status_values(self):
        """Test all presentation status values."""
        expected_statuses = [
            "discovered",
            "assets_downloaded",
            "transcribed",
            "slides_extracted",
            "ocr_done",
            "references_extracted",
            "papers_downloaded",
            "error",
        ]

        for status_value in expected_statuses:
            status = PresentationStatus(status_value)
            assert status.value == status_value

    def test_presentation_creation(self):
        """Test creating a presentation."""
        pres = Presentation(
            id="test-001",
            title="Test Paper",
            authors=["Author One", "Author Two"],
            neurips_url="https://neurips.cc/virtual/2025/paper/001",
        )
        assert pres.id == "test-001"
        assert pres.title == "Test Paper"
        assert len(pres.authors) == 2
        assert pres.status == PresentationStatus.DISCOVERED

    def test_presentation_with_slideslive(self):
        """Test presentation with SlidesLive info."""
        pres = Presentation(
            id="test-002",
            title="Another Paper",
            authors=["Author"],
            neurips_url="https://neurips.cc/virtual/2025/paper/002",
            slideslive_id="38943210",
            slideslive_url="https://slideslive.com/38943210",
            has_video=True,
            has_slides=True,
        )
        assert pres.slideslive_id == "38943210"
        assert pres.has_video is True
        assert pres.has_slides is True
