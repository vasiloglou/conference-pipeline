"""Schedule scraper for NeurIPS conference presentations."""

import asyncio
import re
from collections.abc import AsyncIterator

from bs4 import BeautifulSoup

from conference_pipeline.auth import NeuripsBrowserSession, NeuripsSession
from conference_pipeline.config import PipelineSettings
from conference_pipeline.models import Presentation, PresentationStatus, Session
from utils.logging import get_logger

logger = get_logger(__name__)


class ScheduleScraper:
    """Scrapes NeurIPS schedule to discover presentations."""

    def __init__(
        self,
        session: NeuripsSession,
        settings: PipelineSettings,
        browser_session: NeuripsBrowserSession | None = None,
    ):
        """Initialize schedule scraper.

        Args:
            session: Authenticated HTTP session.
            settings: Pipeline settings.
            browser_session: Optional browser session for JS-heavy pages.
        """
        self.session = session
        self.settings = settings
        self.browser_session = browser_session

    async def scrape_presentations_from_sessions(
        self,
        sessions: list[Session],
        storage=None,
    ) -> list[Presentation]:
        """Scrape presentations from a list of chosen sessions.

        Args:
            sessions: Sessions to scrape (should have chosen="Y").
            storage: Optional StorageBackend for checkpoint tracking.

        Returns:
            List of presentations with session metadata.
        """
        presentations = []
        seen_ids: set[str] = set()

        # Filter to pending sessions if storage is provided
        sessions_to_process = sessions
        if storage:
            sessions_to_process = storage.get_pending_sessions(sessions)
            logger.info(f"Resuming from checkpoint: {len(sessions_to_process)} sessions remaining")

        for sess in sessions_to_process:
            logger.info(f"Scraping session: {sess.title} ({sess.session_type.value})")

            # Mark session as started
            if storage:
                storage.mark_session_started(sess.id, sess.title, sess.session_type.value)

            try:
                session_presentations = await self._scrape_session_page(sess)

                for pres in session_presentations:
                    if pres.id not in seen_ids:
                        # Add session info to presentation
                        pres.session_title = sess.title
                        pres.track = sess.session_type.value
                        presentations.append(pres)
                        seen_ids.add(pres.id)

                logger.info(f"Found {len(session_presentations)} presentations in {sess.title}")

                # Mark session as completed
                if storage:
                    storage.mark_session_completed(sess.id, len(session_presentations))

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error scraping session {sess.id}: {error_msg}")

                # Mark session as completed with error
                if storage:
                    storage.mark_session_completed(sess.id, 0, error_msg)

            # Rate limiting
            await asyncio.sleep(self.settings.request_delay_seconds)

        logger.info(f"Total presentations from sessions: {len(presentations)}")
        return presentations

    async def _scrape_session_page(self, sess: Session) -> list[Presentation]:
        """Scrape presentations from a single session page.

        Args:
            sess: Session to scrape.

        Returns:
            List of presentations in the session.
        """
        presentations = []

        try:
            if self.browser_session:
                html = await self.browser_session.get_rendered_html(sess.url)
            else:
                response = await self.session.get_full_url(sess.url)
                html = response.text

            soup = BeautifulSoup(html, "lxml")

            # Look for presentation elements
            pres_elements = soup.find_all(
                class_=re.compile(r"presentation|paper|talk|event-item|card", re.I)
            )

            for elem in pres_elements:
                pres = self._parse_presentation_element(elem)
                if pres:
                    presentations.append(pres)

            # Also try table rows
            if not presentations:
                for row in soup.find_all("tr"):
                    pres = self._parse_table_row(row)
                    if pres:
                        presentations.append(pres)

            # Also try links
            if not presentations:
                for link in soup.find_all("a", href=re.compile(r"/paper|/poster|/talk|/event")):
                    pres = self._parse_presentation_link(link)
                    if pres:
                        presentations.append(pres)

        except Exception as e:
            logger.error(f"Error scraping session page {sess.url}: {e}")

        return presentations

    async def scrape_all_presentations(self) -> list[Presentation]:
        """Scrape all presentations from the schedule (legacy method).

        Returns:
            List of discovered presentations.
        """
        presentations = []

        async for calendar_url in self._iter_calendar_urls():
            logger.info(f"Scraping calendar page: {calendar_url}")
            page_presentations = await self._scrape_calendar_page(calendar_url)
            presentations.extend(page_presentations)
            logger.info(f"Found {len(page_presentations)} presentations on {calendar_url}")

            # Rate limiting
            await asyncio.sleep(self.settings.request_delay_seconds)

        logger.info(f"Total presentations discovered: {len(presentations)}")
        return presentations

    async def _iter_calendar_urls(self) -> AsyncIterator[str]:
        """Generate URLs for calendar/schedule pages.

        Yields:
            Calendar page URLs.
        """
        base = self.settings.virtual_base_url

        # Common NeurIPS schedule URL patterns
        schedule_patterns = [
            f"{base}/schedule",
            f"{base}/events/oral",
            f"{base}/events/spotlight",
            f"{base}/events/poster",
            f"{base}/events/tutorial",
            f"{base}/events/workshop",
            f"{base}/events/invited-talk",
        ]

        for url in schedule_patterns:
            yield url

        # Try to discover additional schedule pages from the main schedule
        try:
            response = await self.session.get_full_url(f"{base}/schedule")
            soup = BeautifulSoup(response.text, "lxml")

            # Look for links to other schedule pages or days
            for link in soup.find_all("a", href=True):
                href = link["href"]
                if any(keyword in href.lower() for keyword in ["schedule", "event", "session"]):
                    if href.startswith("/"):
                        full_url = f"{self.settings.neurips_base_url}{href}"
                    elif not href.startswith("http"):
                        full_url = f"{base}/{href}"
                    else:
                        full_url = href

                    if full_url not in schedule_patterns:
                        yield full_url

        except Exception as e:
            logger.warning(f"Error discovering schedule pages: {e}")

    async def _scrape_calendar_page(self, url: str) -> list[Presentation]:
        """Scrape presentations from a calendar page.

        Args:
            url: Calendar page URL.

        Returns:
            List of presentations found on the page.
        """
        try:
            # Use browser session for JS-heavy pages if available
            if self.browser_session:
                html = await self.browser_session.get_rendered_html(url)
            else:
                response = await self.session.get_full_url(url)
                html = response.text

            soup = BeautifulSoup(html, "lxml")
            presentations = []

            # Try different parsing strategies based on page structure

            # Strategy 1: Look for presentation cards/items
            presentation_elements = soup.find_all(
                class_=re.compile(r"presentation|paper|event-item|talk|poster", re.I)
            )

            for elem in presentation_elements:
                pres = self._parse_presentation_element(elem)
                if pres:
                    presentations.append(pres)

            # Strategy 2: Look for table rows with presentations
            if not presentations:
                for row in soup.find_all("tr"):
                    pres = self._parse_table_row(row)
                    if pres:
                        presentations.append(pres)

            # Strategy 3: Look for links to individual presentation pages
            if not presentations:
                for link in soup.find_all("a", href=re.compile(r"/paper|/poster|/talk|/event")):
                    pres = self._parse_presentation_link(link)
                    if pres:
                        presentations.append(pres)

            return presentations

        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return []

    def _parse_presentation_element(self, elem) -> Presentation | None:
        """Parse a presentation from an HTML element.

        Args:
            elem: BeautifulSoup element.

        Returns:
            Presentation if parsed successfully, None otherwise.
        """
        try:
            # Find title
            title_elem = elem.find(class_=re.compile(r"title", re.I)) or elem.find(["h3", "h4", "h5"])
            if not title_elem:
                return None

            title = title_elem.get_text(strip=True)
            if not title:
                return None

            # Find link to presentation page
            link_elem = title_elem.find("a") or elem.find("a", href=True)
            neurips_url = ""
            pres_id = ""

            if link_elem and link_elem.get("href"):
                href = link_elem["href"]
                if href.startswith("/"):
                    neurips_url = f"{self.settings.neurips_base_url}{href}"
                elif href.startswith("http"):
                    neurips_url = href
                else:
                    neurips_url = f"{self.settings.virtual_base_url}/{href}"

                # Extract ID from URL
                id_match = re.search(r"/(\d+)/?$", neurips_url)
                if id_match:
                    pres_id = id_match.group(1)
                else:
                    pres_id = re.sub(r"[^a-zA-Z0-9]", "_", title)[:50]

            if not pres_id:
                pres_id = re.sub(r"[^a-zA-Z0-9]", "_", title)[:50]

            # Find authors
            authors = []
            authors_elem = elem.find(class_=re.compile(r"author", re.I))
            if authors_elem:
                authors_text = authors_elem.get_text(strip=True)
                # Split by common separators
                authors = [a.strip() for a in re.split(r"[,;·]", authors_text) if a.strip()]

            # Find session/track info
            session_title = None
            track = None
            session_elem = elem.find(class_=re.compile(r"session|track", re.I))
            if session_elem:
                session_title = session_elem.get_text(strip=True)

            return Presentation(
                id=pres_id,
                title=title,
                authors=authors,
                session_title=session_title,
                track=track,
                neurips_url=neurips_url,
                status=PresentationStatus.DISCOVERED,
            )

        except Exception as e:
            logger.debug(f"Error parsing presentation element: {e}")
            return None

    def _parse_table_row(self, row) -> Presentation | None:
        """Parse a presentation from a table row.

        Args:
            row: BeautifulSoup table row element.

        Returns:
            Presentation if parsed successfully, None otherwise.
        """
        try:
            cells = row.find_all(["td", "th"])
            if len(cells) < 2:
                return None

            # Try to find title and link
            title = None
            neurips_url = ""
            pres_id = ""

            for cell in cells:
                link = cell.find("a", href=True)
                if link:
                    text = link.get_text(strip=True)
                    if len(text) > 10:  # Likely a title
                        title = text
                        href = link["href"]
                        if href.startswith("/"):
                            neurips_url = f"{self.settings.neurips_base_url}{href}"
                        elif href.startswith("http"):
                            neurips_url = href

                        id_match = re.search(r"/(\d+)/?$", neurips_url)
                        if id_match:
                            pres_id = id_match.group(1)
                        break

            if not title:
                return None

            if not pres_id:
                pres_id = re.sub(r"[^a-zA-Z0-9]", "_", title)[:50]

            return Presentation(
                id=pres_id,
                title=title,
                authors=[],
                neurips_url=neurips_url,
                status=PresentationStatus.DISCOVERED,
            )

        except Exception as e:
            logger.debug(f"Error parsing table row: {e}")
            return None

    def _parse_presentation_link(self, link) -> Presentation | None:
        """Parse a presentation from a link element.

        Args:
            link: BeautifulSoup link element.

        Returns:
            Presentation if parsed successfully, None otherwise.
        """
        try:
            href = link.get("href", "")
            title = link.get_text(strip=True)

            if not title or len(title) < 5:
                return None

            if href.startswith("/"):
                neurips_url = f"{self.settings.neurips_base_url}{href}"
            elif href.startswith("http"):
                neurips_url = href
            else:
                neurips_url = f"{self.settings.virtual_base_url}/{href}"

            id_match = re.search(r"/(\d+)/?$", neurips_url)
            if id_match:
                pres_id = id_match.group(1)
            else:
                pres_id = re.sub(r"[^a-zA-Z0-9]", "_", title)[:50]

            return Presentation(
                id=pres_id,
                title=title,
                authors=[],
                neurips_url=neurips_url,
                status=PresentationStatus.DISCOVERED,
            )

        except Exception as e:
            logger.debug(f"Error parsing presentation link: {e}")
            return None

    async def scrape_presentation_details(self, pres: Presentation) -> Presentation:
        """Fetch and parse detailed presentation page.

        Args:
            pres: Presentation with basic info.

        Returns:
            Presentation with enriched details.
        """
        if not pres.neurips_url:
            return pres

        try:
            if self.browser_session:
                html = await self.browser_session.get_rendered_html(pres.neurips_url)
            else:
                response = await self.session.get_full_url(pres.neurips_url)
                html = response.text

            soup = BeautifulSoup(html, "lxml")

            # Try to extract more details
            # Authors
            if not pres.authors:
                authors_elem = soup.find(class_=re.compile(r"author", re.I))
                if authors_elem:
                    authors_text = authors_elem.get_text(strip=True)
                    pres.authors = [a.strip() for a in re.split(r"[,;·]", authors_text) if a.strip()]

            # Abstract (could be useful for reference extraction)
            abstract_elem = soup.find(class_=re.compile(r"abstract", re.I))
            if abstract_elem:
                # Store abstract if needed in the future
                pass

            # Session info
            session_elem = soup.find(class_=re.compile(r"session", re.I))
            if session_elem and not pres.session_title:
                pres.session_title = session_elem.get_text(strip=True)

            return pres

        except Exception as e:
            logger.warning(f"Error fetching presentation details for {pres.id}: {e}")
            return pres
