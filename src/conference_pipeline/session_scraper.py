"""Session scraper for NeurIPS conference.

Discovers sessions by type (tutorials, workshops, keynotes, etc.) and outputs
to JSON for manual filtering before presentation discovery.
"""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

from conference_pipeline.auth import NeuripsBrowserSession, NeuripsSession
from conference_pipeline.config import PipelineSettings
from conference_pipeline.models import Session, SessionType, TARGET_SESSION_TYPES
from utils.logging import get_logger
from utils.concurrency import RateLimiter

logger = get_logger(__name__)


# URL patterns for different session types
SESSION_URL_PATTERNS = {
    SessionType.TUTORIAL: [r"/tutorial", r"tutorials"],
    SessionType.WORKSHOP: [r"/workshop", r"workshops"],
    SessionType.ORAL: [r"/oral", r"orals"],
    SessionType.KEYNOTE: [r"/keynote", r"invited.?talk", r"plenary"],
    SessionType.EXPO: [r"/expo", r"sponsor"],
    SessionType.EXPO_TALK_PANEL: [r"expo.*talk.*panel", r"panel.*expo"],
    SessionType.EXHIBITOR_SPOT_TALK: [r"exhibitor.*spot", r"spot.*talk"],
    SessionType.EXPO_WORKSHOP: [r"expo.*workshop"],
    SessionType.WELCOME_REMARKS: [r"welcome.*remark", r"opening.*remark", r"remarks.*welcome"],
    SessionType.TEST_OF_TIME: [r"test.?of.?time", r"classic.?paper"],
    SessionType.HINTON_AWARD: [r"hinton", r"award"],
    SessionType.POSTER: [r"/poster"],
    SessionType.SPOTLIGHT: [r"/spotlight"],
    SessionType.INVITED_TALK: [r"invited"],
    SessionType.AFFINITY_WORKSHOP: [r"affinity"],
    SessionType.COMPETITION: [r"competition", r"challenge"],
    SessionType.MEETUP: [r"meetup"],
    SessionType.SOCIALS: [r"social"],
}

# Title patterns for session type classification
SESSION_TITLE_PATTERNS = {
    SessionType.TUTORIAL: [r"^tutorial", r"tutorial:"],
    SessionType.WORKSHOP: [r"^workshop", r"workshop:"],
    SessionType.ORAL: [r"^oral", r"oral session"],
    SessionType.KEYNOTE: [r"keynote", r"invited talk", r"plenary"],
    SessionType.EXPO: [r"expo", r"sponsor", r"industry"],
    SessionType.EXPO_TALK_PANEL: [r"expo talk panel", r"expo panel"],
    SessionType.EXHIBITOR_SPOT_TALK: [r"exhibitor spot talk", r"spot talk"],
    SessionType.EXPO_WORKSHOP: [r"expo workshop"],
    SessionType.WELCOME_REMARKS: [r"welcome remark", r"opening remark", r"remarks:.*welcome"],
    SessionType.TEST_OF_TIME: [r"test of time", r"classic paper"],
    SessionType.HINTON_AWARD: [r"hinton", r"award"],
    SessionType.SPOTLIGHT: [r"spotlight"],
    SessionType.AFFINITY_WORKSHOP: [r"affinity"],
    SessionType.COMPETITION: [r"competition", r"challenge"],
}

# CSS class patterns used on the NeurIPS schedule page (eventsession divs)
# These map CSS classes to session types for parsing the schedule grid
# IMPORTANT: More specific patterns (expo-workshop, affinity-workshop) must come
# BEFORE less specific patterns (workshop) to avoid incorrect matching
SESSION_CSS_CLASS_PATTERNS = {
    # More specific types first
    SessionType.EXPO_WORKSHOP: "expo-workshop",
    SessionType.AFFINITY_WORKSHOP: "affinity-workshop",
    SessionType.EXPO_TALK_PANEL: "expo-talk-panel",
    SessionType.EXHIBITOR_SPOT_TALK: "exhibitor-spot-talks",
    # Then general types
    SessionType.TUTORIAL: "tutorial",
    SessionType.WORKSHOP: "workshop",  # Must come after expo-workshop, affinity-workshop
    SessionType.ORAL: "oral-session",
    SessionType.KEYNOTE: "invited-talk",
    SessionType.WELCOME_REMARKS: "remarks",
    SessionType.TEST_OF_TIME: "test-of-time",
    SessionType.HINTON_AWARD: "award",
    SessionType.COMPETITION: "competition",
}


class SessionScraper:
    """Scrapes NeurIPS schedule to discover sessions by type."""

    def __init__(
        self,
        session: NeuripsSession,
        settings: PipelineSettings,
        browser_session: NeuripsBrowserSession | None = None,
    ):
        """Initialize session scraper.

        Args:
            session: Authenticated HTTP session.
            settings: Pipeline settings.
            browser_session: Optional browser session for JS-heavy pages.
        """
        self.session = session
        self.settings = settings
        self.browser_session = browser_session
        self._rate_limiter = RateLimiter(
            rate=max(1, self.settings.get_domain_limit("neurips")),
            burst=max(1, self.settings.get_domain_limit("neurips")),
        )

    async def discover_sessions(
        self,
        session_types: list[SessionType] | None = None,
    ) -> list[Session]:
        """Discover all sessions matching the target types.

        Args:
            session_types: Types to discover. Defaults to TARGET_SESSION_TYPES.

        Returns:
            List of discovered sessions with chosen=None.
        """
        target_types = session_types or TARGET_SESSION_TYPES
        sessions_by_url: dict[str, Session] = {}

        # First, scrape the main schedule page
        schedule_sessions = await self._scrape_schedule_page()
        for sess in schedule_sessions:
            if sess.url:
                sessions_by_url[sess.url] = sess

        # Then scrape specific event type pages - these have more accurate type info
        # and include abstracts from event cards
        # More specific types should not be overridden by less specific ones
        more_specific_types = {
            # expo_workshop is more specific than workshop
            SessionType.EXPO_WORKSHOP: {SessionType.WORKSHOP},
            # expo_talk_panel is more specific than expo
            SessionType.EXPO_TALK_PANEL: {SessionType.EXPO},
            # exhibitor_spot_talk is more specific than expo
            SessionType.EXHIBITOR_SPOT_TALK: {SessionType.EXPO},
        }

        for session_type in target_types:
            type_sessions = await self._scrape_event_type_page(session_type)
            for sess in type_sessions:
                if sess.url:
                    # Event type pages have authoritative type classification
                    # Update or add the session
                    if sess.url in sessions_by_url:
                        existing = sessions_by_url[sess.url]
                        # Don't override more specific types with less specific ones
                        should_update_type = True
                        for specific_type, generic_types in more_specific_types.items():
                            if existing.session_type == specific_type and sess.session_type in generic_types:
                                # Keep the more specific type
                                should_update_type = False
                                break

                        if should_update_type:
                            if existing.session_type == SessionType.OTHER or sess.session_type in target_types:
                                existing.session_type = sess.session_type
                        # Also update description/location/abstract if we got better data
                        if sess.description and not existing.description:
                            existing.description = sess.description
                        if sess.location and not existing.location:
                            existing.location = sess.location
                    else:
                        sessions_by_url[sess.url] = sess

            # Rate limiting
            await asyncio.sleep(self.settings.request_delay_seconds)

        # Convert back to list and filter to target types
        sessions = list(sessions_by_url.values())
        filtered = [s for s in sessions if s.session_type in target_types]

        # Count how many abstracts we got
        abstracts_found = sum(1 for s in filtered if s.description)
        logger.info(f"Discovered {len(filtered)} sessions of target types ({abstracts_found} with abstracts)")
        return filtered

    async def _scrape_schedule_page(self) -> list[Session]:
        """Scrape the main schedule page for sessions."""
        sessions = []
        sessions_by_url: dict[str, Session] = {}
        base = self.settings.virtual_base_url
        schedule_url = f"{base}/schedule"

        try:
            await self._rate_limiter.acquire()
            if self.browser_session:
                html = await self.browser_session.get_rendered_html(schedule_url)
            else:
                response = await self.session.get_full_url(schedule_url)
                html = response.text

            soup = BeautifulSoup(html, "lxml")

            # Parse schedule page events using CSS class patterns
            # The schedule page uses divs with class="eventsession" and type-specific classes
            for session_type, css_class in SESSION_CSS_CLASS_PATTERNS.items():
                type_events = soup.find_all(class_=re.compile(css_class, re.I))
                for elem in type_events:
                    sess = self._parse_schedule_event(elem, session_type)
                    if sess and sess.url:
                        # Deduplicate by URL
                        if sess.url not in sessions_by_url:
                            sessions_by_url[sess.url] = sess

            # Also look for session/event containers using the general parser
            session_elements = soup.find_all(
                class_=re.compile(r"session|event|calendar-item", re.I)
            )

            for elem in session_elements:
                sess = self._parse_session_element(elem)
                if sess and sess.url and sess.url not in sessions_by_url:
                    sessions_by_url[sess.url] = sess

            # Also look for links to event pages
            for link in soup.find_all("a", href=True):
                href = link["href"]
                text = link.get_text(strip=True)

                # Check if this looks like a session link
                if self._is_session_link(href, text):
                    sess = self._create_session_from_link(href, text)
                    if sess and sess.url and sess.url not in sessions_by_url:
                        sessions_by_url[sess.url] = sess

            sessions = list(sessions_by_url.values())
            logger.info(f"Found {len(sessions)} sessions on schedule page")

        except Exception as e:
            logger.error(f"Error scraping schedule page: {e}")

        return sessions

    def _parse_schedule_event(
        self,
        elem,
        session_type: SessionType,
    ) -> Session | None:
        """Parse an event from the NeurIPS schedule grid.

        The schedule page uses a specific format:
        <div class="eventsession pad expo-talk-panel room-...">
          <div class="hdr-style">Expo Talk Panel:</div>
          <div class="title-style"><a href="/virtual/2025/128659">Title</a></div>
          <div class="centered"><span class="end-time">(ends 9:30 AM)</span></div>
        </div>
        """
        try:
            # Find title - look for title-style class (NeurIPS schedule format)
            title_elem = elem.find(class_="title-style")
            if not title_elem:
                # Fall back to other title patterns
                title_elem = (
                    elem.find(class_="event-title") or
                    elem.find(class_=re.compile(r"title", re.I)) or
                    elem.find(["h2", "h3", "h4", "h5"])
                )

            if not title_elem:
                return None

            # Get title text
            title = title_elem.get_text(strip=True)
            if not title or len(title) < 3:
                return None

            # Find link
            link_elem = title_elem.find("a", href=True) or elem.find("a", href=True)
            if not link_elem or not link_elem.get("href"):
                return None

            url = self._normalize_url(link_elem["href"])
            if not url:
                return None

            # Extract ID from URL
            session_id = self._extract_id(url, title)

            # Find time info
            time_elem = elem.find(class_="end-time") or elem.find(class_=re.compile(r"time", re.I))
            time_str = time_elem.get_text(strip=True) if time_elem else None

            # Find room/location from class names
            location = None
            classes = elem.get("class", [])
            for cls in classes:
                if cls.startswith("room-"):
                    location = cls.replace("room-", "").replace("-", " ").title()
                    break

            return Session(
                id=session_id,
                title=title,
                session_type=session_type,
                url=url,
                time=time_str,
                location=location,
                chosen=None,
                discovered_at=datetime.utcnow(),
            )

        except Exception as e:
            logger.debug(f"Error parsing schedule event: {e}")
            return None

    async def _scrape_event_type_page(self, session_type: SessionType) -> list[Session]:
        """Scrape a specific event type page.

        Args:
            session_type: Type of session to scrape.

        Returns:
            List of sessions of the given type.
        """
        sessions = []
        base = self.settings.virtual_base_url

        # Construct possible URLs for this session type
        type_urls = self._get_event_type_urls(session_type)

        for url in type_urls:
            try:
                await self._rate_limiter.acquire()
                if self.browser_session:
                    html = await self.browser_session.get_rendered_html(url)
                else:
                    response = await self.session.get_full_url(url)
                    if response.status_code == 404:
                        logger.debug(f"URL not found: {url}")
                        continue
                    html = response.text

                soup = BeautifulSoup(html, "lxml")

                # Look for individual session/event items
                items = soup.find_all(
                    class_=re.compile(r"event|session|item|card", re.I)
                )

                page_sessions = []
                for elem in items:
                    sess = self._parse_session_element(elem, default_type=session_type)
                    if sess:
                        page_sessions.append(sess)

                # Also parse links
                for link in soup.find_all("a", href=True):
                    href = link["href"]
                    text = link.get_text(strip=True)
                    if len(text) > 10 and self._is_session_link(href, text):
                        sess = self._create_session_from_link(href, text, session_type)
                        if sess:
                            page_sessions.append(sess)

                if page_sessions:
                    logger.info(f"Found {len(page_sessions)} {session_type.value} sessions from {url}")
                    sessions.extend(page_sessions)

                await asyncio.sleep(0.5)  # Rate limit between pages

            except Exception as e:
                # Log at INFO level so we can see what's failing
                logger.info(f"Failed to scrape {url}: {type(e).__name__}: {e}")

        return sessions

    def _get_event_type_urls(self, session_type: SessionType) -> list[str]:
        """Get possible URLs for an event type."""
        base = self.settings.virtual_base_url
        urls = []

        type_name = session_type.value.replace("_", "-")
        # Primary events URL
        urls.append(f"{base}/events/{type_name}")

        # Only add a small number of known alternates; avoid broad plurals that 404/500
        if session_type == SessionType.ORAL:
            urls.append(f"{base}/oral")
        elif session_type == SessionType.TUTORIAL:
            urls.append(f"{base}/tutorial")
        elif session_type == SessionType.WORKSHOP:
            urls.append(f"{base}/workshop")
        elif session_type == SessionType.COMPETITION:
            urls.append(f"{base}/competition")

        return urls

    def _parse_session_element(
        self,
        elem,
        default_type: SessionType = SessionType.OTHER,
    ) -> Session | None:
        """Parse a session from an HTML element."""
        try:
            # Skip navigation and non-content elements
            elem_classes = " ".join(elem.get("class", []))
            if any(skip in elem_classes.lower() for skip in [
                "nav-", "dropdown", "navbar", "menu", "sidebar", "footer", "header"
            ]):
                return None

            # Find title - prefer event-title class specifically for NeurIPS
            title_elem = (
                elem.find(class_="event-title") or
                elem.find(class_=re.compile(r"title", re.I)) or
                elem.find(["h2", "h3", "h4", "h5"])
            )
            if not title_elem:
                return None

            title = title_elem.get_text(strip=True)
            if not title or len(title) < 5:
                return None

            # Skip navigation-like titles (common nav items on NeurIPS site)
            nav_titles = {
                "menu", "navigation", "search", "home", "back", "next", "previous",
                "mexico city", "invited talks", "competitions", "datasets & benchmarks",
                "journal track", "creative ai track", "creative ai", "affinity events",
                "sponsors", "organizers", "schedule", "papers", "workshops", "tutorials",
                "posters", "socials", "registration", "venue",
            }
            if title.lower().strip() in nav_titles:
                return None

            # Find link
            link_elem = title_elem.find("a") or elem.find("a", href=True)
            url = ""
            if link_elem and link_elem.get("href"):
                url = self._normalize_url(link_elem["href"])

            if not url:
                return None

            # Skip non-event URLs (must be a specific event page with numeric ID)
            if not re.search(r"/\d+/?$", url) and not re.search(r"/(tutorial|workshop|oral|keynote|expo|competition)", url.lower()):
                return None

            # Generate ID from URL or title
            session_id = self._extract_id(url, title)

            # Classify session type - prefer default_type for event-type pages
            # Only override if URL strongly indicates a different type
            classified_type = self._classify_session_type(title, url)
            if default_type != SessionType.OTHER:
                # Trust the page we're on unless URL clearly contradicts
                session_type = default_type
            else:
                session_type = classified_type or default_type

            # Find date/time
            date_elem = elem.find(class_=re.compile(r"date|time", re.I))
            date_str = date_elem.get_text(strip=True) if date_elem else None
            date_val, time_val = self._split_date_time(date_str) if date_str else (None, None)

            # Find location
            loc_elem = elem.find(class_=re.compile(r"location|room|venue", re.I))
            location = loc_elem.get_text(strip=True) if loc_elem else None

            # Find description/abstract - look for event-abstract specifically first (NeurIPS pattern)
            description = None
            desc_elem = (
                elem.find(class_="event-abstract") or
                elem.find(class_=re.compile(r"abstract|description|summary", re.I))
            )
            if desc_elem:
                description = desc_elem.get_text(strip=True)
                # Clean up whitespace and truncate to 2000 chars
                description = " ".join(description.split())
                if len(description) > 2000:
                    description = description[:1997] + "..."

            return Session(
                id=session_id,
                title=title,
                session_type=session_type,
                url=url,
                date=date_val or date_str,
                time=time_val,
                location=location,
                description=description,
                chosen=None,
                discovered_at=datetime.utcnow(),
            )

        except Exception as e:
            logger.debug(f"Error parsing session element: {e}")
            return None

    def _create_session_from_link(
        self,
        href: str,
        text: str,
        default_type: SessionType = SessionType.OTHER,
    ) -> Session | None:
        """Create a session from a link."""
        try:
            # Skip navigation-like titles
            nav_titles = {
                "menu", "navigation", "search", "home", "back", "next", "previous",
                "mexico city", "invited talks", "competitions", "datasets & benchmarks",
                "journal track", "creative ai track", "creative ai", "affinity events",
                "sponsors", "organizers", "schedule", "papers", "workshops", "tutorials",
                "posters", "socials", "registration", "venue",
            }
            if text.lower().strip() in nav_titles:
                return None

            url = self._normalize_url(href)
            if not url:
                return None

            # Skip non-event URLs (must be a specific event page with numeric ID)
            if not re.search(r"/\d+/?$", url) and not re.search(r"/(tutorial|workshop|oral|keynote|expo|competition)", url.lower()):
                return None

            session_id = self._extract_id(url, text)

            # Prefer default_type for event-type pages
            if default_type != SessionType.OTHER:
                session_type = default_type
            else:
                session_type = self._classify_session_type(text, url) or default_type

            return Session(
                id=session_id,
                title=text,
                session_type=session_type,
                url=url,
                time=None,
                chosen=None,
                discovered_at=datetime.utcnow(),
            )

        except Exception as e:
            logger.debug(f"Error creating session from link: {e}")
            return None

    def _normalize_url(self, href: str) -> str:
        """Normalize a URL to full form."""
        if not href:
            return ""

        # Skip timezone redirect URLs and other non-content URLs
        if "set_timezone" in href or "nextp=" in href:
            return ""

        if href.startswith("http"):
            return href
        elif href.startswith("/"):
            return f"{self.settings.neurips_base_url}{href}"
        else:
            return f"{self.settings.virtual_base_url}/{href}"

    def _extract_id(self, url: str, title: str) -> str:
        """Extract or generate a session ID."""
        # Try to extract numeric ID from URL
        id_match = re.search(r"/(\d+)/?$", url)
        if id_match:
            return id_match.group(1)

        # Try to extract from URL path
        path_match = re.search(r"/([a-z0-9_-]+)/?$", url.lower())
        if path_match:
            return path_match.group(1)

        # Generate from title
        return re.sub(r"[^a-zA-Z0-9]", "_", title.lower())[:50]

    def _is_session_link(self, href: str, text: str) -> bool:
        """Check if a link points to a session page."""
        if not href or not text:
            return False

        href_lower = href.lower()
        text_lower = text.lower()

        # Check URL patterns
        session_keywords = [
            "tutorial", "workshop", "oral", "keynote", "expo",
            "invited", "poster", "spotlight", "session", "event"
        ]

        for keyword in session_keywords:
            if keyword in href_lower:
                return True

        # Check title patterns
        for keyword in session_keywords:
            if keyword in text_lower:
                return True

        return False

    def _classify_session_type(self, title: str, url: str) -> SessionType | None:
        """Classify session type from title and URL."""
        title_lower = title.lower()
        url_lower = url.lower()

        # Check URL patterns first (more reliable)
        for session_type, patterns in SESSION_URL_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, url_lower):
                    return session_type

        # Check title patterns
        for session_type, patterns in SESSION_TITLE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, title_lower):
                    return session_type

        return None

    def _split_date_time(self, date_str: str) -> tuple[str | None, str | None]:
        """Attempt to split a combined date/time string into components."""
        if not date_str:
            return None, None

        # Simple pattern: date part followed by time
        time_match = re.search(r"(\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)?)", date_str)
        time_part = time_match.group(1) if time_match else None

        if time_match:
            date_part = date_str.replace(time_match.group(0), "").strip(" ,;-")
        else:
            date_part = date_str.strip()

        return (date_part or None), time_part

    async def save_sessions_json(
        self,
        sessions: list[Session],
        output_path: Path,
    ) -> None:
        """Save sessions to JSON file for manual filtering.

        Args:
            sessions: Sessions to save.
            output_path: Output file path.
        """
        data = {
            "discovered_at": datetime.utcnow().isoformat(),
            "total_sessions": len(sessions),
            "session_types": list(set(s.session_type.value for s in sessions)),
            "instructions": (
                "Set 'chosen' to 'Y' for sessions you want to process, "
                "'N' for sessions to skip. Leave as null to decide later."
            ),
            "sessions": [
                {
                    "id": s.id,
                    "title": s.title,
                    "session_type": s.session_type.value,
                    "url": s.url,
                    "date": s.date,
                    "time": s.time,
                    "location": s.location,
                    "abstract": s.description,  # Abstract fetched from session page
                    "presentation_count": s.presentation_count,
                    "chosen": s.chosen,
                }
                for s in sessions
            ],
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(sessions)} sessions to {output_path}")

    async def load_chosen_sessions(self, input_path: Path) -> list[Session]:
        """Load sessions from JSON and return only those with chosen='Y'.

        Args:
            input_path: Path to sessions JSON file.

        Returns:
            List of sessions with chosen='Y'.
        """
        with open(input_path) as f:
            data = json.load(f)

        sessions = []
        for s in data.get("sessions", []):
            if s.get("chosen") == "Y":
                # Handle both "abstract" (new) and "description" (old) keys
                description = s.get("abstract") or s.get("description")
                sessions.append(
                    Session(
                        id=s["id"],
                        title=s["title"],
                        session_type=SessionType(s["session_type"]),
                        url=s["url"],
                        date=s.get("date"),
                        time=s.get("time"),
                        location=s.get("location"),
                        description=description,
                        presentation_count=s.get("presentation_count"),
                        chosen="Y",
                    )
                )

        logger.info(f"Loaded {len(sessions)} chosen sessions from {input_path}")
        return sessions


async def discover_and_save_sessions(
    settings: PipelineSettings,
    output_path: Path,
    session_types: list[SessionType] | None = None,
) -> list[Session]:
    """Convenience function to discover sessions and save to JSON.

    Args:
        settings: Pipeline settings.
        output_path: Output JSON file path.
        session_types: Types to discover. Defaults to TARGET_SESSION_TYPES.

    Returns:
        List of discovered sessions.
    """
    from conference_pipeline.auth import NeuripsSession

    # Create session
    async with NeuripsSession.create(settings) as http_session:
        scraper = SessionScraper(http_session, settings)
        sessions = await scraper.discover_sessions(session_types)
        await scraper.save_sessions_json(sessions, output_path)
        return sessions
