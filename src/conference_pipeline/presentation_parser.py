"""Parser for individual presentation pages to extract SlidesLive info."""

import json
import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from conference_pipeline.auth import NeuripsBrowserSession, NeuripsSession
from conference_pipeline.config import PipelineSettings
from conference_pipeline.models import Presentation
from conference_pipeline.provider_receipts import (
    anthropic_messages_create,
    openai_chat_completions_create_async,
)
from utils.logging import get_logger

logger = get_logger(__name__)


class PresentationPageParser:
    """Parses individual presentation pages to find SlidesLive embeds."""

    def __init__(
        self,
        session: NeuripsSession,
        settings: PipelineSettings,
        browser_session: NeuripsBrowserSession | None = None,
    ):
        """Initialize parser.

        Args:
            session: Authenticated HTTP session.
            settings: Pipeline settings.
            browser_session: Optional browser session for JS-heavy pages.
        """
        self.session = session
        self.settings = settings
        self.browser_session = browser_session

    async def enrich_presentation(self, pres: Presentation) -> Presentation:
        """Enrich a presentation with SlidesLive information.

        Args:
            pres: Presentation with basic info.

        Returns:
            Presentation with SlidesLive info populated.
        """
        if not pres.neurips_url:
            logger.warning(f"No URL for presentation {pres.id}, skipping enrichment")
            return pres

        try:
            # Use browser for JS-rendered content if available
            if self.browser_session:
                html = await self.browser_session.get_rendered_html(
                    pres.neurips_url,
                    wait_selector="body",  # Wait for body (may not have slideslive on oral pages)
                    wait_time=3.0,
                )
                if self.browser_session.last_url:
                    pres.neurips_url = self.browser_session.last_url
            else:
                response = await self.session.get_full_url(pres.neurips_url)
                html = response.text

            self._update_canonical_url(pres, html)
            pres = self._extract_from_html(pres, html)

            llm_assessment = None
            if not pres.slideslive_id and self.browser_session and self.settings.use_llm_video_detection:
                try:
                    llm_assessment = await self._llm_assess_video_presence(
                        page_url=pres.neurips_url,
                        html=html,
                    )
                    if llm_assessment:
                        logger.info(
                            f"LLM video assessment for {pres.id}: "
                            f"has_video={llm_assessment.get('has_video')} "
                            f"conf={llm_assessment.get('confidence')} "
                            f"follow={llm_assessment.get('recommended_follow_url')}"
                        )
                except Exception as e:
                    logger.debug(f"LLM video assessment failed for {pres.id}: {e}")

            # Some pages only render the SlidesLive embed after a reload; try once.
            # If LLM is enabled and believes there's no video, avoid the long wait.
            # Still do a short force-reload for oral pages because NeurIPS sometimes
            # requires a refresh to populate event links/embeds.
            is_oral_page = "/oral/" in pres.neurips_url.lower()
            reload_wait_time: float | None = 8.0
            if llm_assessment and llm_assessment.get("has_video") is False:
                reload_wait_time = 2.5 if is_oral_page else None

            if not pres.slideslive_id and self.browser_session and reload_wait_time is not None:
                html = await self.browser_session.get_rendered_html(
                    pres.neurips_url,
                    wait_selector=None,
                    wait_time=reload_wait_time,
                    force_reload=True,
                )
                if self.browser_session.last_url:
                    pres.neurips_url = self.browser_session.last_url
                self._update_canonical_url(pres, html)
                pres = self._extract_from_html(pres, html)

            # Some oral pages link to a dedicated presentation page that holds the embed.
            # Follow that link once if we still don't have a SlidesLive ID.
            if not pres.slideslive_id:
                soup = BeautifulSoup(html, "lxml")
                follow_urls: list[str] = []
                if llm_assessment and llm_assessment.get("recommended_follow_url"):
                    follow_urls.append(str(llm_assessment["recommended_follow_url"]))

                ranked = self._rank_presentation_links(soup, pres)
                follow_urls.extend([url for url, _score in ranked[:5]])

                # Preserve order while de-duplicating.
                seen: set[str] = set()
                follow_urls = [u for u in follow_urls if u and (u not in seen and not seen.add(u))]

                for follow_url in follow_urls:
                    if pres.slideslive_id:
                        break
                    pres = await self._follow_and_extract(pres, follow_url)

            # Log result
            if pres.slideslive_id:
                logger.info(f"Found SlidesLive ID {pres.slideslive_id} for {pres.id}")
                pres.has_video = True
            else:
                logger.debug(f"No SlidesLive embed found for {pres.id}")

            return pres

        except Exception as e:
            logger.error(f"Error enriching presentation {pres.id}: {e}")
            return pres

    async def _follow_and_extract(self, pres: Presentation, follow_url: str) -> Presentation:
        """Follow a nested event link and try to extract SlidesLive info from it."""
        logger.info(f"Following nested presentation link for {pres.id}: {follow_url}")
        pres.neurips_url = follow_url

        try:
            if self.browser_session:
                child_html = await self.browser_session.get_rendered_html(
                    follow_url,
                    wait_selector=None,
                    wait_time=8.0,
                )
                if self.browser_session.last_url:
                    pres.neurips_url = self.browser_session.last_url
            else:
                resp = await self.session.get_full_url(follow_url)
                child_html = resp.text

            self._update_canonical_url(pres, child_html)
            pres = self._extract_from_html(pres, child_html)
            logger.info(f"After following link, slideslive_id = {pres.slideslive_id}")

            if pres.slideslive_id:
                return pres

            # Some content is only available on location-specific pages under /loc/<city>/...
            loc_urls = self._find_location_variant_urls(child_html, follow_url)
            for loc_url in loc_urls:
                if pres.slideslive_id:
                    break
                logger.info(f"Trying location-specific URL for {pres.id}: {loc_url}")
                pres.neurips_url = loc_url
                if self.browser_session:
                    loc_html = await self.browser_session.get_rendered_html(
                        loc_url,
                        wait_selector=None,
                        wait_time=8.0,
                        force_reload=True,
                    )
                    if self.browser_session.last_url:
                        pres.neurips_url = self.browser_session.last_url
                else:
                    resp = await self.session.get_full_url(loc_url)
                    loc_html = resp.text

                if loc_html:
                    self._update_canonical_url(pres, loc_html)
                    pres = self._extract_from_html(pres, loc_html)

            if pres.slideslive_id:
                return pres

            # Reload once more on the nested page to trigger lazy embed if still missing.
            if self.browser_session:
                child_html = await self.browser_session.get_rendered_html(
                    pres.neurips_url,
                    wait_selector=None,
                    wait_time=12.0,
                    force_reload=True,
                )
                if self.browser_session.last_url:
                    pres.neurips_url = self.browser_session.last_url
                self._update_canonical_url(pres, child_html)
                pres = self._extract_from_html(pres, child_html)

            return pres
        except Exception as e:  # pragma: no cover - best-effort fallback
            logger.warning(f"Failed following presentation link for {pres.id}: {e}")
            try:
                resp = await self.session.get_full_url(follow_url)
                self._update_canonical_url(pres, resp.text)
                pres = self._extract_from_html(pres, resp.text)
                logger.info(f"HTTP fallback found slideslive_id = {pres.slideslive_id}")
            except Exception as inner_e:  # pragma: no cover
                logger.debug(f"HTTP fallback also failed for {pres.id}: {inner_e}")
            return pres

    async def _llm_assess_video_presence(self, page_url: str, html: str) -> dict[str, object] | None:
        """Use an LLM to quickly assess whether a page likely contains a SlidesLive embed.

        This is best-effort and intentionally uses a small, sanitized snippet of the page.
        """
        provider = self.settings.video_detection_provider

        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        title = soup.title.get_text(strip=True) if soup.title else ""
        text = soup.get_text("\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Fast deterministic signal before calling an LLM
        lowered = text.lower()
        if "no slideslive embed found" in lowered:
            return {"has_video": False, "confidence": 0.95, "recommended_follow_url": None, "reason": "Page explicitly says no embed found."}

        # Extract candidate links (paths only, no query params) to avoid leaking tokens.
        candidates: list[str] = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            href = href.split("#", 1)[0]
            href = href.split("?", 1)[0]
            if re.search(r"/(session|poster|oral|presentation)/\d+", href.lower()):
                resolved = self._resolve_url(href)
                if resolved not in candidates:
                    candidates.append(resolved)
            if len(candidates) >= 30:
                break

        def redact(s: str) -> str:
            s = re.sub(r"(api_key|token|sessionid|csrftoken)=[^&\\s]+", r"\\1=REDACTED", s, flags=re.I)
            s = re.sub(r"[A-Za-z0-9+/]{60,}={0,2}", "REDACTED_B64", s)
            return s

        prompt = {
            "page_url": page_url,
            "title": title[:200],
            "text_excerpt": redact(text[:8000]),
            "candidate_event_links": candidates,
        }

        system = (
            "You are a helper that detects whether a NeurIPS virtual page contains a SlidesLive video embed. "
            "Return STRICT JSON only with keys: has_video (bool), confidence (0..1), "
            "recommended_follow_url (string|null), reason (string). "
            "Only recommend a follow URL if it is exactly one of candidate_event_links."
        )
        user = json.dumps(prompt, ensure_ascii=False)

        if provider == "openai":
            if not self.settings.openai_api_key:
                return None
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=self.settings.openai_api_key)
            resp = await openai_chat_completions_create_async(client, caller=__file__,
                model=self.settings.video_detection_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                max_tokens=250,
            )
            content = (resp.choices[0].message.content or "").strip()
        elif provider == "anthropic":
            if not self.settings.anthropic_api_key:
                return None
            import anthropic

            client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)

            def _call() -> str:
                msg = anthropic_messages_create(client, caller=__file__,
                    model=self.settings.video_detection_model,
                    max_tokens=250,
                    messages=[{"role": "user", "content": system + "\n\n" + user}],
                )
                return msg.content[0].text

            import asyncio as _asyncio
            content = (await _asyncio.to_thread(_call)).strip()
        else:
            return None

        # Parse JSON from model output
        try:
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                data = json.loads(content[start : end + 1])
            else:
                data = json.loads(content)
        except Exception:
            return None

        # Validate recommended URL
        rec = data.get("recommended_follow_url")
        if rec is not None and rec not in candidates:
            data["recommended_follow_url"] = None

        return data

    def _extract_from_html(self, pres: Presentation, html: str) -> Presentation:
        """Apply multiple heuristics to pull SlidesLive info from HTML."""
        soup = BeautifulSoup(html, "lxml")

        # Strategy 1: Look for div with slideslive ID
        slideslive_div = soup.find(id=re.compile(r"slideslive", re.I))
        if slideslive_div and not pres.slideslive_id:
            pres = self._extract_from_div(pres, slideslive_div)

        # Strategy 2: Look for iframe with SlidesLive source
        if not pres.slideslive_id:
            slideslive_iframe = soup.find("iframe", src=re.compile(r"slideslive", re.I))
            if slideslive_iframe:
                pres = self._extract_from_iframe(pres, slideslive_iframe)

        # Strategy 3: Look for script tags loading SlidesLive
        if not pres.slideslive_id:
            for script in soup.find_all("script", src=True):
                if "slideslive" in script["src"].lower():
                    pres = self._extract_from_script(pres, script)
                    break

        # Strategy 4: Search for SlidesLive IDs in any attribute or text
        if not pres.slideslive_id:
            pres = self._search_for_slideslive_id(pres, html)

        # Strategy 5: Look for data attributes with video/presentation info
        if not pres.slideslive_id:
            pres = self._extract_from_data_attrs(pres, soup)

        return pres

    def _update_canonical_url(self, pres: Presentation, html: str) -> None:
        """Update `pres.neurips_url` using the page canonical URL when available.

        NeurIPS frequently redirects `/virtual/{year}/poster/{id}` to a
        location-specific URL like `/virtual/{year}/loc/{city}/poster/{id}`.
        Some embeds only show up on the canonical location URL, so we persist it.
        """
        try:
            soup = BeautifulSoup(html, "lxml")
            canonical = soup.find("link", rel="canonical")
            href = canonical.get("href") if canonical else None
            if href and isinstance(href, str) and href.startswith("http"):
                pres.neurips_url = href
        except Exception:
            return

    def _find_location_variant_urls(self, html: str, base_url: str) -> list[str]:
        """Find location-specific variant URLs for the same event (e.g. /loc/<city>/poster/<id>).

        Some NeurIPS pages render embeds only on one location variant; try all candidates.
        """
        try:
            soup = BeautifulSoup(html, "lxml")
            base_lower = base_url.lower()
            m = re.search(r"/(poster|oral|presentation|session)/(\d+)", base_lower)
            if not m:
                return []
            kind, event_id = m.group(1), m.group(2)
            pattern = re.compile(rf"/virtual/\\d+/loc/[^/]+/{kind}/{event_id}\\b", re.I)

            found: list[str] = []
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if pattern.search(href):
                    resolved = self._resolve_url(href)
                    if resolved not in found:
                        found.append(resolved)

            # If the page doesn't include explicit /loc/ links, try common locations.
            if not found:
                for city in ["san-diego", "mexico-city"]:
                    candidate = f"{self.settings.neurips_base_url}/virtual/{self.settings.year}/loc/{city}/{kind}/{event_id}"
                    found.append(candidate)

            return found
        except Exception:
            return []

    def _find_presentation_link(self, soup: BeautifulSoup, pres: Presentation) -> str | None:
        """Find a deeper presentation link (e.g., /virtual/2025/presentation/... or /virtual/2025/poster/...) to follow.

        Oral pages often link to poster pages that contain the actual SlidesLive embed.
        This method extracts that link, handling login redirects and timezone redirects.
        """
        ranked = self._rank_presentation_links(soup, pres)
        return ranked[0][0] if ranked else None

    def _rank_presentation_links(self, soup: BeautifulSoup, pres: Presentation) -> list[tuple[str, float]]:
        """Rank event links on the page by likelihood of containing a SlidesLive embed."""
        anchors = soup.find_all("a", href=True)

        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "").lower()).strip()

        def event_info(url: str) -> tuple[str | None, str | None]:
            m = re.search(r"/(poster|oral|presentation|session)/(\d+)", url.lower())
            if not m:
                return None, None
            return m.group(1), m.group(2)

        pres_title_norm = norm(pres.title)
        pres_url_norm = norm(pres.neurips_url)

        # Collect candidates, including redirect targets.
        candidates: list[dict[str, object]] = []
        for a in anchors:
            href = (a.get("href") or "").strip()
            if not href:
                continue

            href_lower = href.lower()
            text = a.get_text(strip=True)

            # Skip calendar and events pages - these don't have video embeds
            if "/calendar" in href_lower or "/events/" in href_lower:
                continue

            resolved = self._resolve_url(href)
            kind, event_id = event_info(resolved)
            if kind is None:
                continue

            # Track whether this came from a redirect wrapper URL
            from_redirect = ("set_timezone" in href_lower or "login" in href_lower) and "nextp=" in href_lower
            candidates.append(
                {
                    "href": href,
                    "resolved": resolved,
                    "text": text,
                    "kind": kind,
                    "event_id": event_id,
                    "from_redirect": from_redirect,
                }
            )

        if not candidates:
            return []

        # Build set of event IDs mentioned by non-redirect links on the page.
        direct_event_ids = {c["event_id"] for c in candidates if not c["from_redirect"]}

        scored: list[tuple[str, float]] = []

        for c in candidates:
            resolved = str(c["resolved"])
            kind = str(c["kind"])
            event_id = str(c["event_id"]) if c["event_id"] is not None else ""
            text = norm(str(c["text"]))
            from_redirect = bool(c["from_redirect"])

            score = 0.0

            # Prefer pages most likely to contain the embed.
            # Empirically, NeurIPS sometimes hosts the SlidesLive embed on `/session/<id>` pages.
            if kind == "session":
                score += 120
            elif kind == "poster":
                score += 100
            elif kind == "presentation":
                score += 80
            elif kind == "oral":
                score += 60

            # Prefer IDs that match the current page's event id.
            if event_id and (f"/{event_id}" in pres_url_norm or event_id == pres.id):
                score += 50

            # Prefer anchor text that matches the title.
            if pres_title_norm and pres_title_norm in text:
                score += 30

            # Redirect targets are useful only if they correspond to an event link on the page.
            if from_redirect:
                if event_id and event_id in direct_event_ids:
                    score += 15
                else:
                    score -= 100

            scored.append((resolved, score))

        scored.sort(key=lambda item: item[1], reverse=True)

        if scored:
            top_url, top_score = scored[0]
            logger.debug(
                f"Chose nested link for {pres.id}: {top_url} "
                f"(score={top_score}, title_match={bool(pres_title_norm)})"
            )

        return scored

    def _resolve_url(self, href: str) -> str:
        """Resolve a URL, handling redirects and relative paths."""
        from urllib.parse import parse_qs, unquote, urlparse

        href_lower = href.lower()

        # Handle redirect URLs with nextp parameter
        if ("set_timezone" in href_lower or "login" in href_lower) and "nextp=" in href_lower:
            parsed = urlparse(href)
            qs = parse_qs(parsed.query)
            nextp = qs.get("nextp", [None])[0]
            if nextp:
                next_target = unquote(nextp).strip()
                if next_target.startswith("http"):
                    return next_target
                if next_target.startswith("/"):
                    return urljoin(self.settings.neurips_base_url, next_target)
                return urljoin(self.settings.neurips_base_url + "/", next_target)

        if href.startswith("http"):
            return href
        if href.startswith("/"):
            return urljoin(self.settings.neurips_base_url, href)
        return urljoin(self.settings.neurips_base_url + "/", href)

    def _extract_from_div(self, pres: Presentation, div) -> Presentation:
        """Extract SlidesLive info from a div element.

        Args:
            pres: Presentation to update.
            div: BeautifulSoup div element.

        Returns:
            Updated presentation.
        """
        try:
            div_id = div.get("id", "")

            # ID format: slideslive:38970234 or just 38970234
            id_match = re.search(r"(\d{7,})", div_id)
            if id_match:
                pres.slideslive_id = id_match.group(1)
                pres.slideslive_url = f"{self.settings.slideslive_base_url}/{pres.slideslive_id}"

            # Check for data attributes
            for attr in ["data-slideslive-id", "data-presentation-id", "data-video-id"]:
                if div.get(attr):
                    pres.slideslive_id = div[attr]
                    pres.slideslive_url = f"{self.settings.slideslive_base_url}/{pres.slideslive_id}"
                    break

            return pres

        except Exception as e:
            logger.debug(f"Error extracting from div: {e}")
            return pres

    def _extract_from_iframe(self, pres: Presentation, iframe) -> Presentation:
        """Extract SlidesLive info from an iframe element.

        Args:
            pres: Presentation to update.
            iframe: BeautifulSoup iframe element.

        Returns:
            Updated presentation.
        """
        try:
            src = iframe.get("src", "")

            # Extract ID from URL like https://slideslive.com/embed/presentation/38970234
            id_match = re.search(r"slideslive\.com/(?:embed/)?(?:presentation/)?(\d+)", src)
            if id_match:
                pres.slideslive_id = id_match.group(1)
                pres.slideslive_url = f"{self.settings.slideslive_base_url}/{pres.slideslive_id}"

            return pres

        except Exception as e:
            logger.debug(f"Error extracting from iframe: {e}")
            return pres

    def _extract_from_script(self, pres: Presentation, script) -> Presentation:
        """Extract SlidesLive info from a script tag.

        Args:
            pres: Presentation to update.
            script: BeautifulSoup script element.

        Returns:
            Updated presentation.
        """
        try:
            src = script.get("src", "")

            # URL pattern: https://slideslive.com/js_embed/presentations/38970234
            id_match = re.search(r"presentations?/(\d+)", src)
            if id_match:
                pres.slideslive_id = id_match.group(1)
                pres.slideslive_url = f"{self.settings.slideslive_base_url}/{pres.slideslive_id}"

            return pres

        except Exception as e:
            logger.debug(f"Error extracting from script: {e}")
            return pres

    def _search_for_slideslive_id(self, pres: Presentation, html: str) -> Presentation:
        """Search HTML for SlidesLive IDs using regex.

        Args:
            pres: Presentation to update.
            html: Raw HTML content.

        Returns:
            Updated presentation.
        """
        try:
            # Common patterns for SlidesLive IDs
            patterns = [
                r'slideslive["\'\s:=]+(\d{7,})',
                r"presentation[_-]?id['\"\s:=]+(\d{7,})",
                r"video[_-]?id['\"\s:=]+(\d{7,})",
                r"slideslive\.com/(?:embed/)?(?:presentation/)?(\d{7,})",
            ]

            for pattern in patterns:
                match = re.search(pattern, html, re.I)
                if match:
                    pres.slideslive_id = match.group(1)
                    pres.slideslive_url = f"{self.settings.slideslive_base_url}/{pres.slideslive_id}"
                    break

            return pres

        except Exception as e:
            logger.debug(f"Error searching for SlidesLive ID: {e}")
            return pres

    def _extract_from_data_attrs(self, pres: Presentation, soup: BeautifulSoup) -> Presentation:
        """Extract SlidesLive info from data attributes.

        Args:
            pres: Presentation to update.
            soup: BeautifulSoup parsed HTML.

        Returns:
            Updated presentation.
        """
        try:
            # Look for elements with data-* attributes containing video info
            data_attrs = [
                "data-slideslive-id",
                "data-presentation-id",
                "data-video-id",
                "data-embed-id",
            ]

            for attr in data_attrs:
                elem = soup.find(attrs={attr: True})
                if elem:
                    value = elem[attr]
                    # Validate it looks like a SlidesLive ID (7+ digits)
                    if re.match(r"^\d{7,}$", str(value)):
                        pres.slideslive_id = value
                        pres.slideslive_url = f"{self.settings.slideslive_base_url}/{value}"
                        break

            # Also check for has_slides indicators
            slides_indicators = soup.find_all(
                class_=re.compile(r"slides?|pdf|download", re.I)
            )
            if slides_indicators:
                for elem in slides_indicators:
                    href = elem.get("href", "")
                    if ".pdf" in href.lower() or "slides" in href.lower():
                        pres.has_slides = True
                        break

            return pres

        except Exception as e:
            logger.debug(f"Error extracting from data attributes: {e}")
            return pres

    async def check_slideslive_availability(self, pres: Presentation) -> Presentation:
        """Verify SlidesLive embed is accessible.

        Args:
            pres: Presentation with SlidesLive ID.

        Returns:
            Presentation with updated availability flags.
        """
        if not pres.slideslive_id:
            return pres

        try:
            # Check if SlidesLive page exists
            url = f"{self.settings.slideslive_base_url}/{pres.slideslive_id}"
            response = await self.session.get_full_url(url)

            if response.status_code == 200:
                html = response.text

                # Check for video player
                if "video" in html.lower() or "player" in html.lower():
                    pres.has_video = True

                # Check for slides
                if "slide" in html.lower() or "pdf" in html.lower():
                    pres.has_slides = True
            else:
                logger.warning(
                    f"SlidesLive page returned {response.status_code} for {pres.slideslive_id}"
                )
                pres.slideslive_id = None
                pres.slideslive_url = None

            return pres

        except Exception as e:
            logger.warning(f"Error checking SlidesLive availability for {pres.id}: {e}")
            return pres
