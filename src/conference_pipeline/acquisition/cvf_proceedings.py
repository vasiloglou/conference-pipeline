"""CVF proceedings: the registered workshop owner and the public transport for Stage 4.

CVF conferences host their workshops' OpenReview venues under a CVF parent
venue (``thecvf.com/<Venue>/<Year>/Workshop``) and their archival papers on
CVF Open Access. The public program-item lane
(:mod:`conference_pipeline.acquisition.program_item_acquisition`) routes every
workshop request to *one* registered owner per parent venue instead of a
second OpenReview client; this module supplies that owner
(:func:`workshop_owner`) and the credential-free public transport the lane
opens after lineage and capacity pass (:func:`public_url_opener_factory`).
The CVF Open Access workshop-paper downloader itself lives in
:mod:`conference_pipeline.acquisition.cvf_openaccess_workshops`.

Recommended for
---------------
``RECOMMENDED_FOR = ("CVPR",)``.

* **CVPR stage and quirk covered:** Stage 4 ``download-corpus`` r3, the "CVF
  workshop owner": CVPR workshop requests resolve through the registered
  OpenReview workshop pipeline under the CVF parent venue, run once per
  parent venue with no paper cap.
* **May also fit:** ICCV and WACV (same CVF parent-venue and Open Access
  conventions); the parent venue comes from the orchestrator.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

RECOMMENDED_FOR = ("CVPR",)

DEFAULT_PUBLIC_TIMEOUT_SECONDS = 30


def workshop_owner(workspace: Path) -> Callable[[str], Mapping[str, Any]]:
    """Return the registered workshop owner, not a second OpenReview client."""

    from conference_pipeline.acquisition import run_workshop_discovery_and_download

    def run(parent_venue: str) -> Mapping[str, Any]:
        return run_workshop_discovery_and_download(workspace, parent_venue=parent_venue, max_papers=0)

    return run


def public_url_opener_factory(
    user_agent: str, *, timeout: float = DEFAULT_PUBLIC_TIMEOUT_SECONDS
) -> Callable[[], Callable[[str], Any]]:
    """A factory for the Stage-4 public-only transport (no credentials or cookies)."""

    def factory() -> Callable[[str], Any]:
        from urllib.request import Request, urlopen

        def opener(url: str) -> Any:
            request = Request(url, headers={"User-Agent": user_agent})
            return urlopen(request, timeout=timeout)

        return opener

    return factory
