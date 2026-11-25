"""Local contract tests for the immutable CVPR r4 registration-media overlay."""

from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from conference_pipeline.acquisition.cvpr_registration_media import (
    FULL_R4_SCOPE,
    PINNED_R3_PREDECESSORS,
    R3_RUN_ID,
    RegistrationMediaScope,
    _media_candidates_from_page,
    build_candidate_plan,
    build_registration_media_overlay,
    cvpr_authenticated_event_opener_factory,
    cvpr_registered_virtual_opener_factory,
    extract_cvpr_youtube_nocookie_links,
    is_allowed_cvpr_virtual_event_url,
    preflight_registered_virtual_session,
    recover_registered_official_pages,
    resume_registration_media_overlay,
    transport_eligible_requests,
    validate_registration_media_overlay,
)
from conference_pipeline.acquisition.embedded_media import EMBEDDED_MEDIA_HOSTS
from conference_pipeline.conference_identity import VirtualSiteBoundary, parse_virtual_site_boundary


FIXTURE_BOUNDARY = parse_virtual_site_boundary("https://cvpr.thecvf.com/virtual/2026")

# Every existing r4 fixture is intentionally local and derives this exact
# boundary from its explicit virtual URL.  Keep the raw imports below so the
# portability regression can also prove the production signatures themselves
# have no implicit conference default.
_raw_build_candidate_plan = build_candidate_plan
_raw_build_registration_media_overlay = build_registration_media_overlay
_raw_cvpr_authenticated_event_opener_factory = cvpr_authenticated_event_opener_factory
_raw_cvpr_registered_virtual_opener_factory = cvpr_registered_virtual_opener_factory
_raw_extract_cvpr_youtube_nocookie_links = extract_cvpr_youtube_nocookie_links
_raw_is_allowed_cvpr_virtual_event_url = is_allowed_cvpr_virtual_event_url
_raw_recover_registered_official_pages = recover_registered_official_pages
_raw_resume_registration_media_overlay = resume_registration_media_overlay
_raw_transport_eligible_requests = transport_eligible_requests
_raw_validate_registration_media_overlay = validate_registration_media_overlay
_raw_media_candidates_from_page = _media_candidates_from_page


def _fixture_boundary_call(function: object, *args: object, **kwargs: object) -> object:
    assert callable(function)
    kwargs.setdefault("boundary", FIXTURE_BOUNDARY)
    return function(*args, **kwargs)


def build_candidate_plan(*args: object, **kwargs: object) -> dict[str, object]:
    return _fixture_boundary_call(_raw_build_candidate_plan, *args, **kwargs)  # type: ignore[return-value]


def build_registration_media_overlay(*args: object, **kwargs: object) -> dict[str, object]:
    return _fixture_boundary_call(_raw_build_registration_media_overlay, *args, **kwargs)  # type: ignore[return-value]


def cvpr_authenticated_event_opener_factory(*args: object, **kwargs: object) -> object:
    return _fixture_boundary_call(_raw_cvpr_authenticated_event_opener_factory, *args, **kwargs)


def cvpr_registered_virtual_opener_factory(*args: object, **kwargs: object) -> object:
    return _fixture_boundary_call(_raw_cvpr_registered_virtual_opener_factory, *args, **kwargs)


def extract_cvpr_youtube_nocookie_links(*args: object, **kwargs: object) -> list[str]:
    return _fixture_boundary_call(_raw_extract_cvpr_youtube_nocookie_links, *args, **kwargs)  # type: ignore[return-value]


def is_allowed_cvpr_virtual_event_url(*args: object, **kwargs: object) -> bool:
    return _fixture_boundary_call(_raw_is_allowed_cvpr_virtual_event_url, *args, **kwargs)  # type: ignore[return-value]


def recover_registered_official_pages(*args: object, **kwargs: object) -> dict[str, object]:
    return _fixture_boundary_call(_raw_recover_registered_official_pages, *args, **kwargs)  # type: ignore[return-value]


def resume_registration_media_overlay(*args: object, **kwargs: object) -> dict[str, object]:
    return _fixture_boundary_call(_raw_resume_registration_media_overlay, *args, **kwargs)  # type: ignore[return-value]


def transport_eligible_requests(*args: object, **kwargs: object) -> list[dict[str, object]]:
    return _fixture_boundary_call(_raw_transport_eligible_requests, *args, **kwargs)  # type: ignore[return-value]


def validate_registration_media_overlay(*args: object, **kwargs: object) -> None:
    _fixture_boundary_call(_raw_validate_registration_media_overlay, *args, **kwargs)


def _media_candidates_from_page(*args: object, **kwargs: object) -> tuple[dict[str, str], bool]:
    return _fixture_boundary_call(_raw_media_candidates_from_page, *args, **kwargs)  # type: ignore[return-value]


def _frozen_r3_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Create the complete frozen cardinality surface without production payloads."""

    requests: list[dict[str, object]] = []
    program_items: list[dict[str, object]] = []
    next_item = 0
    for request_index in range(FULL_R4_SCOPE.transport_eligible_request_count):
        item_ids: list[str] = []
        for _ in range(2):
            item_id = f"cvpr2026:program-item:{next_item:05d}"
            next_item += 1
            item_ids.append(item_id)
        requests.append(
            {
                "request_key": f"eligible-request-{request_index:05d}",
                "source_url": f"https://cvpr.thecvf.com/virtual/2026/poster/{request_index + 1}",
                "status": "policy_excluded_official_video",
                "linked_program_item_ids": item_ids,
                "source_evidence": {"source_record_id": f"source-{request_index:05d}"},
            }
        )
        for item_id in item_ids:
            program_items.append(
                {
                    "program_item_id": item_id,
                    "request_key": requests[-1]["request_key"],
                    "blocker": {"code": "policy_excluded_official_video"},
                    "source_record_id": f"source-{request_index:05d}",
                    "source_family_id": "virtual-paper-media-posture",
                    "selection_status": "included",
                }
            )

    # The exact frozen policy mapping has six request rows covering thirteen IDs.
    for boundary_index, item_count in enumerate((3, 2, 2, 2, 2, 2)):
        item_ids = []
        for _ in range(item_count):
            item_id = f"cvpr2026:program-item:{next_item:05d}"
            next_item += 1
            item_ids.append(item_id)
        requests.append(
            {
                "request_key": f"boundary-request-{boundary_index:02d}",
                "source_url": f"https://outside.example/virtual/2026/{boundary_index}",
                "status": "policy_excluded_official_video",
                "linked_program_item_ids": item_ids,
                "source_evidence": {"source_record_id": f"boundary-{boundary_index:02d}"},
            }
        )
        for item_id in item_ids:
            program_items.append(
                {
                    "program_item_id": item_id,
                    "request_key": requests[-1]["request_key"],
                    "blocker": {"code": "policy_excluded_official_video"},
                    "source_record_id": f"boundary-{boundary_index:02d}",
                    "source_family_id": "virtual-paper-media-posture",
                    "selection_status": "included",
                }
            )

    while len(program_items) < FULL_R4_SCOPE.total_program_item_count:
        item_id = f"cvpr2026:program-item:{next_item:05d}"
        next_item += 1
        program_items.append(
            {
                "program_item_id": item_id,
                "request_key": f"r3-request-{next_item:05d}",
                "blocker": {"code": "unreachable_http_error"},
                "source_record_id": f"r3-source-{next_item:05d}",
                "source_family_id": "program-schedule",
                "selection_status": "included",
            }
        )
    return requests, program_items


def _plan() -> tuple[dict[str, object], list[dict[str, object]]]:
    requests, program_items = _frozen_r3_rows()
    return (
        build_candidate_plan(
            r3_run_id=R3_RUN_ID,
            predecessors=PINNED_R3_PREDECESSORS,
            r3_request_results=requests,
            r3_program_item_results=program_items,
        ),
        program_items,
    )


def test_frozen_candidate_plan_preserves_exact_lineage_partition_and_full_scope() -> None:
    plan, program_items = _plan()

    assert plan["r3_run_id"] == R3_RUN_ID
    assert plan["predecessors"] == PINNED_R3_PREDECESSORS
    assert plan["counts"] == FULL_R4_SCOPE.as_dict()
    assert len(plan["request_rows"]) == 2_579
    assert len({item_id for row in plan["request_rows"] for item_id in row["linked_program_item_ids"]}) == 5_159
    assert len(transport_eligible_requests(plan)) == 2_573
    assert sum(len(row["linked_program_item_ids"]) for row in transport_eligible_requests(plan)) == 5_146

    boundary_rows = [row for row in plan["request_rows"] if not row["transport_eligible"]]
    assert len(boundary_rows) == 6
    assert sum(len(row["linked_program_item_ids"]) for row in boundary_rows) == 13
    assert all(row["final_disposition"] == "outside_authenticated_official_event_boundary" for row in boundary_rows)
    assert all("opener_input" not in row for row in boundary_rows)

    overlay = build_registration_media_overlay(
        candidate_plan=plan,
        r3_program_item_results=program_items,
    )
    validate_registration_media_overlay(overlay)
    assert len(overlay["program_item_accounting"]) == 10_515
    assert overlay["counts"] == FULL_R4_SCOPE.as_dict()
    assert sum(row["result_origin"] == "r4_registration_media" for row in overlay["program_item_accounting"]) == 5_159
    assert sum(row["result_origin"] == "r3_predecessor" for row in overlay["program_item_accounting"]) == 5_356


def test_candidate_plan_rejects_stale_lineage_and_wrong_r3_run() -> None:
    requests, program_items = _frozen_r3_rows()
    stale = copy.deepcopy(PINNED_R3_PREDECESSORS)
    stale[2]["file_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="pinned predecessor"):
        build_candidate_plan(
            r3_run_id=R3_RUN_ID,
            predecessors=stale,
            r3_request_results=requests,
            r3_program_item_results=program_items,
        )
    with pytest.raises(ValueError, match="pinned r3 run"):
        build_candidate_plan(
            r3_run_id="cvpr2026-live-production-r2",
            predecessors=PINNED_R3_PREDECESSORS,
            r3_request_results=requests,
            r3_program_item_results=program_items,
        )


@pytest.mark.parametrize(
    "url",
    (
        "https://cvpr.thecvf.com/virtual/2026/search",
        "https://cvpr.thecvf.com/virtual/2026/index.html",
        "https://cvpr.thecvf.com/virtual/2026/sponsor_list",
        "https://cvpr.thecvf.com/virtual/2026/eventlistwithbios/2026KeynotesPanels",
    ),
)
def test_r4_legacy_static_virtual_paths_are_exactly_preserved(url: str) -> None:
    assert is_allowed_cvpr_virtual_event_url(url)
    assert not is_allowed_cvpr_virtual_event_url(f"{url}/unexpected")


def test_registered_media_url_policy_requires_an_explicit_configured_boundary() -> None:
    """Pure r4 URL policy must not recreate a CVPR host/root/year default."""

    boundary_parameter = inspect.signature(_raw_is_allowed_cvpr_virtual_event_url).parameters["boundary"]
    assert boundary_parameter.default is inspect.Parameter.empty
    overlay_boundary = inspect.signature(_raw_validate_registration_media_overlay).parameters["boundary"]
    assert overlay_boundary.default is inspect.Parameter.empty

    exact = "https://cvpr.thecvf.com/virtual/2026/search"
    assert _raw_is_allowed_cvpr_virtual_event_url(exact, boundary=FIXTURE_BOUNDARY)
    assert not _raw_is_allowed_cvpr_virtual_event_url(f"{exact}/unexpected", boundary=FIXTURE_BOUNDARY)
    assert not _raw_is_allowed_cvpr_virtual_event_url(
        "https://outside.example/virtual/2026/search", boundary=FIXTURE_BOUNDARY
    )
    assert not _raw_is_allowed_cvpr_virtual_event_url(
        "https://cvpr.thecvf.com/virtual/2025/search", boundary=FIXTURE_BOUNDARY
    )
    assert not _raw_is_allowed_cvpr_virtual_event_url(
        "https://cvpr.thecvf.com/virtual/2026/search?next=1", boundary=FIXTURE_BOUNDARY
    )
    assert not _raw_is_allowed_cvpr_virtual_event_url(
        "https://cvpr.thecvf.com/virtual/2026/search#fragment", boundary=FIXTURE_BOUNDARY
    )
    assert not _raw_is_allowed_cvpr_virtual_event_url(
        "https://cvpr.thecvf.com/virtual/2026/../search", boundary=FIXTURE_BOUNDARY
    )
    assert not is_allowed_cvpr_virtual_event_url(
        exact, boundary=VirtualSiteBoundary(scheme="https", host="outside.example", root_segments=("virtual", "2026"))
    )


def test_candidate_plan_rejects_duplicate_requests_boundary_drift_and_incomplete_fanout() -> None:
    requests, program_items = _frozen_r3_rows()
    duplicate = copy.deepcopy(requests)
    duplicate.append(copy.deepcopy(duplicate[0]))
    with pytest.raises(ValueError, match="duplicate policy request key"):
        build_candidate_plan(
            r3_run_id=R3_RUN_ID,
            predecessors=PINNED_R3_PREDECESSORS,
            r3_request_results=duplicate,
            r3_program_item_results=program_items,
        )

    boundary_drift = copy.deepcopy(requests)
    boundary_drift[0]["source_url"] = "https://outside.example/not-an-event"
    with pytest.raises(ValueError, match="transport-eligible"):
        build_candidate_plan(
            r3_run_id=R3_RUN_ID,
            predecessors=PINNED_R3_PREDECESSORS,
            r3_request_results=boundary_drift,
            r3_program_item_results=program_items,
        )

    incomplete_fanout = copy.deepcopy(program_items)
    incomplete_fanout[0]["blocker"] = {"code": "no_admissible_artifact"}
    with pytest.raises(ValueError, match="candidate program-item mapping"):
        build_candidate_plan(
            r3_run_id=R3_RUN_ID,
            predecessors=PINNED_R3_PREDECESSORS,
            r3_request_results=requests,
            r3_program_item_results=incomplete_fanout,
        )


def test_overlay_rejects_missing_or_duplicate_full_accounting_rows() -> None:
    plan, program_items = _plan()
    overlay = build_registration_media_overlay(candidate_plan=plan, r3_program_item_results=program_items)

    missing = copy.deepcopy(overlay)
    missing["program_item_accounting"].pop()
    with pytest.raises(ValueError, match="full corpus"):
        validate_registration_media_overlay(missing)

    duplicate = copy.deepcopy(overlay)
    duplicate["program_item_accounting"].append(copy.deepcopy(duplicate["program_item_accounting"][0]))
    with pytest.raises(ValueError, match="duplicate program item"):
        validate_registration_media_overlay(duplicate)


def test_overlay_is_secret_safe_and_supports_smaller_error_fixtures_without_production_payloads() -> None:
    plan, program_items = _plan()
    overlay = build_registration_media_overlay(candidate_plan=plan, r3_program_item_results=program_items)
    serialized = json.dumps(overlay, sort_keys=True).lower()
    for forbidden in ("cookie", "header", "raw_body", "session_path", "environment_variable"):
        assert forbidden not in serialized

    small_scope = RegistrationMediaScope(
        total_program_item_count=2,
        candidate_program_item_count=1,
        r3_predecessor_program_item_count=1,
        policy_request_count=1,
        transport_eligible_request_count=1,
        transport_eligible_program_item_count=1,
        boundary_request_count=0,
        boundary_program_item_count=0,
    )
    small_plan = build_candidate_plan(
        r3_run_id=R3_RUN_ID,
        predecessors=PINNED_R3_PREDECESSORS,
        r3_request_results=[
            {
                "request_key": "fixture-request",
                "source_url": "https://cvpr.thecvf.com/virtual/2026/tutorial/36154?token=redacted",
                "status": "policy_excluded_official_video",
                "linked_program_item_ids": ["candidate"],
            }
        ],
        r3_program_item_results=[
            {"program_item_id": "candidate", "request_key": "fixture-request", "blocker": {"code": "policy_excluded_official_video"}},
            {"program_item_id": "prior", "request_key": "r3", "blocker": {"code": "no_admissible_artifact"}},
        ],
        expected_scope=small_scope,
    )
    assert "token=redacted" not in small_plan["request_rows"][0]["source_url"]


_SESSION_ENV_NAME = "CVPR2026_VIRTUAL_COOKIES_FILE"
_COOKIE_PATH = "/external/session/cvpr-cookie.jar"
_COOKIE_BYTES = "cvpr-cookie-name=cvpr-cookie-value"
_WORKSPACE = Path("/repo/conferences/cvpr2026")
_REPOSITORY = Path("/repo")


def _safe_capacity(plan: dict[str, object]) -> dict[str, object]:
    return {
        "status": "pass_pre_request",
        "positive_available_bytes": True,
        "no_cap_hits": True,
        "manifest_sha256": "a" * 64,
        "candidate_plan_sha256": plan["candidate_plan_sha256"],
    }


def _fake_stat(*, mode: int = 0o600, uid: int = 501, kind: int = stat.S_IFREG) -> SimpleNamespace:
    return SimpleNamespace(st_mode=kind | mode, st_uid=uid)


def _safe_preflight(
    plan: dict[str, object],
    *,
    path: str = _COOKIE_PATH,
    stat_result: SimpleNamespace | None = None,
    readable: bool = True,
) -> dict[str, object]:
    return preflight_registered_virtual_session(
        workspace=_WORKSPACE,
        repository_root=_REPOSITORY,
        candidate_plan=plan,
        boundary=FIXTURE_BOUNDARY,
        verified_predecessors=PINNED_R3_PREDECESSORS,
        capacity=_safe_capacity(plan),
        environ={_SESSION_ENV_NAME: path},
        lstat=lambda _path: stat_result or _fake_stat(),
        is_readable=lambda _path: readable,
        current_uid=lambda: 501,
    )


def test_external_session_preflight_serializes_only_safe_metadata() -> None:
    plan, _ = _plan()
    preflight = _safe_preflight(plan)

    assert preflight["auth_mode"] == "external_cookie_session"
    assert preflight["auth_status"] == "preflight_passed"
    assert preflight["preflight_status"] == "pass"
    assert preflight["transport_eligible_request_count"] == 2_573
    assert preflight["secrets_copied"] is False
    serialized = json.dumps([preflight, {"exception": "blocked"}], sort_keys=True)
    for forbidden in (_SESSION_ENV_NAME, _COOKIE_PATH, _COOKIE_BYTES, "cvpr-cookie-name"):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    ("path", "stat_result", "readable"),
    [
        ("", _fake_stat(), True),
        ("/repo/conferences/cvpr2026/session/cookies", _fake_stat(), True),
        (_COOKIE_PATH, _fake_stat(kind=stat.S_IFLNK), True),
        (_COOKIE_PATH, _fake_stat(kind=stat.S_IFDIR), True),
        (_COOKIE_PATH, _fake_stat(uid=502), True),
        (_COOKIE_PATH, _fake_stat(mode=0o640), True),
        (_COOKIE_PATH, _fake_stat(), False),
    ],
)
def test_invalid_external_session_metadata_creates_no_jar_or_opener(
    path: str,
    stat_result: SimpleNamespace,
    readable: bool,
) -> None:
    plan, _ = _plan()
    jar_loads: list[object] = []
    opener_builds: list[object] = []
    opener, preflight = cvpr_registered_virtual_opener_factory(
        workspace=_WORKSPACE,
        repository_root=_REPOSITORY,
        candidate_plan=plan,
        verified_predecessors=PINNED_R3_PREDECESSORS,
        capacity=_safe_capacity(plan),
        environ={_SESSION_ENV_NAME: path},
        lstat=lambda _path: stat_result,
        is_readable=lambda _path: readable,
        current_uid=lambda: 501,
        cookie_jar_loader=lambda _path: jar_loads.append(_path),
        opener_builder=lambda _jar: opener_builds.append(_jar),
    )

    assert opener is None
    assert preflight["preflight_status"] == "blocked"
    assert preflight["auth_status"] == "registration_auth_required"
    assert jar_loads == []
    assert opener_builds == []
    assert _SESSION_ENV_NAME not in json.dumps(preflight, sort_keys=True)


@pytest.mark.parametrize("invalid_kind", ["predecessor", "candidate_digest", "capacity"])
def test_invalid_lineage_or_capacity_stops_before_cookie_jar_load(invalid_kind: str) -> None:
    plan, _ = _plan()
    predecessors = copy.deepcopy(PINNED_R3_PREDECESSORS)
    capacity = _safe_capacity(plan)
    if invalid_kind == "predecessor":
        predecessors[0]["file_sha256"] = "0" * 64
    elif invalid_kind == "candidate_digest":
        plan["candidate_plan_sha256"] = "0" * 64
    else:
        capacity["no_cap_hits"] = False

    jar_loads: list[object] = []
    opener, preflight = cvpr_registered_virtual_opener_factory(
        workspace=_WORKSPACE,
        repository_root=_REPOSITORY,
        candidate_plan=plan,
        verified_predecessors=predecessors,
        capacity=capacity,
        environ={_SESSION_ENV_NAME: _COOKIE_PATH},
        lstat=lambda _path: _fake_stat(),
        is_readable=lambda _path: True,
        current_uid=lambda: 501,
        cookie_jar_loader=lambda _path: jar_loads.append(_path),
        opener_builder=lambda _jar: pytest.fail("invalid preflight built an opener"),
    )

    assert opener is None
    assert preflight["preflight_status"] == "blocked"
    assert jar_loads == []


class _FakeResponse:
    def __init__(
        self,
        status: int,
        *,
        url: str = "https://cvpr.thecvf.com/virtual/2026/poster/1",
        location: str | None = None,
        authenticated_account: bool = False,
        registration_wall: bool = False,
        session_expired: bool = False,
        body: bytes | None = None,
        content_type: str | None = None,
    ) -> None:
        self.status = status
        self.url = url
        self.headers = {} if location is None else {"Location": location}
        if body is None:
            if authenticated_account:
                body = b'<main data-authenticated-account="true"></main>'
            elif registration_wall:
                body = b'<main data-registration-wall="true">Registration required</main>'
            elif session_expired:
                body = b"<main>Session expired</main>"
            else:
                body = b""
        if content_type is None and body:
            content_type = "text/html"
        if content_type is not None:
            self.headers["Content-Type"] = content_type
        self.authenticated_account = authenticated_account
        self.registration_wall = registration_wall
        self.session_expired = session_expired
        self.body = body

    def read(self, _size: int) -> bytes:
        return self.body

    def close(self) -> None:
        return None


class _FakeOpener:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.urls: list[str] = []

    def __call__(self, url: str) -> _FakeResponse:
        self.urls.append(url)
        result = self.responses.pop(0)
        if isinstance(result, BaseException):
            raise result
        assert hasattr(result, "status") and hasattr(result, "headers") and hasattr(result, "url")
        return result


def _factory_with_fake_opener(plan: dict[str, object], fake_opener: _FakeOpener):
    jar_loads: list[Path] = []
    opener, preflight = cvpr_registered_virtual_opener_factory(
        workspace=_WORKSPACE,
        repository_root=_REPOSITORY,
        candidate_plan=plan,
        verified_predecessors=PINNED_R3_PREDECESSORS,
        capacity=_safe_capacity(plan),
        environ={_SESSION_ENV_NAME: _COOKIE_PATH},
        lstat=lambda _path: _fake_stat(),
        is_readable=lambda _path: True,
        current_uid=lambda: 501,
        cookie_jar_loader=lambda path: jar_loads.append(path) or object(),
        opener_builder=lambda _jar: fake_opener,
    )
    assert preflight["preflight_status"] == "pass"
    assert jar_loads == [Path(_COOKIE_PATH)]
    assert opener is not None
    return opener, fake_opener


def test_registered_opener_accepts_only_frozen_urls_and_validates_redirects_before_next_hop() -> None:
    plan, _ = _plan()
    allowed = transport_eligible_requests(plan)
    initial = allowed[0]["source_url"]
    redirected = allowed[1]["source_url"]
    fake = _FakeOpener(
        [
            _FakeResponse(302, location=redirected),
            _FakeResponse(200, url=redirected, authenticated_account=True),
        ]
    )
    opener, fake = _factory_with_fake_opener(plan, fake)

    outcome = opener(initial)

    assert outcome == {
        "auth_mode": "external_cookie_session",
        "auth_status": "authenticated",
        "status": "authenticated",
        "http_status": 200,
        "final_url": redirected,
        "redirect_count": 1,
        "secrets_copied": False,
    }
    assert fake.urls == [initial, redirected]
    with pytest.raises(ValueError, match="frozen transport-eligible"):
        opener("https://cvpr.thecvf.com/virtual/2026/tutorial/not-frozen")
    with pytest.raises(ValueError, match="frozen transport-eligible"):
        opener("https://outside.example/virtual/2026/poster/1")


@pytest.mark.parametrize(
    "target",
    (
        "https://cvpr.thecvf.com/virtual/2026/../account",
        "https://cvpr.thecvf.com/virtual/2026/not-an-event/secret",
    ),
)
def test_registered_opener_rejects_dot_segment_and_unknown_virtual_redirects(target: str) -> None:
    plan, _ = _plan()
    initial = transport_eligible_requests(plan)[0]["source_url"]
    fake = _FakeOpener([_FakeResponse(302, location=target)])
    opener, fake = _factory_with_fake_opener(plan, fake)

    outcome = opener(initial)

    assert outcome["status"] == "unreachable_authenticated_media"
    assert fake.urls == [initial]


def test_authenticated_event_opener_requires_frozen_redirect_and_final_urls() -> None:
    plan, _ = _plan()
    allowed = transport_eligible_requests(plan)
    initial = allowed[0]["source_url"]
    redirected = allowed[1]["source_url"]

    redirect_fake = _FakeOpener([_FakeResponse(302, location=redirected), _FakeResponse(200, url=redirected, authenticated_account=True)])
    opener, preflight = cvpr_authenticated_event_opener_factory(
        workspace=_WORKSPACE,
        repository_root=_REPOSITORY,
        frozen_urls=[initial, redirected],
        environ={_SESSION_ENV_NAME: _COOKIE_PATH},
        lstat=lambda _path: _fake_stat(),
        is_readable=lambda _path: True,
        current_uid=lambda: 501,
        cookie_jar_loader=lambda _path: object(),
        opener_builder=lambda _jar: redirect_fake,
    )
    assert preflight["preflight_status"] == "pass"
    assert opener is not None
    assert opener(initial)["status"] == "authenticated"
    assert redirect_fake.urls == [initial, redirected]

    escaped = "https://cvpr.thecvf.com/virtual/2026/../account"
    final_fake = _FakeOpener([_FakeResponse(200, url=escaped, authenticated_account=True)])
    opener, preflight = cvpr_authenticated_event_opener_factory(
        workspace=_WORKSPACE,
        repository_root=_REPOSITORY,
        frozen_urls=[initial],
        environ={_SESSION_ENV_NAME: _COOKIE_PATH},
        lstat=lambda _path: _fake_stat(),
        is_readable=lambda _path: True,
        current_uid=lambda: 501,
        cookie_jar_loader=lambda _path: object(),
        opener_builder=lambda _jar: final_fake,
    )
    assert preflight["preflight_status"] == "pass"
    assert opener is not None
    assert opener(initial)["status"] == "unreachable_authenticated_media"
    assert final_fake.urls == [initial]


def test_registered_opener_can_supply_bounded_html_only_to_the_in_memory_recovery_lane() -> None:
    plan, _ = _plan()
    initial = transport_eligible_requests(plan)[0]["source_url"]
    fake = _FakeOpener(
        [
            _FakeResponse(
                200,
                authenticated_account=True,
                body=b'<main data-authenticated-account="true"><a href="https://youtu.be/abc123"></a></main>',
                content_type="text/html; charset=utf-8",
            )
        ]
    )
    opener, _ = cvpr_registered_virtual_opener_factory(
        workspace=_WORKSPACE,
        repository_root=_REPOSITORY,
        candidate_plan=plan,
        verified_predecessors=PINNED_R3_PREDECESSORS,
        capacity=_safe_capacity(plan),
        environ={_SESSION_ENV_NAME: _COOKIE_PATH},
        lstat=lambda _path: _fake_stat(),
        is_readable=lambda _path: True,
        current_uid=lambda: 501,
        cookie_jar_loader=lambda _path: object(),
        opener_builder=lambda _jar: fake,
        capture_page_body=True,
    )

    assert opener is not None
    outcome = opener(initial)
    assert outcome["status"] == "authenticated"
    assert outcome["content_type"] == "text/html; charset=utf-8"
    assert outcome["body"] == b'<main data-authenticated-account="true"><a href="https://youtu.be/abc123"></a></main>'


def test_registered_opener_accepts_a_bounded_three_megabyte_cvpr_event_page() -> None:
    plan, _ = _plan()
    initial = transport_eligible_requests(plan)[0]["source_url"]
    body = b'<main data-authenticated-account="true">' + (b"x" * (3 * 1024 * 1024)) + b"</main>"
    fake = _FakeOpener([_FakeResponse(200, authenticated_account=True, body=body, content_type="text/html")])
    opener, _ = cvpr_registered_virtual_opener_factory(
        workspace=_WORKSPACE,
        repository_root=_REPOSITORY,
        candidate_plan=plan,
        verified_predecessors=PINNED_R3_PREDECESSORS,
        capacity=_safe_capacity(plan),
        environ={_SESSION_ENV_NAME: _COOKIE_PATH},
        lstat=lambda _path: _fake_stat(),
        is_readable=lambda _path: True,
        current_uid=lambda: 501,
        cookie_jar_loader=lambda _path: object(),
        opener_builder=lambda _jar: fake,
        capture_page_body=True,
    )

    assert opener is not None
    outcome = opener(initial)
    assert outcome["status"] == "authenticated"
    assert outcome["body"] == body


def test_registered_opener_derives_access_state_from_standard_bounded_html_not_custom_response_attributes() -> None:
    plan, _ = _plan()
    initial = transport_eligible_requests(plan)[0]["source_url"]

    class StandardResponse:
        def __init__(self, body: bytes) -> None:
            self.status = 200
            self.url = initial
            self.headers = {"Content-Type": "text/html; charset=utf-8"}
            self._body = body

        def read(self, _size: int) -> bytes:
            return self._body

        def close(self) -> None:
            return None

    authenticated = _FakeOpener(
        [StandardResponse(b'<main data-authenticated-account="true">session</main>')]
    )
    opener, _ = _factory_with_fake_opener(plan, authenticated)
    assert opener(initial)["status"] == "authenticated"

    registration_wall = _FakeOpener(
        [StandardResponse(b'<main data-registration-wall="true">Registration required</main>')]
    )
    opener, _ = _factory_with_fake_opener(plan, registration_wall)
    outcome = opener(initial)
    assert outcome["status"] == "registration_auth_required"
    assert outcome["auth_status"] == "registration_auth_required"


@pytest.mark.parametrize(
    ("responses", "expected_status", "expected_auth_status", "call_count"),
    [
        ([_FakeResponse(401)], "registration_auth_required", "registration_auth_required", 1),
        ([_FakeResponse(403, authenticated_account=True)], "registered_access_denied", "registered_access_denied", 1),
        ([_FakeResponse(200, registration_wall=True)], "registration_auth_required", "registration_auth_required", 1),
        ([_FakeResponse(200, session_expired=True)], "registration_auth_required", "registration_auth_required", 1),
        ([_FakeResponse(404, authenticated_account=True)], "official_video_not_published", "authenticated", 1),
        ([TimeoutError("do not serialize this")], "unreachable_authenticated_media", "registration_auth_required", 1),
        (
            [_FakeResponse(302, location="https://cvpr.thecvf.com/accounts/login")],
            "registration_auth_required",
            "registration_auth_required",
            1,
        ),
        (
            [_FakeResponse(302, location="https://outside.example/virtual/2026/poster/9")],
            "unreachable_authenticated_media",
            "registration_auth_required",
            1,
        ),
    ],
)
def test_registered_opener_maps_auth_and_redirect_failures_without_serializing_raw_inputs(
    responses: list[object],
    expected_status: str,
    expected_auth_status: str,
    call_count: int,
) -> None:
    plan, _ = _plan()
    initial = transport_eligible_requests(plan)[0]["source_url"]
    fake = _FakeOpener(responses)
    opener, fake = _factory_with_fake_opener(plan, fake)

    outcome = opener(initial)

    assert outcome["status"] == expected_status
    assert outcome["auth_status"] == expected_auth_status
    assert outcome["secrets_copied"] is False
    assert len(fake.urls) == call_count
    serialized = json.dumps(outcome, sort_keys=True)
    for forbidden in (_SESSION_ENV_NAME, _COOKIE_PATH, _COOKIE_BYTES, "do not serialize this"):
        assert forbidden not in serialized


def _small_media_scope() -> RegistrationMediaScope:
    return RegistrationMediaScope(
        total_program_item_count=2,
        candidate_program_item_count=1,
        r3_predecessor_program_item_count=1,
        policy_request_count=1,
        transport_eligible_request_count=1,
        transport_eligible_program_item_count=1,
        boundary_request_count=0,
        boundary_program_item_count=0,
    )


def _small_media_overlay() -> tuple[dict[str, object], dict[str, object]]:
    scope = _small_media_scope()
    request_url = "https://cvpr.thecvf.com/virtual/2026/tutorial/36154"
    plan = build_candidate_plan(
        r3_run_id=R3_RUN_ID,
        predecessors=PINNED_R3_PREDECESSORS,
        r3_request_results=[
            {
                "request_key": "official-page",
                "source_url": request_url,
                "status": "policy_excluded_official_video",
                "linked_program_item_ids": ["candidate"],
                "source_evidence": {"source_record_id": "r3-source"},
            }
        ],
        r3_program_item_results=[
            {
                "program_item_id": "candidate",
                "request_key": "official-page",
                "blocker": {"code": "policy_excluded_official_video"},
            },
            {
                "program_item_id": "r3-predecessor",
                "request_key": "r3-request",
                "blocker": {"code": "no_admissible_artifact"},
            },
        ],
        expected_scope=scope,
    )
    return build_registration_media_overlay(
        candidate_plan=plan,
        r3_program_item_results=[
            {
                "program_item_id": "candidate",
                "request_key": "official-page",
                "blocker": {"code": "policy_excluded_official_video"},
            },
            {
                "program_item_id": "r3-predecessor",
                "request_key": "r3-request",
                "blocker": {"code": "no_admissible_artifact"},
            },
        ],
        expected_scope=scope,
    ), plan


class _FakePageRecoveryOpener:
    def __init__(self, page_results: list[object]) -> None:
        self.page_results = page_results
        self.urls: list[str] = []

    def __call__(self, url: str) -> object:
        self.urls.append(url)
        result = self.page_results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def _authenticated_page(html: str, *, url: str = "https://cvpr.thecvf.com/virtual/2026/tutorial/36154") -> dict[str, object]:
    return {
        "status": "authenticated",
        "auth_status": "authenticated",
        "http_status": 200,
        "final_url": url,
        "content_type": "text/html; charset=utf-8",
        "body": html.encode("utf-8"),
    }


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
    ],
)
def test_registered_page_recovery_canonicalizes_direct_youtube_provenance(tmp_path: Path, url: str) -> None:
    overlay, plan = _small_media_overlay()
    opener = _FakePageRecoveryOpener([_authenticated_page(f'<a href="{url}">watch</a>')])

    recovered = recover_registered_official_pages(
        overlay=overlay,
        candidate_plan=plan,
        workspace=tmp_path,
        authenticated_page_opener=opener,
        expected_scope=_small_media_scope(),
    )

    request = recovered["official_event_requests"][0]
    media = recovered["official_event_media_records"]
    assert request["result_status"] == "direct_youtube_recovered"
    assert request["page_byte_count"] > 0
    assert len(request["page_sha256"]) == 64
    assert media == [
        {
            "parent_request_key": "official-page",
            "linked_program_item_ids": ["candidate"],
            "video_id": "dQw4w9WgXcQ",
            "media_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "provenance": "direct_youtube_reference",
            "caption_status": "youtube_caption_unavailable",
            "caption": None,
            "audio_downloaded": False,
            "video_downloaded": False,
        }
    ]
    assert "body" not in json.dumps(recovered, sort_keys=True)


def test_cvpr_local_youtube_nocookie_exception_requires_validated_parent_and_does_not_widen_global_hosts(tmp_path: Path) -> None:
    overlay, plan = _small_media_overlay()
    original_hosts = EMBEDDED_MEDIA_HOSTS
    html = '<iframe src="https://www.youtube-nocookie.com/embed/DNfzBTW-Drw"></iframe>'
    opener = _FakePageRecoveryOpener([_authenticated_page(html)])

    recovered = recover_registered_official_pages(
        overlay=overlay,
        candidate_plan=plan,
        workspace=tmp_path,
        authenticated_page_opener=opener,
        expected_scope=_small_media_scope(),
    )

    assert EMBEDDED_MEDIA_HOSTS == original_hosts
    assert "youtube-nocookie.com" not in EMBEDDED_MEDIA_HOSTS
    assert recovered["official_event_media_records"][0]["media_url"] == "https://www.youtube.com/watch?v=DNfzBTW-Drw"
    assert recovered["official_event_media_records"][0]["provenance"] == "cvpr_local_youtube_nocookie_embed"
    assert extract_cvpr_youtube_nocookie_links(
        html,
        parent_url="https://outside.example/virtual/2026/tutorial/36154",
    ) == []


def test_registered_page_recovery_records_unsupported_and_no_link_outcomes(tmp_path: Path) -> None:
    for html, expected in (
        ('<iframe src="https://player.vimeo.com/video/12345"></iframe>', "unsupported_official_embed_host"),
        ("<main>no embedded recording</main>", "no_admissible_official_youtube_link"),
    ):
        overlay, plan = _small_media_overlay()
        recovered = recover_registered_official_pages(
            overlay=overlay,
            candidate_plan=plan,
            workspace=tmp_path,
            authenticated_page_opener=_FakePageRecoveryOpener([_authenticated_page(html)]),
            expected_scope=_small_media_scope(),
        )
        assert recovered["official_event_requests"][0]["result_status"] == expected
        assert recovered["official_event_media_records"] == []


def test_r4_media_parser_extracts_marker_substring_youtube_id_before_sanitization() -> None:
    candidates, unsupported = _media_candidates_from_page(
        page_html='<iframe src="https://www.youtube.com/embed/AbcToken123"></iframe>',
        parent_url="https://cvpr.thecvf.com/virtual/2026/tutorial/36154",
    )

    assert candidates == {"AbcToken123": "direct_youtube_reference"}
    assert unsupported is False


def test_registered_page_recovery_reuses_caption_primitive_and_never_claims_downloaded_media(tmp_path: Path) -> None:
    overlay, plan = _small_media_overlay()
    caption_path = tmp_path / "storage/download-corpus/registration-media/captions/dQw4w9WgXcQ.en.vtt"
    caption_path.parent.mkdir(parents=True)
    caption_path.write_text("WEBVTT\n", encoding="utf-8")

    def caption_helper(video, destination):
        assert video.video_id == "dQw4w9WgXcQ"
        assert destination == caption_path.parent
        return {"caption_path": caption_path, "transcript_source": "youtube_caption_manual", "language": "en"}

    recovered = recover_registered_official_pages(
        overlay=overlay,
        candidate_plan=plan,
        workspace=tmp_path,
        authenticated_page_opener=_FakePageRecoveryOpener(
            [_authenticated_page("https://www.youtube.com/watch?v=dQw4w9WgXcQ")]
        ),
        caption_helper=caption_helper,
        expected_scope=_small_media_scope(),
    )

    media = recovered["official_event_media_records"][0]
    assert media["caption_status"] == "youtube_caption_available"
    assert media["caption"] == {
        "local_path": "storage/download-corpus/registration-media/captions/dQw4w9WgXcQ.en.vtt",
        "mime": "text/vtt",
        "byte_count": len(b"WEBVTT\n"),
        "sha256": __import__("hashlib").sha256(b"WEBVTT\n").hexdigest(),
        "transcript_source": "youtube_caption_manual",
        "language": "en",
    }
    assert media["audio_downloaded"] is False
    assert media["video_downloaded"] is False


def test_resume_rejects_stale_overlay_and_rechecks_changed_local_caption(tmp_path: Path) -> None:
    overlay, plan = _small_media_overlay()
    caption_path = tmp_path / "storage/download-corpus/registration-media/captions/dQw4w9WgXcQ.en.vtt"
    caption_path.parent.mkdir(parents=True)
    caption_path.write_text("first", encoding="utf-8")
    recovered = recover_registered_official_pages(
        overlay=overlay,
        candidate_plan=plan,
        workspace=tmp_path,
        authenticated_page_opener=_FakePageRecoveryOpener(
            [_authenticated_page("https://www.youtube.com/watch?v=dQw4w9WgXcQ")]
        ),
        caption_helper=lambda _video, _dest: {"caption_path": caption_path, "transcript_source": "youtube_caption_auto"},
        expected_scope=_small_media_scope(),
    )
    assert recovered["official_event_media_records"][0]["caption"]["transcript_source"] == "youtube_caption_auto"
    stale = copy.deepcopy(recovered)
    stale["predecessor_lineage"]["candidate_plan_sha256"] = "0" * 64
    stale["overlay_sha256"] = __import__("conference_pipeline.acquisition.cvpr_registration_media", fromlist=["canonical_registration_media_digest"]).canonical_registration_media_digest(stale)
    with pytest.raises(ValueError, match="candidate-plan"):
        resume_registration_media_overlay(
            fresh_overlay=build_registration_media_overlay(
                candidate_plan=plan,
                r3_program_item_results=[
                    {"program_item_id": "candidate", "request_key": "official-page", "blocker": {"code": "policy_excluded_official_video"}},
                    {"program_item_id": "r3-predecessor", "request_key": "r3-request", "blocker": {"code": "no_admissible_artifact"}},
                ],
                expected_scope=_small_media_scope(),
            ),
            prior_overlay=stale,
            workspace=tmp_path,
            expected_scope=_small_media_scope(),
        )

    caption_path.write_text("changed", encoding="utf-8")
    resumed = resume_registration_media_overlay(
        fresh_overlay=build_registration_media_overlay(
            candidate_plan=plan,
            r3_program_item_results=[
                {"program_item_id": "candidate", "request_key": "official-page", "blocker": {"code": "policy_excluded_official_video"}},
                {"program_item_id": "r3-predecessor", "request_key": "r3-request", "blocker": {"code": "no_admissible_artifact"}},
            ],
            expected_scope=_small_media_scope(),
        ),
        prior_overlay=recovered,
        workspace=tmp_path,
        expected_scope=_small_media_scope(),
    )
    assert resumed["official_event_requests"][0]["result_status"] == "pending"
    assert resumed["official_event_media_records"] == []


def test_partial_checkpoint_resume_reuses_only_completed_matching_request_rows(tmp_path: Path) -> None:
    scope = RegistrationMediaScope(
        total_program_item_count=3,
        candidate_program_item_count=2,
        r3_predecessor_program_item_count=1,
        policy_request_count=2,
        transport_eligible_request_count=2,
        transport_eligible_program_item_count=2,
        boundary_request_count=0,
        boundary_program_item_count=0,
    )
    first_url = "https://cvpr.thecvf.com/virtual/2026/tutorial/1"
    second_url = "https://cvpr.thecvf.com/virtual/2026/tutorial/2"
    requests = [
        {"request_key": "first", "source_url": first_url, "status": "policy_excluded_official_video", "linked_program_item_ids": ["first-item"]},
        {"request_key": "second", "source_url": second_url, "status": "policy_excluded_official_video", "linked_program_item_ids": ["second-item"]},
    ]
    items = [
        {"program_item_id": "first-item", "request_key": "first", "blocker": {"code": "policy_excluded_official_video"}},
        {"program_item_id": "second-item", "request_key": "second", "blocker": {"code": "policy_excluded_official_video"}},
        {"program_item_id": "r3-item", "request_key": "r3", "blocker": {"code": "no_admissible_artifact"}},
    ]
    plan = build_candidate_plan(
        r3_run_id=R3_RUN_ID,
        predecessors=PINNED_R3_PREDECESSORS,
        r3_request_results=requests,
        r3_program_item_results=items,
        expected_scope=scope,
    )
    overlay = build_registration_media_overlay(candidate_plan=plan, r3_program_item_results=items, expected_scope=scope)
    snapshots: list[dict[str, object]] = []

    def checkpoint(snapshot: dict[str, object]) -> None:
        snapshots.append(copy.deepcopy(snapshot))
        raise RuntimeError("simulated interruption after durable checkpoint")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        recover_registered_official_pages(
            overlay=overlay,
            candidate_plan=plan,
            workspace=tmp_path,
            authenticated_page_opener=_FakePageRecoveryOpener([_authenticated_page("<main>none</main>", url=first_url)]),
            checkpoint=checkpoint,
            expected_scope=scope,
        )
    assert snapshots[0]["official_event_requests"][0]["result_status"] == "no_admissible_official_youtube_link"
    resumed = resume_registration_media_overlay(
        fresh_overlay=build_registration_media_overlay(candidate_plan=plan, r3_program_item_results=items, expected_scope=scope),
        prior_overlay=snapshots[0],
        workspace=tmp_path,
        expected_scope=scope,
    )
    assert [row.get("result_status", "pending") for row in resumed["official_event_requests"]] == [
        "no_admissible_official_youtube_link",
        "pending",
    ]
    remaining = _FakePageRecoveryOpener([_authenticated_page("<main>none</main>", url=second_url)])
    completed = recover_registered_official_pages(
        overlay=resumed,
        candidate_plan=plan,
        workspace=tmp_path,
        authenticated_page_opener=remaining,
        expected_scope=scope,
    )
    assert remaining.urls == [second_url]
    assert completed["recovery_counts"]["completed_request_count"] == 2


def test_full_zero_link_completion_preserves_boundary_outcomes_without_opener_calls(tmp_path: Path) -> None:
    plan, program_items = _plan()
    overlay = build_registration_media_overlay(candidate_plan=plan, r3_program_item_results=program_items)
    boundary_urls = {
        row["source_url"] for row in overlay["official_event_requests"] if not row["transport_eligible"]
    }

    def no_link_opener(url: str) -> dict[str, object]:
        assert url not in boundary_urls
        return _authenticated_page("<main>no recording</main>", url=url)

    recovered = recover_registered_official_pages(
        overlay=overlay,
        candidate_plan=plan,
        workspace=tmp_path,
        authenticated_page_opener=no_link_opener,
    )

    assert len(recovered["official_event_media_records"]) == 0
    assert recovered["recovery_counts"] == {
        "completed_request_count": 2_579,
        "direct_youtube_media_record_count": 0,
        "transport_eligible_request_count": 2_573,
        "boundary_no_network_request_count": 6,
    }
    assert all(
        row["result_status"] == "outside_authenticated_official_event_boundary"
        for row in recovered["official_event_requests"]
        if not row["transport_eligible"]
    )
    assert all(
        row["result_status"] == "no_admissible_official_youtube_link"
        for row in recovered["official_event_requests"]
        if row["transport_eligible"]
    )
    assert all(key not in recovered for key in ("asr_queue", "ocr_queue", "provider_queue"))
