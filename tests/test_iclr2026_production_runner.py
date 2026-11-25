from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

from tests.lineage_fixtures import stamp_upstream_stage_receipts
from conference_pipeline.analysis.narrative import DEFAULT_DEEP_NARRATIVE_MODEL


REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER = REPO_ROOT / "scripts" / "run_iclr2026_production.py"
WORKSHOP_TALK_MEDIA_OWNER = REPO_ROOT / "scripts" / "acquire_iclr2026_workshop_talk_media.py"
SECRET_VALUE = "super-secret-test-value"


def write_workspace(tmp_path: Path) -> Path:
    # conferences/<slug>: since 2026-07-16 the stage-owner registry scopes
    # acquisition stages by workspace identity, and verify_stage_contract
    # refuses a workspace it cannot normalize to conferences/<slug> -- by
    # design, so one conference can never satisfy another's contract. A bare
    # "iclr2026" directory failed that check in every test that reached the
    # verifier, which read as four unrelated failures for two months.
    workspace = tmp_path / "conferences" / "iclr2026"
    (workspace / "storage").mkdir(parents=True)
    (workspace / "config.yaml").write_text(
        (
            'conference:\n'
            '  slug: "iclr2026"\n'
            '  name: "ICLR 2026"\n'
            'storage:\n'
            '  manifest_path: "storage/storage_manifest.json"\n'
        ),
        encoding="utf-8",
    )
    (workspace / "storage" / "storage_manifest.json").write_text(
        json.dumps({"schema_version": 1, "artifact_families": []}) + "\n",
        encoding="utf-8",
    )
    return workspace


def write_topic_strategy(workspace: Path, run_id: str) -> Path:
    topics = [
        ("01_llm_reasoning_test_time", "LLM reasoning and test-time compute"),
        ("02_agentic_ai_systems", "Agentic AI systems and tool-use"),
        ("03_evaluation_data_benchmarks", "Evaluation, benchmarks, and data curation"),
        ("04_generative_diffusion_flow", "Generative models: diffusion, flow, and media"),
        ("05_multimodal_world_embodied", "Multimodal, spatial, and embodied intelligence"),
        ("06_alignment_safety_security", "Alignment, safety, security, and privacy"),
        ("07_interpretability_representations", "Interpretability and representation understanding"),
        ("08_ai_for_science", "AI for science and scientific discovery"),
        ("09_efficiency_systems_architectures", "Efficiency, systems, and scalable architectures"),
        ("10_learning_dynamics_theory", "Learning dynamics, theory, and scaling laws"),
        ("11_rl_decision_robotics", "Reinforcement learning, decision-making, and robotics"),
        ("12_human_centered_society", "Human-centered AI and societal stakes"),
    ]
    path = workspace / "analysis" / "topic_strategy.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    final_topics = [
        {
            "rank": index,
            "topic_id": topic_id,
            "title": title,
            "starter_queries": [title],
            "evidence_title_patterns": title.split(" and "),
        }
        for index, (topic_id, title) in enumerate(topics, start=1)
    ]
    payload = {
        "schema_version": 1,
        "status": "succeeded",
        "mode": "live",
        "final_topics": final_topics,
        "provider_receipt_paths": ["storage/workflow-runs/test/provider-calls/topic-strategy.json"],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt_path = (
        workspace / "storage" / "analysis" / "topic-strategy" / run_id / "topic_strategy_receipt.json"
    )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_env_file(tmp_path: Path) -> Path:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                f"OPENAI_API_KEY={SECRET_VALUE}-openai",
                f"ANTHROPIC_API_KEY={SECRET_VALUE}-anthropic",
                f"GOOGLE_API_KEY={SECRET_VALUE}-google",
                f"TAVILY_API_KEY={SECRET_VALUE}-tavily",
                "ICLR2026_PROVIDER_COST_APPROVED=yes",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return env_file


def write_source_inventory(workspace: Path, base_url: str, *, missing_papers: bool = False) -> Path:
    source_urls = {
        "conference": f"{base_url}/Conferences/2026",
        "calendar": f"{base_url}/virtual/2026/calendar",
        "papers": f"{base_url}/missing-papers" if missing_papers else f"{base_url}/virtual/2026/papers.html",
        "workshops": f"{base_url}/virtual/2026/events/workshop",
        "openreview_venue": f"{base_url}/openreview",
        "downloads_generated": f"{base_url}/virtual/2026/downloads",
        "downloads_canonical_candidate": f"{base_url}/Downloads/2026",
    }
    family_specs = [
        ("iclr-conference-site", "Conference Site", source_urls["conference"]),
        ("iclr-calendar", "Calendar", source_urls["calendar"]),
        ("accepted-papers", "Accepted Papers", source_urls["papers"]),
        ("posters", "Posters", source_urls["calendar"]),
        ("oral-sessions", "Oral Sessions", source_urls["calendar"]),
        ("workshops", "Workshops", source_urls["workshops"]),
        ("invited-talks-keynotes", "Invited Talks and Keynotes", source_urls["calendar"]),
        ("tutorials", "Tutorials", source_urls["calendar"]),
        ("panels", "Panels", source_urls["calendar"]),
        ("competitions", "Competitions", source_urls["calendar"]),
        ("conference-specific-events", "Conference Specific Events", source_urls["calendar"]),
        ("openreview-venue", "OpenReview Venue", source_urls["openreview_venue"]),
        ("downloads", "Downloads", source_urls["downloads_generated"]),
    ]
    payload = {
        "schema_version": 1,
        "conference": {"slug": "iclr2026", "name": "ICLR 2026"},
        "source_urls": source_urls,
        "source_families": [
            {
                "id": family_id,
                "label": label,
                "source_url": source_url,
                "source_kind": "html",
                "expected_coverage": "test coverage",
                "status": "present",
                "support_label": "test",
                "live_proof_status": "fixture",
                "auth_assumption": "public",
                "notes": "test fixture",
            }
            for family_id, label, source_url in family_specs
        ],
    }
    path = workspace / "storage" / "source-intake" / "source_inventory.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


class _SourceHandler(BaseHTTPRequestHandler):
    success_paths = {
        "/Conferences/2026",
        "/virtual/2026/calendar",
        "/virtual/2026/papers.html",
        "/virtual/2026/poster/fixture-paper",
        "/virtual/2026/events/workshop",
        "/openreview",
        "/Downloads/2026",
    }
    binary_paths = {
        "/papers/paper-1.pdf": ("application/pdf", b"%PDF-1.4\nfixture paper\n%%EOF\n"),
        "/media/session-1.mp3": ("audio/mpeg", b"fixture-audio-bytes"),
        "/media/item-session.mp4": ("video/mp4", b"fixture-video-bytes"),
    }

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in self.binary_paths:
            content_type, body = self.binary_paths[path]
            self.send_response(200)
            self.send_header("content-type", content_type)
            self.end_headers()
            self.wfile.write(body)
            return
        if path not in self.success_paths:
            self.send_response(404)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"<html><title>Missing</title><body>missing</body></html>")
            return
        self.send_response(200)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.end_headers()
        body = (
            f"<html><head><title>{path}</title></head>"
            f"<body><a href='/linked'>linked</a><a href='{path}?tab=details'>details</a>"
            "<a href='/virtual/2026/poster/fixture-paper'>poster page</a>"
            "<a href='/papers/paper-1.pdf'>paper pdf</a>"
            "<a href='/media/session-1.mp3'>session audio</a>"
            "<a href='/media/item-session.mp4'>item video</a>"
            "</body></html>"
        )
        self.wfile.write(body.encode("utf-8"))


def start_source_server() -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SourceHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, f"http://{host}:{port}"


class _WorkshopTalkMediaHandler(BaseHTTPRequestHandler):
    pages: dict[str, str] = {}
    authenticated_pages: dict[str, str] = {}
    binaries: dict[str, tuple[str, bytes]] = {}
    required_cookie: str | None = None

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in self.binaries:
            content_type, body = self.binaries[path]
            self.send_response(200)
            self.send_header("content-type", content_type)
            self.end_headers()
            self.wfile.write(body)
            return
        has_required_cookie = bool(
            self.required_cookie and self.required_cookie in self.headers.get("Cookie", "")
        )
        if has_required_cookie and path in self.authenticated_pages:
            self.send_response(200)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(self.authenticated_pages[path].encode("utf-8"))
            return
        if path not in self.pages:
            self.send_response(404)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"<html><body>missing</body></html>")
            return
        self.send_response(200)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(self.pages[path].encode("utf-8"))


def start_workshop_talk_media_server(
    *,
    pages: dict[str, str],
    authenticated_pages: dict[str, str] | None = None,
    binaries: dict[str, tuple[str, bytes]] | None = None,
    required_cookie: str | None = None,
) -> tuple[ThreadingHTTPServer, str]:
    class Handler(_WorkshopTalkMediaHandler):
        pass

    Handler.pages = dict(pages)
    Handler.authenticated_pages = dict(authenticated_pages or {})
    Handler.binaries = dict(binaries or {})
    Handler.required_cookie = required_cookie
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, f"http://{host}:{port}"


def run_runner(args: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    full_env = os.environ.copy()
    full_env.pop("OPENAI_API_KEY", None)
    full_env.pop("ANTHROPIC_API_KEY", None)
    full_env.pop("GOOGLE_API_KEY", None)
    full_env.pop("TAVILY_API_KEY", None)
    full_env.pop("ICLR2026_PROVIDER_COST_APPROVED", None)
    if env:
        full_env.update(env)
    return subprocess.run(
        [sys.executable, str(RUNNER), *args],
        cwd=REPO_ROOT,
        env=full_env,
        text=True,
        capture_output=True,
        check=False,
    )


def run_workshop_talk_media_owner(
    args: list[str], *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    return subprocess.run(
        [sys.executable, str(WORKSHOP_TALK_MEDIA_OWNER), *args],
        cwd=REPO_ROOT,
        env=full_env,
        text=True,
        capture_output=True,
        check=False,
    )


def load_runner_module():
    spec = importlib.util.spec_from_file_location("iclr2026_production_runner_test", RUNNER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_workshop_talk_media_owner_module():
    spec = importlib.util.spec_from_file_location(
        "iclr2026_workshop_talk_media_owner_test",
        WORKSHOP_TALK_MEDIA_OWNER,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_runtime_contract_failure_writes_blocker_receipt(tmp_path: Path) -> None:
    workspace = write_workspace(tmp_path)
    runner = load_runner_module()
    stage = runner.STAGES["source-discovery"]

    receipt = runner.build_stage_receipt(
        workspace=workspace,
        run_id="contract-blocker-test",
        stage=stage,
        mode="live",
        status="succeeded",
        started_at=runner.utc_now(),
        checks={
            "provider_calls_made": False,
            "live_provider_calls_made": False,
            "secrets_copied": False,
        },
    )

    assert receipt["status"] == "blocked"
    assert receipt["stage_contract"]["status"] == "blocked"
    assert receipt["blocker_receipt_path"]
    blocker_path = Path(str(receipt["blocker_receipt_path"]))
    assert blocker_path.exists()
    blocker = json.loads(blocker_path.read_text(encoding="utf-8"))
    assert blocker["blocker_class"] == "stage_blocked"
    assert blocker["stage_id"] == "source-discovery"
    assert "stage_contract_required_output_missing" in blocker["reason"]

    summary = runner.write_summary(
        workspace=workspace,
        run_id="contract-blocker-test",
        mode="live",
        status="succeeded",
        stage_receipt=receipt,
    )
    assert summary["status"] == "blocked"
    assert summary["blocker_receipt_path"] == str(blocker_path)


def write_pdf_fixture(path: Path, text: str) -> None:
    import fitz

    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def write_png_fixture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753"
            "de0000000c49444154789c6360f8ffff3f0005fe02fea7c5819b0000000049454e44ae426082"
        )
    )


def write_workshop_media_inventory_fixture(workspace: Path, venueids: list[str]) -> None:
    workshops = [
        {
            "venueid": venueid,
            "paper_count": 1,
            "pdf_present_count": 1,
            "missing_pdf_count": 0,
            "pdf_availability_class": "openreview_with_pdf",
            "tiers": {"Accept": 1},
            "sample_notes": [],
        }
        for venueid in venueids
    ]
    workshops_dir = workspace / "storage" / "workshops"
    workshops_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = workshops_dir / "inventory.json"
    inventory_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "receipt_type": "conference_pipeline.workshop_inventory",
                "workshop_count": len(workshops),
                "workshops": workshops,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (workshops_dir / "download_receipt.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "receipt_type": "conference_pipeline.workshop_download_receipt",
                "status": "succeeded",
                "inventory_path": str(inventory_path.relative_to(workspace)),
                "totals": {
                    "available": len(workshops),
                    "succeeded": len(workshops),
                    "failed": 0,
                    "resume_skipped": 0,
                },
                "per_workshop": [],
                "skipped_external_workshops": [],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_transcription_ocr_download_fixture(
    workspace: Path,
    run_id: str,
    *,
    stage_status: str = "succeeded",
    write_stage_receipt: bool = True,
) -> dict[str, Path]:
    main_pdf = workspace / "storage" / "download-corpus" / "papers" / "main-paper.pdf"
    workshop_dir = workspace / "storage" / "workshops" / "paper-downloads"
    workshop_pdf = workshop_dir / "workshop-paper.pdf"
    image_path = workspace / "storage" / "download-corpus" / "media" / "poster.png"
    write_pdf_fixture(main_pdf, "Main paper fixture text for Phase Five.")
    write_pdf_fixture(workshop_pdf, "Workshop paper fixture text for Phase Five.")
    write_png_fixture(image_path)

    run_path = workspace / "storage" / "workflow-runs" / run_id
    run_path.mkdir(parents=True, exist_ok=True)
    stage_receipt_path = run_path / "stage-04-download-corpus.json"
    if write_stage_receipt:
        stage_receipt_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "receipt_type": "conference_pipeline.iclr2026_production_stage_receipt",
                    "run_id": run_id,
                    "workspace": str(workspace),
                    "stage_id": "download-corpus",
                    "stage_order": 4,
                    "mode": "live",
                    "status": stage_status,
                    "checks": {
                        "local_artifact_count": 3,
                        "direct_download_artifact_count": 2,
                        "paper_artifact_count": 2,
                        "workshop_paper_pdf_count": 1,
                        "media_artifact_count": 1,
                        "live_provider_calls_made": False,
                        "secrets_copied": False,
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    main_artifact = {
        "source_family_id": "openreview-venue",
        "artifact_kind": "paper_pdf",
        "status": "succeeded",
        "content_type": "application/pdf",
        "local_path": str(main_pdf.relative_to(workspace)),
        "byte_count": main_pdf.stat().st_size,
        "sha256": file_sha256(main_pdf),
        "secret_values_recorded": False,
    }
    image_artifact = {
        "source_family_id": "posters",
        "artifact_kind": "media_asset",
        "status": "succeeded",
        "content_type": "image/png",
        "local_path": str(image_path.relative_to(workspace)),
        "byte_count": image_path.stat().st_size,
        "sha256": file_sha256(image_path),
        "secret_values_recorded": False,
    }
    inventory_path = workspace / "storage" / "download-corpus" / "download_corpus.json"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "receipt_type": "conference_pipeline.iclr2026_download_corpus",
                "workspace": str(workspace),
                "run_id": run_id,
                "stage_id": "download-corpus",
                "status": "succeeded",
                "local_artifact_count": 3,
                "direct_download_artifact_count": 2,
                "paper_artifact_count": 2,
                "workshop_paper_pdf_count": 1,
                "media_artifact_count": 1,
                "local_artifacts": [main_artifact, image_artifact],
                "direct_download_artifacts": [main_artifact, image_artifact],
                "workshop_acquisition": {
                    "status": "succeeded",
                    "download": {
                        "output_dir": str(workshop_dir.relative_to(workspace)),
                        "totals": {"available": 1, "succeeded": 1, "failed": 0},
                    },
                },
                "blockers": [],
                "boundaries": {
                    "live_provider_calls_made": False,
                    "secrets_copied": False,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "stage_receipt": stage_receipt_path,
        "inventory": inventory_path,
        "main_pdf": main_pdf,
        "workshop_pdf": workshop_pdf,
        "image": image_path,
    }


def write_workshop_talk_media_scope_fixture(
    workspace: Path,
    run_id: str,
    *,
    records: list[dict[str, object]] | None = None,
    blockers: list[dict[str, object]] | None = None,
) -> dict[str, Path]:
    workshops_dir = workspace / "storage" / "workshops"
    workshops_dir.mkdir(parents=True, exist_ok=True)
    default_blocker = {
        "code": "inaccessible_agenda",
        "reason": "fixture workshop agenda was inaccessible",
        "retryable": True,
        "source_url": "https://iclr.cc/virtual/2026/workshop/demo",
        "media_url": None,
        "workshop_id": "ICLR.cc/2026/Workshop/DEMO",
        "workshop_title": "Demo Workshop",
    }
    blockers = list(blockers if blockers is not None else [default_blocker])
    if records is None:
        records = [
            {
                "status": "blocked",
                "blocker_code": blocker.get("code"),
                "reason": blocker.get("reason"),
                "source_url": blocker.get("source_url"),
                "media_url": blocker.get("media_url"),
                "local_path": None,
                "sha256": None,
                "byte_count": None,
                "workshop_id": blocker.get("workshop_id"),
                "workshop_title": blocker.get("workshop_title"),
                "talk_id": None,
                "talk_title": None,
                "talk_speaker": None,
            }
            for blocker in blockers
        ]
    media_record_count = sum(1 for record in records if record.get("status") == "succeeded")
    blocked_workshop_count = len(
        {
            str(record.get("workshop_id") or blocker.get("workshop_id"))
            for record, blocker in zip(records, blockers or [{}] * len(records), strict=False)
            if record.get("status") != "succeeded" or blocker.get("code")
        }
    )
    inventory_path = workshops_dir / "talk_media_inventory.json"
    inventory_payload = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.workshop_talk_media_inventory",
        "workspace": str(workspace),
        "run_id": run_id,
        "status": "succeeded",
        "workshop_count": max(len(records), len(blockers), 1),
        "media_record_count": media_record_count,
        "blocked_workshop_count": blocked_workshop_count,
        "blocker_count": len(blockers),
        "required_media_fields": [
            "workshop_id",
            "workshop_title",
            "talk_id",
            "talk_title",
            "talk_speaker",
            "source_url",
            "media_url",
            "local_path",
            "sha256",
            "byte_count",
            "status",
        ],
        "records": records,
        "blockers": blockers,
        "workshops": [
            {
                "workshop_id": record.get("workshop_id"),
                "workshop_title": record.get("workshop_title"),
                "status": "succeeded" if record.get("status") == "succeeded" else "blocked_fixture",
                "media_record_count": 1 if record.get("status") == "succeeded" else 0,
                "blocker_count": 0 if record.get("status") == "succeeded" else 1,
                "records": [record],
                "blockers": [] if record.get("status") == "succeeded" else blockers[:1],
            }
            for record in records
        ],
    }
    inventory_path.write_text(
        json.dumps(inventory_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt_path = workshops_dir / "talk_media_acquisition_receipt.json"
    receipt_payload = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.workshop_talk_media_acquisition",
        "workspace": str(workspace),
        "run_id": run_id,
        "status": "succeeded",
        "inventory_path": str(inventory_path.relative_to(workspace)),
        "inventory_sha256": file_sha256(inventory_path),
        "checks": {
            "workshop_count": inventory_payload["workshop_count"],
            "media_record_count": media_record_count,
            "blocked_workshop_count": blocked_workshop_count,
            "blocker_count": len(blockers),
            "covered_workshop_count": inventory_payload["workshop_count"],
            "all_workshops_have_media_or_blocker": True,
        },
        "blockers": blockers,
        "boundaries": {
            "phase4_artifacts_redownloaded": False,
            "live_provider_calls_made": False,
            "secrets_copied": False,
        },
    }
    receipt_path.write_text(
        json.dumps(receipt_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"receipt": receipt_path, "inventory": inventory_path}


def write_corpus_inventory_index_fixture(
    workspace: Path,
    run_id: str,
    *,
    write_transcription_receipt: bool = True,
    transcription_status: str = "succeeded",
    stale_transcription_digest: bool = False,
    main_paper_authors: list[str] | None = None,
) -> dict[str, Path]:
    download_dir = workspace / "storage" / "download-corpus"
    paper_pdf = download_dir / "papers" / "main-paper.pdf"
    workshop_pdf = workspace / "storage" / "workshops" / "papers" / "workshop-paper.pdf"
    audio_path = download_dir / "media" / "talk.mp3"
    normal_slide = (
        workspace
        / "cache"
        / "acquisition-downloads"
        / "media_asset"
        / "poster-1"
        / "slideslive_slides"
        / "images"
        / "slide_001.jpg"
    )
    skipped_slide = normal_slide.with_name("slide_002.jpg")
    text_dir = workspace / "storage" / "transcription-ocr" / "text-extraction"
    main_text = text_dir / "main" / "main-paper.txt"
    workshop_text = text_dir / "workshops" / "workshop-paper.txt"
    transcript_text = workspace / "storage" / "transcription-ocr" / "speech-transcripts" / "talk.txt"
    transcript_json = transcript_text.with_suffix(".json")
    annotated_transcript_json = (
        workspace
        / "storage"
        / "transcription-ocr"
        / "workshop-talks"
        / "talk_annotated.json"
    )

    write_pdf_fixture(paper_pdf, "Main source PDF bytes")
    write_pdf_fixture(workshop_pdf, "Workshop source PDF bytes")
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"fixture audio bytes")
    write_png_fixture(normal_slide)
    write_png_fixture(skipped_slide)
    main_text.parent.mkdir(parents=True, exist_ok=True)
    main_text.write_text(
        "Main Paper Title\nSection A\nThis main paper explains source aware indexing "
        "with reasoning agents, tool use, post-training, reliable evaluation, "
        "and multimodal evidence. " * 8,
        encoding="utf-8",
    )
    workshop_text.parent.mkdir(parents=True, exist_ok=True)
    workshop_text.write_text(
        "Workshop Paper Title\nSubsection B\nWorkshop evidence belongs in the "
        "paper store for agentic systems, safety, alignment, and benchmark "
        "evaluation. " * 8,
        encoding="utf-8",
    )
    transcript_text.parent.mkdir(parents=True, exist_ok=True)
    transcript_text.write_text(
        "The presentation transcript discusses reasoning agents, artifacts, "
        "demos, implementation details, multimodal tool use, and evaluation. " * 8,
        encoding="utf-8",
    )
    transcript_json.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 12.0,
                        "text": (
                            "The presentation transcript discusses reasoning agents, "
                            "artifacts, multimodal tool use, and evaluation. " * 8
                        ),
                    }
                ]
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    annotated_transcript_json.parent.mkdir(parents=True, exist_ok=True)
    annotated_transcript_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "owner_id": "transcription_ocr.workshop_talk_boundary_discovery",
                "segments": [
                    {
                        "start": 0.0,
                        "end": 12.0,
                        "text": (
                            "The presentation transcript discusses reasoning agents, "
                            "artifacts, multimodal tool use, and evaluation. " * 8
                        ),
                        "workshop_id": "workshop_demo",
                        "workshop_title": "Workshop Demo",
                        "talk_id": "talk_001",
                        "talk_title": "Talk Demo",
                        "talk_speaker": "Dr Demo",
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    run_path = workspace / "storage" / "workflow-runs" / run_id
    run_path.mkdir(parents=True, exist_ok=True)
    stage4_receipt = run_path / "stage-04-download-corpus.json"

    def artifact(path: Path, *, family: str, content_type: str, kind: str, **extra: object) -> dict[str, object]:
        payload: dict[str, object] = {
            "source_family_id": family,
            "artifact_kind": kind,
            "status": "succeeded",
            "content_type": content_type,
            "local_path": str(path.relative_to(workspace)),
            "byte_count": path.stat().st_size,
            "sha256": file_sha256(path),
            "storage_lane": "cache/acquisition-downloads",
            "secret_values_recorded": False,
        }
        payload.update(extra)
        return payload

    main_artifact = artifact(
        paper_pdf,
        family="openreview-venue",
        content_type="application/pdf",
        kind="paper_pdf",
        title="Main Paper Title",
        **({"authors": main_paper_authors} if main_paper_authors is not None else {}),
    )
    workshop_artifact = artifact(
        workshop_pdf,
        family="workshops",
        content_type="application/pdf",
        kind="paper_pdf",
        title="Workshop Paper Title",
        storage_lane="storage/workshops/papers",
    )
    audio_artifact = artifact(
        audio_path,
        family="iclr-calendar",
        content_type="audio/mpeg",
        kind="media_asset",
        media_kind="slideslive_audio",
        slideslive_id="39000001",
        source_title="Source Demo",
        session_title="Session Demo",
        workshop_id="workshop_demo",
        workshop_title="Workshop Demo",
        talk_id="talk_001",
        talk_title="Talk Demo",
        talk_speaker="Dr Demo",
    )
    normal_slide_artifact = artifact(
        normal_slide,
        family="iclr-calendar",
        content_type="image/jpeg",
        kind="media_asset",
        media_kind="slideslive_slide_image",
        slideslive_id="39000001",
        slide_index=1,
    )
    skipped_slide_artifact = artifact(
        skipped_slide,
        family="iclr-calendar",
        content_type="image/jpeg",
        kind="media_asset",
        media_kind="slideslive_slide_image",
        slideslive_id="39000001",
        slide_index=2,
        skip_ocr=True,
        skip_reason="reference_or_duplicate_slide",
    )

    download_inventory = download_dir / "download_corpus.json"
    download_inventory.parent.mkdir(parents=True, exist_ok=True)
    download_inventory.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "receipt_type": "conference_pipeline.iclr2026_download_corpus",
                "workspace": str(workspace),
                "run_id": run_id,
                "stage_id": "download-corpus",
                "status": "succeeded",
                "local_artifact_count": 5,
                "direct_download_artifact_count": 5,
                "paper_artifact_count": 2,
                "workshop_paper_pdf_count": 1,
                "media_artifact_count": 3,
                "local_artifacts": [
                    main_artifact,
                    workshop_artifact,
                    audio_artifact,
                    normal_slide_artifact,
                    skipped_slide_artifact,
                ],
                "direct_download_artifacts": [
                    main_artifact,
                    workshop_artifact,
                    audio_artifact,
                    normal_slide_artifact,
                    skipped_slide_artifact,
                ],
                "blockers": [],
                "boundaries": {
                    "live_provider_calls_made": False,
                    "secrets_copied": False,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    stage4_receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "receipt_type": "conference_pipeline.iclr2026_production_stage_receipt",
                "run_id": run_id,
                "workspace": str(workspace),
                "stage_id": "download-corpus",
                "stage_order": 4,
                "mode": "live",
                "status": "succeeded",
                "checks": {
                    "download_corpus_inventory_path": str(download_inventory.relative_to(workspace)),
                    "download_corpus_inventory_sha256": file_sha256(download_inventory),
                    "local_artifact_count": 5,
                    "direct_download_artifact_count": 5,
                    "paper_artifact_count": 2,
                    "workshop_paper_pdf_count": 1,
                    "media_artifact_count": 3,
                    "live_provider_calls_made": False,
                    "secrets_copied": False,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    transcription_inventory = workspace / "storage" / "transcription-ocr" / "transcription_ocr_inventory.json"
    transcription_inventory.parent.mkdir(parents=True, exist_ok=True)
    transcription_payload = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.iclr2026_transcription_ocr_inventory",
        "workspace": str(workspace),
        "run_id": run_id,
        "stage_id": "transcription-ocr",
        "status": "succeeded",
        "lanes": {
            "paper_text_extraction": {
                "records": [
                    {
                        "lane": "paper_text_extraction",
                        "status": "succeeded",
                        "source_path": str(paper_pdf.relative_to(workspace)),
                        "source_sha256": file_sha256(paper_pdf),
                        "source_byte_count": paper_pdf.stat().st_size,
                        "output_path": str(main_text.relative_to(workspace)),
                        "output_sha256": file_sha256(main_text),
                        "output_byte_count": main_text.stat().st_size,
                        "page_count": 1,
                        "title": "Main Paper Title",
                        "section_title": "Methods",
                        "subsection_title": "Corpus Index",
                    }
                ]
            },
            "workshop_paper_text_extraction": {
                "records": [
                    {
                        "lane": "workshop_paper_text_extraction",
                        "status": "succeeded",
                        "source_path": str(workshop_pdf.relative_to(workspace)),
                        "source_sha256": file_sha256(workshop_pdf),
                        "source_byte_count": workshop_pdf.stat().st_size,
                        "output_path": str(workshop_text.relative_to(workspace)),
                        "output_sha256": file_sha256(workshop_text),
                        "output_byte_count": workshop_text.stat().st_size,
                        "page_count": 1,
                        "title": "Workshop Paper Title",
                    }
                ]
            },
            "session_speech_transcription": {
                "records": [
                    {
                        "lane": "session_speech_transcription",
                        "status": "succeeded",
                        "source_path": str(audio_path.relative_to(workspace)),
                        "source_sha256": file_sha256(audio_path),
                        "source_byte_count": audio_path.stat().st_size,
                        "transcript_path": str(transcript_text.relative_to(workspace)),
                        "transcript_sha256": file_sha256(transcript_text),
                        "transcript_byte_count": transcript_text.stat().st_size,
                        "transcript_json_path": str(transcript_json.relative_to(workspace)),
                        "annotated_transcript_json_path": str(
                            annotated_transcript_json.relative_to(workspace)
                        ),
                        "workshop_talk_discovery_receipt_path": (
                            "storage/transcription-ocr/workshop-talks/talk_discovery_receipt.json"
                        ),
                        "workshop_talks_json_path": "storage/transcription-ocr/workshop-talks/talks.json",
                        "provider": "openai",
                        "provider_model": "gpt-4o-transcribe",
                        "provider_calls_made": False,
                        "slideslive_id": "39000001",
                        "source_title": "Source Demo",
                        "session_title": "Session Demo",
                        "workshop_id": "workshop_demo",
                        "workshop_title": "Workshop Demo",
                        "talk_id": "talk_001",
                        "talk_title": "Talk Demo",
                        "talk_speaker": "Dr Demo",
                    }
                ]
            },
            "image_ocr": {
                "records": [
                    {
                        "lane": "image_ocr",
                        "status": "not_run_no_paid_provider_by_default",
                        "source_path": str(normal_slide.relative_to(workspace)),
                        "source_sha256": file_sha256(normal_slide),
                        "source_byte_count": normal_slide.stat().st_size,
                        "provider_calls_made": False,
                        "text_output_count": 0,
                    },
                    {
                        "lane": "image_ocr",
                        "status": "not_run_no_paid_provider_by_default",
                        "source_path": str(skipped_slide.relative_to(workspace)),
                        "source_sha256": file_sha256(skipped_slide),
                        "source_byte_count": skipped_slide.stat().st_size,
                        "provider_calls_made": False,
                        "text_output_count": 0,
                        "skip_ocr": True,
                        "skip_reason": "reference_or_duplicate_slide",
                    },
                ]
            },
            "unsupported_or_blocked": {"records": []},
        },
        "lane_counts": {},
        "blockers": [],
        "output_inventory": {
            "text_output_count": 3,
            "speech_transcript_output_count": 1,
            "image_text_output_count": 0,
        },
        "boundaries": {
            "provider_calls_made": False,
            "live_provider_calls_made": False,
            "secrets_copied": False,
        },
    }
    transcription_inventory.write_text(
        json.dumps(transcription_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    transcription_sha = file_sha256(transcription_inventory)
    if stale_transcription_digest:
        transcription_inventory.write_text(
            json.dumps(
                {**transcription_payload, "generated_at": "changed-after-receipt"},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    stage5_receipt = run_path / "stage-05-transcription-ocr.json"
    if write_transcription_receipt:
        stage5_receipt.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "receipt_type": "conference_pipeline.iclr2026_production_stage_receipt",
                    "run_id": run_id,
                    "workspace": str(workspace),
                    "stage_id": "transcription-ocr",
                    "stage_order": 5,
                    "mode": "live",
                    "status": transcription_status,
                    "checks": {
                        "output_inventory": {
                            "path": str(transcription_inventory.relative_to(workspace)),
                            "sha256": transcription_sha,
                            "status": "succeeded",
                        },
                        "download_corpus_inventory_sha256": file_sha256(download_inventory),
                        "provider_calls_made": False,
                        "live_provider_calls_made": False,
                        "secrets_copied": False,
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    return {
        "stage4_receipt": stage4_receipt,
        "stage5_receipt": stage5_receipt,
        "download_inventory": download_inventory,
        "transcription_inventory": transcription_inventory,
        "normal_slide": normal_slide,
        "skipped_slide": skipped_slide,
    }


def run_source_discovery_and_corpus_freeze(workspace: Path, base_url: str, run_id: str) -> None:
    write_source_inventory(workspace, base_url)
    discovery_result = run_runner(
        [
            "--workspace",
            str(workspace),
            "--run-id",
            run_id,
            "--stage",
            "source-discovery",
            "--mode",
            "live",
            "--no-env-file",
            "--source-timeout-seconds",
            "2",
            "--summary",
        ],
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
    )
    assert discovery_result.returncode == 0, discovery_result.stderr
    freeze_result = run_runner(
        [
            "--workspace",
            str(workspace),
            "--run-id",
            run_id,
            "--stage",
            "corpus-freeze",
            "--mode",
            "live",
            "--no-env-file",
            "--summary",
        ],
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
    )
    assert freeze_result.returncode == 0, freeze_result.stderr


def test_runner_rejects_non_live_mode_without_receipts(tmp_path: Path) -> None:
    workspace = write_workspace(tmp_path)

    result = run_runner(
        [
            "--workspace",
            str(workspace),
            "--run-id",
            "test-non-live",
            "--stage",
            "runner-preflight",
            "--mode",
            "dry-run",
            "--summary",
        ],
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
    )

    assert result.returncode != 0
    assert "live-only" in result.stderr
    assert not (workspace / "storage" / "workflow-runs" / "test-non-live").exists()


def test_runner_preflight_writes_secret_free_success_receipt(tmp_path: Path) -> None:
    workspace = write_workspace(tmp_path)
    env_file = write_env_file(tmp_path)

    result = run_runner(
        [
            "--workspace",
            str(workspace),
            "--run-id",
            "test-preflight",
            "--stage",
            "runner-preflight",
            "--mode",
            "live",
            "--env-file",
            str(env_file),
            "--min-free-gb",
            "0",
            "--summary",
        ],
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    receipt_path = workspace / "storage" / "workflow-runs" / "test-preflight" / "stage-01-runner-preflight.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    serialized = json.dumps({"summary": summary, "receipt": receipt})

    assert summary["status"] == "succeeded"
    assert summary["halt_status"] is None
    assert receipt["status"] == "succeeded"
    assert receipt["checks"]["live_provider_calls_made"] is False
    assert receipt["checks"]["provider_cost_approval"]["approved"] is True
    assert all(item["present"] for item in receipt["checks"]["provider_environment"])
    assert SECRET_VALUE not in serialized
    assert "OPENAI_API_KEY" in serialized
    assert receipt["secrets_copied"] is False


def test_runner_preflight_missing_env_writes_blocker(tmp_path: Path) -> None:
    workspace = write_workspace(tmp_path)

    result = run_runner(
        [
            "--workspace",
            str(workspace),
            "--run-id",
            "test-missing-env",
            "--stage",
            "runner-preflight",
            "--mode",
            "live",
            "--no-env-file",
            "--min-free-gb",
            "0",
            "--summary",
        ],
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
    )

    assert result.returncode == 1
    summary = json.loads(result.stdout)
    receipt_path = workspace / "storage" / "workflow-runs" / "test-missing-env" / "stage-01-runner-preflight.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    blocker_paths = list(
        (workspace / "storage" / "workflow-runs" / "test-missing-env" / "blocker-receipts").rglob("*.json")
    )

    assert summary["status"] == "blocked"
    assert summary["halt_status"] == "halted"
    assert summary["halt_stage_id"] == "runner-preflight"
    assert receipt["status"] == "blocked"
    assert "missing required provider env vars" in receipt["error"]
    assert len(blocker_paths) == 1
    blocker = json.loads(blocker_paths[0].read_text(encoding="utf-8"))
    assert blocker["blocker_class"] == "stage_blocked"
    assert "OPENAI_API_KEY" in blocker["reason"]
    assert SECRET_VALUE not in json.dumps({"summary": summary, "receipt": receipt, "blocker": blocker})


def test_source_discovery_writes_inventory_and_recovers_downloads(tmp_path: Path) -> None:
    workspace = write_workspace(tmp_path)
    server, base_url = start_source_server()
    try:
        write_source_inventory(workspace, base_url)

        result = run_runner(
            [
                "--workspace",
                str(workspace),
                "--run-id",
                "test-source-discovery",
                "--stage",
                "source-discovery",
                "--mode",
                "live",
                "--no-env-file",
                "--source-timeout-seconds",
                "2",
                "--summary",
            ],
            env={"PYTHONPATH": str(REPO_ROOT / "src")},
        )
    finally:
        server.shutdown()

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    receipt_path = workspace / "storage" / "workflow-runs" / "test-source-discovery" / "stage-02-source-discovery.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    discovery_path = workspace / "storage" / "source-discovery" / "source_discovery_inventory.json"
    report_path = workspace / "storage" / "source-discovery" / "source_discovery_report.md"
    discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
    serialized = json.dumps({"summary": summary, "receipt": receipt, "discovery": discovery})

    assert summary["status"] == "succeeded"
    assert summary["halt_status"] is None
    assert receipt["status"] == "succeeded"
    assert discovery["status"] == "succeeded"
    assert discovery["source_family_count"] == 13
    assert discovery["blocked_count"] == 0
    assert report_path.exists()
    assert any(
        recovery["status"] == "canonical-downloads-candidate-succeeded"
        for recovery in discovery["safe_recoveries"]
    )
    downloads = next(item for item in discovery["sources"] if item["id"] == "downloads")
    assert downloads["discovery_status"] == "recovered_to_canonical_candidate"
    assert receipt["checks"]["live_provider_calls_made"] is False
    assert receipt["checks"]["download_heavy_acquisition_made"] is False
    assert SECRET_VALUE not in serialized


def test_source_discovery_unreachable_source_writes_blocker(tmp_path: Path) -> None:
    workspace = write_workspace(tmp_path)
    server, base_url = start_source_server()
    try:
        write_source_inventory(workspace, base_url, missing_papers=True)

        result = run_runner(
            [
                "--workspace",
                str(workspace),
                "--run-id",
                "test-source-discovery-blocked",
                "--stage",
                "source-discovery",
                "--mode",
                "live",
                "--no-env-file",
                "--source-timeout-seconds",
                "2",
                "--summary",
            ],
            env={"PYTHONPATH": str(REPO_ROOT / "src")},
        )
    finally:
        server.shutdown()

    assert result.returncode == 1
    summary = json.loads(result.stdout)
    receipt_path = (
        workspace
        / "storage"
        / "workflow-runs"
        / "test-source-discovery-blocked"
        / "stage-02-source-discovery.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    discovery_path = workspace / "storage" / "source-discovery" / "source_discovery_inventory.json"
    discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
    blocker_paths = list(
        (workspace / "storage" / "workflow-runs" / "test-source-discovery-blocked" / "blocker-receipts").rglob(
            "*.json"
        )
    )

    assert summary["status"] == "blocked"
    assert summary["halt_status"] == "halted"
    assert summary["halt_stage_id"] == "source-discovery"
    assert receipt["status"] == "blocked"
    assert discovery["status"] == "blocked"
    assert discovery["blocked_count"] == 1
    assert discovery["blockers"][0]["family_id"] == "accepted-papers"
    assert len(blocker_paths) == 1
    blocker = json.loads(blocker_paths[0].read_text(encoding="utf-8"))
    assert blocker["blocker_class"] == "stage_blocked"
    assert "accepted-papers" in blocker["reason"]


def test_corpus_freeze_includes_all_discovered_sources_without_exclusions(tmp_path: Path) -> None:
    workspace = write_workspace(tmp_path)
    server, base_url = start_source_server()
    try:
        write_source_inventory(workspace, base_url)
        discovery_result = run_runner(
            [
                "--workspace",
                str(workspace),
                "--run-id",
                "test-corpus-freeze",
                "--stage",
                "source-discovery",
                "--mode",
                "live",
                "--no-env-file",
                "--source-timeout-seconds",
                "2",
                "--summary",
            ],
            env={"PYTHONPATH": str(REPO_ROOT / "src")},
        )
        assert discovery_result.returncode == 0, discovery_result.stderr

        result = run_runner(
            [
                "--workspace",
                str(workspace),
                "--run-id",
                "test-corpus-freeze",
                "--stage",
                "corpus-freeze",
                "--mode",
                "live",
                "--no-env-file",
                "--summary",
            ],
            env={"PYTHONPATH": str(REPO_ROOT / "src")},
        )
    finally:
        server.shutdown()

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    receipt_path = workspace / "storage" / "workflow-runs" / "test-corpus-freeze" / "stage-03-corpus-freeze.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    selection_path = workspace / "storage" / "corpus-freeze" / "corpus_freeze.json"
    report_path = workspace / "storage" / "corpus-freeze" / "corpus_freeze_report.md"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    serialized = json.dumps({"summary": summary, "receipt": receipt, "selection": selection})

    assert summary["status"] == "succeeded"
    assert receipt["status"] == "succeeded"
    assert selection["status"] == "succeeded"
    assert selection["included_count"] == 13
    assert selection["excluded_count"] == 0
    assert report_path.exists()
    assert selection["command_policy"]["silent_item_caps_present"] is False
    assert all(record["planned_acquisition"]["planned_stages"] for record in selection["included_records"])
    assert receipt["checks"]["all_in_scope_items_have_planned_acquisition_or_unreachable_path"] is True
    assert receipt["checks"]["download_heavy_acquisition_made"] is False
    assert "--max-papers" not in json.dumps(selection["command_policy"]["checked_commands"])
    assert SECRET_VALUE not in serialized


def test_corpus_freeze_missing_source_discovery_writes_blocker(tmp_path: Path) -> None:
    workspace = write_workspace(tmp_path)

    result = run_runner(
        [
            "--workspace",
            str(workspace),
            "--run-id",
            "test-corpus-freeze-missing-discovery",
            "--stage",
            "corpus-freeze",
            "--mode",
            "live",
            "--no-env-file",
            "--summary",
        ],
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
    )

    assert result.returncode == 1
    summary = json.loads(result.stdout)
    receipt_path = (
        workspace
        / "storage"
        / "workflow-runs"
        / "test-corpus-freeze-missing-discovery"
        / "stage-03-corpus-freeze.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    blocker_paths = list(
        (
            workspace
            / "storage"
            / "workflow-runs"
            / "test-corpus-freeze-missing-discovery"
            / "blocker-receipts"
        ).rglob("*.json")
    )

    assert summary["status"] == "blocked"
    assert summary["halt_stage_id"] == "corpus-freeze"
    assert receipt["status"] == "blocked"
    assert len(blocker_paths) == 1
    blocker = json.loads(blocker_paths[0].read_text(encoding="utf-8"))
    assert blocker["blocker_class"] == "missing_receipt"
    assert "source-discovery inventory" in blocker["reason"]


def test_download_corpus_writes_public_artifact_inventory(tmp_path: Path) -> None:
    workspace = write_workspace(tmp_path)
    cookie_file = tmp_path / "iclr-cookie.txt"
    cookie_file.write_text(
        f"sessionid={SECRET_VALUE}-iclr-session\ncsrftoken={SECRET_VALUE}-csrf\n",
        encoding="utf-8",
    )
    server, base_url = start_source_server()
    try:
        run_source_discovery_and_corpus_freeze(workspace, base_url, "test-download-corpus")

        result = run_runner(
            [
                "--workspace",
                str(workspace),
                "--run-id",
                "test-download-corpus",
                "--stage",
                "download-corpus",
                "--mode",
                "live",
                "--resume",
                "--no-env-file",
                "--download-timeout-seconds",
                "2",
                "--skip-workshop-acquisition",
                "--iclr-cookie-file",
                str(cookie_file),
                "--summary",
            ],
            env={"PYTHONPATH": str(REPO_ROOT / "src")},
        )
    finally:
        server.shutdown()

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    receipt_path = workspace / "storage" / "workflow-runs" / "test-download-corpus" / "stage-04-download-corpus.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    inventory_path = workspace / "storage" / "download-corpus" / "download_corpus.json"
    report_path = workspace / "storage" / "download-corpus" / "download_corpus_report.md"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    serialized = json.dumps({"summary": summary, "receipt": receipt, "inventory": inventory})

    assert summary["status"] == "succeeded"
    assert receipt["status"] == "succeeded"
    assert inventory["status"] == "succeeded"
    assert inventory["source_family_count"] == 13
    assert inventory["media_artifact_count"] >= 1
    assert inventory["paper_artifact_count"] >= 1
    assert inventory["workshop_paper_pdf_count"] == 0
    assert inventory["workshop_acquisition"]["status"] == "not_run"
    assert inventory["direct_download_artifact_count"] >= 2
    assert inventory["item_page_expansion"]["target_count"] >= 1
    assert inventory["item_page_expansion"]["complete"] is True
    assert inventory["blockers"] == []
    assert inventory["iclr_cookie_file"]["loaded"] is True
    assert inventory["iclr_cookie_file"]["cookie_names"] == ["csrftoken", "sessionid"]
    assert report_path.exists()
    assert all(
        artifact.get("sha256")
        for artifact in inventory["local_artifacts"]
        if artifact.get("status") == "succeeded"
    )
    assert receipt["checks"]["live_provider_calls_made"] is False
    assert SECRET_VALUE not in serialized


def test_download_corpus_missing_corpus_freeze_writes_blocker(tmp_path: Path) -> None:
    workspace = write_workspace(tmp_path)

    result = run_runner(
        [
            "--workspace",
            str(workspace),
            "--run-id",
            "test-download-corpus-missing-freeze",
            "--stage",
            "download-corpus",
            "--mode",
            "live",
            "--no-env-file",
            "--summary",
        ],
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
    )

    assert result.returncode == 1
    summary = json.loads(result.stdout)
    receipt_path = (
        workspace
        / "storage"
        / "workflow-runs"
        / "test-download-corpus-missing-freeze"
        / "stage-04-download-corpus.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    blocker_paths = list(
        (
            workspace
            / "storage"
            / "workflow-runs"
            / "test-download-corpus-missing-freeze"
            / "blocker-receipts"
        ).rglob("*.json")
    )

    assert summary["status"] == "blocked"
    assert summary["halt_stage_id"] == "download-corpus"
    assert receipt["status"] == "blocked"
    assert len(blocker_paths) == 1
    blocker = json.loads(blocker_paths[0].read_text(encoding="utf-8"))
    assert blocker["blocker_class"] == "missing_receipt"
    assert "corpus-freeze selection" in blocker["reason"]


def test_workshop_talk_media_owner_requires_workshop_download_receipt(tmp_path: Path) -> None:
    workspace = write_workspace(tmp_path)
    workshops_dir = workspace / "storage" / "workshops"
    workshops_dir.mkdir(parents=True, exist_ok=True)
    (workshops_dir / "inventory.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "workshop_count": 1,
                "workshops": [{"venueid": "ICLR.cc/2026/Workshop/DEMO"}],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_workshop_talk_media_owner(
        [
            "--workspace",
            str(workspace),
            "--run-id",
            "test-workshop-talk-media-missing-receipt",
            "--resume",
            "--summary",
        ],
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
    )

    assert result.returncode == 1
    receipt_path = workshops_dir / "talk_media_acquisition_receipt.json"
    inventory_path = workshops_dir / "talk_media_inventory.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))

    assert receipt["status"] == "blocked"
    assert receipt["checks"]["download_receipt_present"] is False
    assert receipt["blockers"][0]["code"] == "missing_workshop_download_receipt"
    assert inventory["status"] == "blocked"
    assert inventory["blockers"][0]["code"] == "missing_workshop_download_receipt"


def test_workshop_talk_media_owner_downloads_media_with_required_provenance(
    tmp_path: Path,
) -> None:
    workspace = write_workspace(tmp_path)
    write_workshop_media_inventory_fixture(workspace, ["ICLR.cc/2026/Workshop/DEMO"])
    receipt_before = file_sha256(workspace / "storage" / "workshops" / "download_receipt.json")
    media_bytes = b"fixture workshop talk audio bytes"
    server, base_url = start_workshop_talk_media_server(
        pages={
            "/virtual/2026/events/workshop": (
                "<html><body><div class='event-card' data-event-id='42' "
                "data-event-title='fixture workshop demo'>"
                "<h3 class='event-title'><a href='/virtual/2026/workshop/42'>"
                "Fixture Workshop (DEMO)</a></h3></div></body></html>"
            ),
            "/virtual/2026/workshop/42": (
                "<html><body><h1>Fixture Workshop (DEMO)</h1>"
                "<a href='/site/demo'>Project Page</a>"
                f"<a href='/media/demo-talk.mp3?token={SECRET_VALUE}&v=demo'>"
                "Opening invited talk recording</a>"
                "</body></html>"
            ),
            "/site/demo": "<html><body><p>agenda without extra media</p></body></html>",
        },
        binaries={"/media/demo-talk.mp3": ("audio/mpeg", media_bytes)},
    )
    try:
        result = run_workshop_talk_media_owner(
            [
                "--workspace",
                str(workspace),
                "--run-id",
                "test-workshop-talk-media-success",
                "--resume",
                "--summary",
                "--workshop-list-url",
                f"{base_url}/virtual/2026/events/workshop",
            ],
            env={"PYTHONPATH": str(REPO_ROOT / "src")},
        )
    finally:
        server.shutdown()

    assert result.returncode == 0, result.stderr
    receipt_path = workspace / "storage" / "workshops" / "talk_media_acquisition_receipt.json"
    inventory_path = workspace / "storage" / "workshops" / "talk_media_inventory.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    record = inventory["records"][0]

    assert receipt["status"] == "succeeded"
    assert receipt["checks"]["workshop_count"] == 1
    assert receipt["checks"]["covered_workshop_count"] == 1
    assert receipt["checks"]["media_record_count"] == 1
    assert receipt["boundaries"]["phase4_artifacts_redownloaded"] is False
    assert receipt["boundaries"]["live_provider_calls_made"] is False
    assert receipt["boundaries"]["secrets_copied"] is False
    assert file_sha256(workspace / "storage" / "workshops" / "download_receipt.json") == receipt_before

    for field in (
        "workshop_id",
        "workshop_title",
        "talk_id",
        "talk_title",
        "talk_speaker",
        "source_url",
        "media_url",
        "local_path",
        "sha256",
        "byte_count",
        "status",
    ):
        assert field in record
    assert record["status"] == "succeeded"
    assert record["workshop_id"] == "ICLR.cc/2026/Workshop/DEMO"
    assert record["talk_title"] == "Opening invited talk recording"
    raw_media_url = f"{base_url}/media/demo-talk.mp3?token={SECRET_VALUE}&v=demo"
    assert record["media_url"] == f"{base_url}/media/demo-talk.mp3?v=demo"
    assert record["download_url_sha256"] == hashlib.sha256(raw_media_url.encode("utf-8")).hexdigest()
    assert record["byte_count"] == len(media_bytes)
    assert record["sha256"] == hashlib.sha256(media_bytes).hexdigest()
    assert (workspace / record["local_path"]).read_bytes() == media_bytes
    assert inventory["workshops"][0]["status"] == "succeeded"
    assert inventory["workshops"][0]["blockers"] == []
    assert SECRET_VALUE not in json.dumps({"receipt": receipt, "inventory": inventory})


def test_workshop_talk_media_owner_records_per_workshop_blockers(tmp_path: Path) -> None:
    workspace = write_workspace(tmp_path)
    write_workshop_media_inventory_fixture(
        workspace,
        [
            "ICLR.cc/2026/Workshop/BLOCKED",
            "ICLR.cc/2026/Workshop/MISSING",
        ],
    )
    server, base_url = start_workshop_talk_media_server(
        pages={
            "/virtual/2026/events/workshop": (
                "<html><body><div class='event-card' data-event-id='77' "
                "data-event-title='blocked workshop'>"
                "<h3 class='event-title'><a href='/virtual/2026/workshop/77'>"
                "Blocked Workshop (BLOCKED)</a></h3></div></body></html>"
            ),
            "/virtual/2026/workshop/77": (
                "<html><body><p>Log in and register to view live content</p></body></html>"
            ),
        }
    )
    try:
        result = run_workshop_talk_media_owner(
            [
                "--workspace",
                str(workspace),
                "--run-id",
                "test-workshop-talk-media-blockers",
                "--resume",
                "--summary",
                "--workshop-list-url",
                f"{base_url}/virtual/2026/events/workshop",
            ],
            env={"PYTHONPATH": str(REPO_ROOT / "src")},
        )
    finally:
        server.shutdown()

    assert result.returncode == 0, result.stderr
    receipt = json.loads(
        (workspace / "storage" / "workshops" / "talk_media_acquisition_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    inventory = json.loads(
        (workspace / "storage" / "workshops" / "talk_media_inventory.json").read_text(
            encoding="utf-8"
        )
    )

    assert receipt["status"] == "succeeded"
    assert receipt["checks"]["workshop_count"] == 2
    assert receipt["checks"]["covered_workshop_count"] == 2
    assert receipt["checks"]["blocked_workshop_count"] == 2
    blocker_codes = {(blocker["workshop_id"], blocker["code"]) for blocker in inventory["blockers"]}
    assert ("ICLR.cc/2026/Workshop/BLOCKED", "auth_wall") in blocker_codes
    assert ("ICLR.cc/2026/Workshop/MISSING", "no_site_url") in blocker_codes
    for workshop in inventory["workshops"]:
        assert workshop["status"].startswith("blocked")
        assert workshop["blockers"]
    for record in inventory["records"]:
        for field in (
            "workshop_id",
            "workshop_title",
            "talk_id",
            "talk_title",
            "talk_speaker",
            "source_url",
            "media_url",
            "local_path",
            "sha256",
            "byte_count",
            "status",
        ):
            assert field in record


def test_workshop_talk_media_owner_sanitizes_direct_media_blocker_urls(tmp_path: Path) -> None:
    workspace = write_workspace(tmp_path)
    write_workshop_media_inventory_fixture(workspace, ["ICLR.cc/2026/Workshop/DEMO"])
    server, base_url = start_workshop_talk_media_server(
        pages={
            "/virtual/2026/events/workshop": (
                "<html><body><div class='event-card' data-event-id='42' "
                "data-event-title='fixture workshop demo'>"
                "<h3 class='event-title'><a href='/virtual/2026/workshop/42'>"
                "Fixture Workshop (DEMO)</a></h3></div></body></html>"
            ),
            "/virtual/2026/workshop/42": (
                f"<html><body><a href='/media/private-talk.mp3?token={SECRET_VALUE}&v=demo'>"
                "Private talk recording</a></body></html>"
            ),
            "/media/private-talk.mp3": "<html><body>authentication required</body></html>",
        }
    )
    try:
        result = run_workshop_talk_media_owner(
            [
                "--workspace",
                str(workspace),
                "--run-id",
                "test-workshop-talk-media-sanitized-blocker",
                "--resume",
                "--summary",
                "--workshop-list-url",
                f"{base_url}/virtual/2026/events/workshop",
            ],
            env={"PYTHONPATH": str(REPO_ROOT / "src")},
        )
    finally:
        server.shutdown()

    assert result.returncode == 0, result.stderr
    inventory = json.loads(
        (workspace / "storage" / "workshops" / "talk_media_inventory.json").read_text(
            encoding="utf-8"
        )
    )
    raw_media_url = f"{base_url}/media/private-talk.mp3?token={SECRET_VALUE}&v=demo"
    blocker = inventory["blockers"][0]
    record = inventory["records"][0]

    assert blocker["code"] == "download_error"
    assert blocker["media_url"] == f"{base_url}/media/private-talk.mp3?v=demo"
    assert blocker["download_url_sha256"] == hashlib.sha256(raw_media_url.encode("utf-8")).hexdigest()
    assert record["media_url"] == blocker["media_url"]
    assert record["download_url_sha256"] == blocker["download_url_sha256"]
    assert SECRET_VALUE not in json.dumps(inventory)


def test_workshop_talk_media_owner_loads_cookie_file_without_recording_secrets(
    tmp_path: Path,
) -> None:
    module = load_workshop_talk_media_owner_module()
    cookie_file = tmp_path / "iclr-cookie.txt"
    cookie_file.write_text(
        f"sessionid={SECRET_VALUE}-iclr-session\ncsrftoken={SECRET_VALUE}-csrf\n",
        encoding="utf-8",
    )

    cookie_header, status = module.load_cookie_file(cookie_file)

    assert cookie_header is not None
    assert status == {
        "attempted": True,
        "exists": True,
        "loaded": True,
        "cookie_names": ["csrftoken", "sessionid"],
        "secret_values_recorded": False,
        "path": str(cookie_file),
    }
    assert module.cookie_headers("https://iclr.cc/virtual/2026/workshop/42", cookie_header) == {
        "Cookie": cookie_header
    }
    assert module.cookie_headers("https://example.com/workshop", cookie_header) == {}
    assert SECRET_VALUE not in json.dumps(status)


def test_workshop_talk_media_owner_retries_authenticated_after_public_blocker(
    tmp_path: Path,
) -> None:
    workspace = write_workspace(tmp_path)
    write_workshop_media_inventory_fixture(workspace, ["ICLR.cc/2026/Workshop/DEMO"])
    cookie_file = tmp_path / "iclr-cookie.txt"
    cookie_file.write_text("sessionid=fixture-session\n", encoding="utf-8")
    media_bytes = b"authenticated workshop talk bytes"
    server, base_url = start_workshop_talk_media_server(
        pages={
            "/virtual/2026/events/workshop": (
                "<html><body><div class='event-card' data-event-id='42' "
                "data-event-title='fixture workshop demo'>"
                "<h3 class='event-title'><a href='/virtual/2026/workshop/42'>"
                "Fixture Workshop (DEMO)</a></h3></div></body></html>"
            ),
            "/virtual/2026/workshop/42": (
                "<html><body><p>Log in and register to view live content</p></body></html>"
            ),
        },
        authenticated_pages={
            "/virtual/2026/workshop/42": (
                "<html><body><a href='/media/auth-demo-talk.mp3'>"
                "Authenticated invited talk recording</a></body></html>"
            ),
        },
        binaries={"/media/auth-demo-talk.mp3": ("audio/mpeg", media_bytes)},
        required_cookie="sessionid=fixture-session",
    )
    try:
        result = run_workshop_talk_media_owner(
            [
                "--workspace",
                str(workspace),
                "--run-id",
                "test-workshop-talk-media-auth-fallback",
                "--resume",
                "--summary",
                "--workshop-list-url",
                f"{base_url}/virtual/2026/events/workshop",
                "--iclr-cookie-file",
                str(cookie_file),
            ],
            env={"PYTHONPATH": str(REPO_ROOT / "src")},
        )
    finally:
        server.shutdown()

    assert result.returncode == 0, result.stderr
    inventory = json.loads(
        (workspace / "storage" / "workshops" / "talk_media_inventory.json").read_text(
            encoding="utf-8"
        )
    )
    receipt = json.loads(
        (workspace / "storage" / "workshops" / "talk_media_acquisition_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    record = inventory["records"][0]

    assert receipt["checks"]["iclr_cookie_file"]["loaded"] is True
    assert record["status"] == "succeeded"
    assert record["talk_title"] == "Authenticated invited talk recording"
    assert record["access_mode"] == "authenticated"
    assert record["access_attempts"] == ["public", "authenticated"]
    assert inventory["workshops"][0]["access_mode"] == "authenticated"
    assert SECRET_VALUE not in json.dumps({"receipt": receipt, "inventory": inventory})


def test_workshop_talk_media_owner_recovers_embedded_audio_with_ytdlp(
    tmp_path: Path,
) -> None:
    workspace = write_workspace(tmp_path)
    write_workshop_media_inventory_fixture(workspace, ["ICLR.cc/2026/Workshop/DEMO"])
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_ytdlp = fake_bin / "yt-dlp"
    ytdlp_log = tmp_path / "yt-dlp-urls.txt"
    fake_ytdlp.write_text(
        f"""#!/usr/bin/env python3
import pathlib
import sys

args = sys.argv[1:]
template = args[args.index("--output") + 1]
path = pathlib.Path(template.replace("%(ext)s", "mp3"))
path.parent.mkdir(parents=True, exist_ok=True)
path.write_bytes(b"fake embedded workshop audio")
with pathlib.Path({str(ytdlp_log)!r}).open("a", encoding="utf-8") as handle:
    handle.write(args[-1] + "\\n")
""",
        encoding="utf-8",
    )
    fake_ytdlp.chmod(0o755)
    server, base_url = start_workshop_talk_media_server(
        pages={
            "/virtual/2026/events/workshop": (
                "<html><body><div class='event-card' data-event-id='42' "
                "data-event-title='fixture workshop demo'>"
                "<h3 class='event-title'><a href='/virtual/2026/workshop/42'>"
                "Fixture Workshop (DEMO)</a></h3></div></body></html>"
            ),
            "/virtual/2026/workshop/42": (
                "<html><body><script src='https://slideslive.com/embed_presentation.js'></script><script>"
                "new SlidesLiveEmbed('presentation-embed-39056835', {presentationId: '39056835'});"
                "</script><a href='https://www.youtube.com/squarespace/'>Social</a></body></html>"
            ),
        }
    )
    env = {
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
    }
    try:
        result = run_workshop_talk_media_owner(
            [
                "--workspace",
                str(workspace),
                "--run-id",
                "test-workshop-talk-media-embedded-ytdlp",
                "--resume",
                "--summary",
                "--workshop-list-url",
                f"{base_url}/virtual/2026/events/workshop",
                "--skip-embedded-slide-recovery",
                "--workshop-worker-count",
                "2",
            ],
            env=env,
        )
    finally:
        server.shutdown()

    assert result.returncode == 0, result.stderr
    receipt = json.loads(
        (workspace / "storage" / "workshops" / "talk_media_acquisition_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    inventory = json.loads(
        (workspace / "storage" / "workshops" / "talk_media_inventory.json").read_text(
            encoding="utf-8"
        )
    )
    record = next(item for item in inventory["records"] if item["status"] == "succeeded")

    assert receipt["checks"]["media_record_count"] == 1
    assert receipt["checks"]["embedded_audio_recovery"]["available"] is True
    assert record["media_url"] == "https://slideslive.com/39056835"
    assert record["retrieval_status"] == "embedded_audio_recovered"
    assert record["local_path"].endswith(".mp3")
    assert (workspace / record["local_path"]).read_bytes() == b"fake embedded workshop audio"
    assert ytdlp_log.read_text(encoding="utf-8").splitlines() == ["https://slideslive.com/39056835"]


def test_workshop_talk_media_owner_parallelizes_embedded_audio_by_workshop(
    tmp_path: Path,
) -> None:
    workspace = write_workspace(tmp_path)
    write_workshop_media_inventory_fixture(
        workspace,
        [
            "ICLR.cc/2026/Workshop/AFA",
            "ICLR.cc/2026/Workshop/AFAA",
        ],
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_ytdlp = fake_bin / "yt-dlp"
    fake_ytdlp.write_text(
        """#!/usr/bin/env python3
import pathlib
import sys

args = sys.argv[1:]
template = args[args.index("--output") + 1]
path = pathlib.Path(template.replace("%(ext)s", "mp3"))
path.parent.mkdir(parents=True, exist_ok=True)
path.write_bytes(("fake audio for " + path.parent.parent.name).encode())
""",
        encoding="utf-8",
    )
    fake_ytdlp.chmod(0o755)
    server, base_url = start_workshop_talk_media_server(
        pages={
            "/virtual/2026/events/workshop": (
                "<html><body>"
                "<div class='event-card' data-event-id='1' data-event-title='advances in financial ai'>"
                "<h3 class='event-title'><a href='/virtual/2026/workshop/afa'>"
                "Advances in Financial AI (AFA)</a></h3></div>"
                "<div class='event-card' data-event-id='2' "
                "data-event-title='algorithmic fairness across alignment'>"
                "<h3 class='event-title'><a href='/virtual/2026/workshop/afaa'>"
                "Algorithmic Fairness Across Alignment (AFAA)</a></h3></div>"
                "</body></html>"
            ),
            "/virtual/2026/workshop/afa": (
                "<html><body><script>"
                "new SlidesLiveEmbed('presentation-embed-39056835', {presentationId: '39056835'});"
                "</script></body></html>"
            ),
            "/virtual/2026/workshop/afaa": (
                "<html><body><script>"
                "new SlidesLiveEmbed('presentation-embed-39056836', {presentationId: '39056836'});"
                "</script></body></html>"
            ),
        }
    )
    env = {
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
    }
    try:
        result = run_workshop_talk_media_owner(
            [
                "--workspace",
                str(workspace),
                "--run-id",
                "test-workshop-talk-media-parallel-audio",
                "--resume",
                "--summary",
                "--workshop-list-url",
                f"{base_url}/virtual/2026/events/workshop",
                "--skip-embedded-slide-recovery",
                "--workshop-worker-count",
                "2",
            ],
            env=env,
        )
    finally:
        server.shutdown()

    assert result.returncode == 0, result.stderr
    receipt = json.loads(
        (workspace / "storage" / "workshops" / "talk_media_acquisition_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    inventory = json.loads(
        (workspace / "storage" / "workshops" / "talk_media_inventory.json").read_text(
            encoding="utf-8"
        )
    )
    records = [item for item in inventory["records"] if item.get("status") == "succeeded"]

    assert receipt["checks"]["media_record_count"] == 2
    assert receipt["checks"]["embedded_audio_recovery"]["workshop_worker_count"] == 2
    assert {record["workshop_id"] for record in records} == {
        "ICLR.cc/2026/Workshop/AFA",
        "ICLR.cc/2026/Workshop/AFAA",
    }
    assert {record["media_url"] for record in records} == {
        "https://slideslive.com/39056835",
        "https://slideslive.com/39056836",
    }
    for record in records:
        assert record["local_path"].endswith(".mp3")
        assert (workspace / record["local_path"]).exists()


def test_workshop_talk_media_owner_recovers_workshop_slides_with_provenance(
    tmp_path: Path,
) -> None:
    workspace = write_workspace(tmp_path)
    module = load_workshop_talk_media_owner_module()
    slide_path = (
        workspace
        / "cache"
        / "acquisition-downloads"
        / "media_asset"
        / "workshop-DEMO-39056835"
        / "slideslive_slides"
        / "images"
        / "slide_001.jpg"
    )
    write_png_fixture(slide_path)
    (workspace / "storage" / "download-corpus").mkdir(parents=True, exist_ok=True)
    (workspace / "storage" / "download-corpus" / "download_corpus.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "direct_download_artifacts": [],
                "local_artifacts": [],
                "direct_download_artifact_count": 0,
                "local_artifact_count": 0,
                "media_artifact_count": 0,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    artifact = {
        "source_family_id": "workshop-talk-media:DEMO",
        "artifact_kind": "media_asset",
        "media_kind": "slideslive_slide_image",
        "source_url": "https://slides.example/slide_001.jpg",
        "final_url": "https://slides.example/slide_001.jpg",
        "local_path": str(slide_path.relative_to(workspace)),
        "storage_lane": "cache/acquisition-downloads",
        "retrieval_status": "downloaded",
        "status": "succeeded",
        "content_type": "image/jpeg",
        "byte_count": slide_path.stat().st_size,
        "sha256": file_sha256(slide_path),
        "slideslive_id": "39056835",
        "slide_index": 1,
        "item_id": "workshop-DEMO-39056835",
        "item_kind": "workshop",
        "item_page_url": "https://iclr.cc/virtual/2026/workshop/42",
        "source_artifact_path": "https://iclr.cc/virtual/2026/workshop/42",
    }

    def fake_recoverer(**kwargs):
        assert kwargs["assets"][0].slideslive_id == "39056835"
        return {
            "status": "succeeded",
            "attempted": True,
            "media_records": [artifact],
            "blockers": [],
        }

    records, blockers, summary, corpus_artifacts = module.recover_workshop_slide_images(
        workspace=workspace,
        run_id="test-workshop-slide-recovery",
        workshop_id="ICLR.cc/2026/Workshop/DEMO",
        workshop_title="Fixture Workshop (DEMO)",
        embedded_candidates=[
            (
                "https://iclr.cc/virtual/2026/workshop/42",
                module.Link(
                    url="https://slideslive.com/39056835?embed_init_token=secret",
                    text="Workshop keynote",
                    tag="a",
                ),
            )
        ],
        resume=True,
        worker_count=1,
        height=1080,
        image_format="jpeg",
        use_ytdlp_fallback=True,
        recoverer=fake_recoverer,
    )

    assert blockers == []
    assert summary["download_corpus_merge"]["status"] == "succeeded"
    assert len(records) == 1
    assert len(corpus_artifacts) == 1
    record = records[0]
    assert record["media_kind"] == "slideslive_slide_image"
    assert record["workshop_id"] == "ICLR.cc/2026/Workshop/DEMO"
    assert record["talk_title"] == "Workshop keynote"
    assert record["media_url"] == "https://slideslive.com/39056835"
    assert record["slide_image_url"] == "https://slides.example/slide_001.jpg"

    download_corpus = json.loads(
        (workspace / "storage" / "download-corpus" / "download_corpus.json").read_text(
            encoding="utf-8"
        )
    )
    merged = download_corpus["direct_download_artifacts"][0]
    assert merged["media_kind"] == "slideslive_slide_image"
    assert merged["workshop_id"] == "ICLR.cc/2026/Workshop/DEMO"
    assert merged["workshop_title"] == "Fixture Workshop (DEMO)"
    assert merged["talk_title"] == "Workshop keynote"
    assert merged["media_url"] == "https://slideslive.com/39056835"


def test_workshop_talk_media_auth_retry_is_not_satisfied_by_slides_only() -> None:
    module = load_workshop_talk_media_owner_module()
    slide_only = module.WorkshopResult(
        workshop={},
        event=None,
        records=[{"status": "succeeded", "media_kind": "slideslive_slide_image"}],
    )
    with_audio = module.WorkshopResult(
        workshop={},
        event=None,
        records=[{"status": "succeeded", "media_kind": "embedded_audio"}],
    )

    assert module.should_retry_with_auth(slide_only) is True
    assert module.should_retry_with_auth(with_audio) is False


def test_embedded_media_host_allowlist_rejects_spoofed_domains() -> None:
    from conference_pipeline.acquisition.embedded_media import (
        embedded_media_links,
        is_embedded_media_host,
        is_recoverable_embedded_media_url,
    )

    assert is_embedded_media_host("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert is_embedded_media_host("https://slideslive.com/39056835")
    assert not is_embedded_media_host("https://youtube.com.evil.example/watch?v=demo")
    assert is_recoverable_embedded_media_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert is_recoverable_embedded_media_url("https://youtu.be/dQw4w9WgXcQ")
    assert not is_recoverable_embedded_media_url("https://slideslive.com.attacker.invalid/39056835")
    assert not is_recoverable_embedded_media_url("https://www.youtube.com/squarespace/")
    assert not is_recoverable_embedded_media_url("https://www.youtube.com/@squarespace")
    assert not is_recoverable_embedded_media_url("https://vimeo.com/channels/staffpicks")
    assert embedded_media_links(
        "<script>window.media='https://youtube.com.evil.example/watch?v=demo'</script>",
        base_url="https://iclr.cc/",
    ) == []
    assert embedded_media_links(
        "<script>window.social='https://www.youtube.com/squarespace/'</script>",
        base_url="https://iclr.cc/",
    ) == []


def test_transcription_ocr_requires_workshop_talk_media_receipt_when_workshop_papers_exist(
    tmp_path: Path,
) -> None:
    workspace = write_workspace(tmp_path)
    write_transcription_ocr_download_fixture(
        workspace,
        "test-transcription-ocr-missing-workshop-talk-media",
    )

    result = run_runner(
        [
            "--workspace",
            str(workspace),
            "--run-id",
            "test-transcription-ocr-missing-workshop-talk-media",
            "--stage",
            "transcription-ocr",
            "--mode",
            "live",
            "--resume",
            "--no-env-file",
            "--summary",
        ],
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
    )

    assert result.returncode == 1
    summary = json.loads(result.stdout)
    receipt_path = (
        workspace
        / "storage"
        / "workflow-runs"
        / "test-transcription-ocr-missing-workshop-talk-media"
        / "stage-05-transcription-ocr.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    inventory_path = workspace / "storage" / "transcription-ocr" / "transcription_ocr_inventory.json"

    assert summary["status"] == "blocked"
    assert receipt["status"] == "blocked"
    assert receipt["checks"]["workshop_talk_media_required"] is True
    assert receipt["checks"]["talk_media_acquisition_receipt_present"] is False
    assert receipt["checks"]["talk_media_inventory_present"] is False
    assert receipt["checks"]["blockers"][0]["code"] == "workshop_talk_media_acquisition_required"
    assert not inventory_path.exists()


def test_transcription_ocr_extracts_pdfs_and_records_media_lanes(tmp_path: Path) -> None:
    workspace = write_workspace(tmp_path)
    paths = write_transcription_ocr_download_fixture(workspace, "test-transcription-ocr")
    talk_media_paths = write_workshop_talk_media_scope_fixture(
        workspace,
        "test-transcription-ocr",
    )

    result = run_runner(
        [
            "--workspace",
            str(workspace),
            "--run-id",
            "test-transcription-ocr",
            "--stage",
            "transcription-ocr",
            "--mode",
            "live",
            "--resume",
            "--no-env-file",
            "--summary",
        ],
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
    )

    assert result.returncode == 1, result.stderr
    summary = json.loads(result.stdout)
    receipt_path = workspace / "storage" / "workflow-runs" / "test-transcription-ocr" / "stage-05-transcription-ocr.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    inventory_path = workspace / "storage" / "transcription-ocr" / "transcription_ocr_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    blocker_paths = list(
        (workspace / "storage" / "workflow-runs" / "test-transcription-ocr" / "blocker-receipts").rglob("*.json")
    )
    serialized = json.dumps({"summary": summary, "receipt": receipt, "inventory": inventory}, sort_keys=True)

    assert summary["status"] == "blocked"
    assert summary["halt_stage_id"] == "transcription-ocr"
    assert receipt["stage_id"] == "transcription-ocr"
    assert receipt["status"] == "blocked"
    assert receipt["required_script_ids"] == ["transcription_ocr.local_media_processing"]
    assert "conference_pipeline.transcription_ocr" in receipt["stage_script_owner_refs"]
    assert isinstance(receipt["script_registry_sha256"], str)
    assert receipt["checks"]["download_corpus_stage_receipt_path"].endswith("stage-04-download-corpus.json")
    assert receipt["checks"]["download_corpus_stage_receipt_sha256"] == file_sha256(paths["stage_receipt"])
    assert receipt["checks"]["download_corpus_inventory_sha256"] == file_sha256(paths["inventory"])
    assert receipt["checks"]["talk_media_acquisition_receipt_sha256"] == file_sha256(
        talk_media_paths["receipt"]
    )
    assert receipt["checks"]["talk_media_inventory_sha256"] == file_sha256(
        talk_media_paths["inventory"]
    )
    assert receipt["checks"]["talk_media_inventory_blocker_count"] == 1
    assert receipt["checks"]["download_corpus_counts"]["paper_artifact_count"] == 2
    assert receipt["checks"]["download_corpus_counts"]["workshop_paper_pdf_count"] == 1
    assert receipt["checks"]["provider_calls_made"] is False
    assert receipt["checks"]["live_provider_calls_made"] is False
    assert receipt["checks"]["provider_cost_posture"]["api_cost_usd"] == 0.0
    assert receipt["checks"]["secrets_copied"] is False

    assert inventory["status"] == "blocked"
    assert inventory["stage_script_owner"]["script_id"] == "transcription_ocr.local_media_processing"
    assert inventory["inputs"]["download_corpus_stage_receipt_sha256"] == file_sha256(paths["stage_receipt"])
    assert inventory["inputs"]["talk_media_acquisition_receipt_sha256"] == file_sha256(
        talk_media_paths["receipt"]
    )
    assert inventory["inputs"]["talk_media_inventory_sha256"] == file_sha256(
        talk_media_paths["inventory"]
    )
    assert inventory["provider_cost_posture"]["paid_provider_calls_made"] is False
    assert inventory["provider_cost_posture"]["live_provider_calls_made"] is False
    assert inventory["provider_cost_posture"]["api_cost_usd"] == 0.0
    assert inventory["provider_cost_posture"]["secret_values_recorded"] is False
    assert inventory["provider_cost_posture"]["secrets_copied"] is False

    main_lane = inventory["lanes"]["paper_text_extraction"]
    workshop_lane = inventory["lanes"]["workshop_paper_text_extraction"]
    image_lane = inventory["lanes"]["image_ocr"]
    speech_lane = inventory["lanes"]["session_speech_transcription"]

    assert main_lane["records"][0]["status"] == "succeeded"
    assert main_lane["records"][0]["lane"] == "paper_text_extraction"
    assert main_lane["records"][0]["page_count"] == 1
    assert "Main paper fixture text" in (workspace / main_lane["records"][0]["output_path"]).read_text(encoding="utf-8")
    assert workshop_lane["records"][0]["status"] == "succeeded"
    assert workshop_lane["records"][0]["lane"] == "workshop_paper_text_extraction"
    assert workshop_lane["records"][0]["page_count"] == 1
    assert "Workshop paper fixture text" in (workspace / workshop_lane["records"][0]["output_path"]).read_text(encoding="utf-8")
    assert image_lane["status"] == "not_run_no_paid_provider_by_default"
    assert image_lane["provider_calls_made"] is False
    assert image_lane["text_output_count"] == 0
    assert image_lane["records"][0]["lane"] == "image_ocr"
    assert speech_lane["status"] == "blocked_no_local_audio_video_inputs"
    assert speech_lane["input_count"] == 0
    assert speech_lane["paper_pdf_substitutes_allowed"] is False
    assert speech_lane["image_substitutes_allowed"] is False
    assert inventory["lanes"]["workshop_talk_media"]["status"] == "blocked"
    assert inventory["lanes"]["workshop_talk_media"]["blocker_count"] == 1
    assert inventory["lanes"]["workshop_talk_media"]["records"][0]["workshop_id"] == (
        "ICLR.cc/2026/Workshop/DEMO"
    )

    blocker_codes = {blocker["code"] for blocker in inventory["blockers"]}
    assert "no_local_audio_video_inputs" in blocker_codes
    assert "image_ocr_not_run_no_paid_provider_default" in blocker_codes
    assert "workshop_talk_media_inaccessible_agenda" in blocker_codes
    assert inventory["lane_counts"]["session_speech_transcription"]["input_count"] == 0
    assert inventory["lane_counts"]["workshop_talk_media"]["blocked_count"] == 1
    assert inventory["lane_counts"]["image_ocr"]["input_count"] == 1
    assert inventory["output_inventory"]["text_output_count"] == 2

    assert len(blocker_paths) == 1
    blocker = json.loads(blocker_paths[0].read_text(encoding="utf-8"))
    assert blocker["blocker_class"] == "stage_blocked"
    assert "no_local_audio_video_inputs" in blocker["reason"]
    assert "workshop_talk_media_inaccessible_agenda" in blocker["reason"]
    assert "paper_text_extraction" in serialized
    assert "workshop_paper_text_extraction" in serialized
    assert "image_ocr" in serialized
    assert "session_speech_transcription" in serialized
    assert "workshop_talk_media" in serialized
    assert "provider_calls_made" in serialized
    assert SECRET_VALUE not in serialized

    unblock_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "conference_pipeline.workflow_unblock",
            "--workspace",
            str(workspace),
            "--run-id",
            "test-transcription-ocr",
            "--blocker-path",
            receipt["blocker_receipt_path"],
            "--operator",
            "pytest",
            "--note",
            "resume path regression test",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        text=True,
        capture_output=True,
        check=False,
    )
    assert unblock_result.returncode == 0, unblock_result.stderr

    resumed_result = run_runner(
        [
            "--workspace",
            str(workspace),
            "--run-id",
            "test-transcription-ocr",
            "--stage",
            "transcription-ocr",
            "--mode",
            "live",
            "--resume",
            "--no-env-file",
            "--summary",
        ],
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
    )
    assert resumed_result.returncode == 1, resumed_result.stderr
    resumed_inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    resumed_main_record = resumed_inventory["lanes"]["paper_text_extraction"]["records"][0]
    assert resumed_main_record["resume_reused_output"] is True
    assert resumed_main_record["page_count"] == 1


def test_transcription_ocr_consumes_successful_workshop_talk_media_records(
    tmp_path: Path,
) -> None:
    workspace = write_workspace(tmp_path)
    write_transcription_ocr_download_fixture(
        workspace,
        "test-transcription-ocr-workshop-talk-media-record",
    )
    media_path = workspace / "storage" / "workshops" / "talk-media" / "demo-talk.mp3"
    media_path.parent.mkdir(parents=True, exist_ok=True)
    media_path.write_bytes(b"fixture workshop talk audio")
    talk_media_paths = write_workshop_talk_media_scope_fixture(
        workspace,
        "test-transcription-ocr-workshop-talk-media-record",
        records=[
            {
                "status": "succeeded",
                "workshop_id": "ICLR.cc/2026/Workshop/DEMO",
                "workshop_title": "Demo Workshop",
                "talk_id": "talk-demo",
                "talk_title": "Demo Talk",
                "talk_speaker": "Dr Demo",
                "source_url": "https://iclr.cc/virtual/2026/workshop/demo",
                "media_url": "https://iclr.cc/media/demo-talk.mp3",
                "local_path": str(media_path.relative_to(workspace)),
                "sha256": file_sha256(media_path),
                "byte_count": media_path.stat().st_size,
                "timestamp_start": 0.0,
                "timestamp_end": 12.0,
            }
        ],
        blockers=[],
    )

    result = run_runner(
        [
            "--workspace",
            str(workspace),
            "--run-id",
            "test-transcription-ocr-workshop-talk-media-record",
            "--stage",
            "transcription-ocr",
            "--mode",
            "live",
            "--resume",
            "--no-env-file",
            "--summary",
        ],
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
    )

    assert result.returncode == 1, result.stderr
    receipt_path = (
        workspace
        / "storage"
        / "workflow-runs"
        / "test-transcription-ocr-workshop-talk-media-record"
        / "stage-05-transcription-ocr.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    inventory = json.loads(
        (workspace / "storage" / "transcription-ocr" / "transcription_ocr_inventory.json").read_text(
            encoding="utf-8"
        )
    )
    speech_records = inventory["lanes"]["session_speech_transcription"]["records"]
    workshop_records = inventory["lanes"]["workshop_talk_media"]["records"]

    assert receipt["checks"]["talk_media_acquisition_receipt_sha256"] == file_sha256(
        talk_media_paths["receipt"]
    )
    assert receipt["checks"]["talk_media_inventory_sha256"] == file_sha256(
        talk_media_paths["inventory"]
    )
    assert receipt["checks"]["talk_media_inventory_media_record_count"] == 1
    assert speech_records
    assert speech_records[0]["source_path"] == str(media_path.relative_to(workspace))
    assert speech_records[0]["local_path"] == str(media_path.relative_to(workspace))
    assert speech_records[0]["source_url"] == "https://iclr.cc/virtual/2026/workshop/demo"
    assert speech_records[0]["media_url"] == "https://iclr.cc/media/demo-talk.mp3"
    assert speech_records[0]["source_sha256"] == file_sha256(media_path)
    assert speech_records[0]["source_byte_count"] == media_path.stat().st_size
    assert speech_records[0]["workshop_id"] == "ICLR.cc/2026/Workshop/DEMO"
    assert speech_records[0]["workshop_title"] == "Demo Workshop"
    assert speech_records[0]["talk_id"] == "talk-demo"
    assert speech_records[0]["talk_title"] == "Demo Talk"
    assert speech_records[0]["talk_speaker"] == "Dr Demo"
    assert speech_records[0]["timestamp_start"] == 0.0
    assert speech_records[0]["timestamp_end"] == 12.0
    assert workshop_records[0]["status"] == "succeeded"
    assert inventory["lane_counts"]["session_speech_transcription"]["input_count"] == 1
    assert inventory["lane_counts"]["workshop_talk_media"]["succeeded_count"] == 1


def test_transcription_ocr_missing_download_corpus_receipt_writes_blocker(tmp_path: Path) -> None:
    workspace = write_workspace(tmp_path)
    write_transcription_ocr_download_fixture(
        workspace,
        "test-transcription-ocr-missing-download-receipt",
        write_stage_receipt=False,
    )

    result = run_runner(
        [
            "--workspace",
            str(workspace),
            "--run-id",
            "test-transcription-ocr-missing-download-receipt",
            "--stage",
            "transcription-ocr",
            "--mode",
            "live",
            "--resume",
            "--no-env-file",
            "--summary",
        ],
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
    )

    assert result.returncode == 1
    summary = json.loads(result.stdout)
    receipt_path = (
        workspace
        / "storage"
        / "workflow-runs"
        / "test-transcription-ocr-missing-download-receipt"
        / "stage-05-transcription-ocr.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    inventory_path = workspace / "storage" / "transcription-ocr" / "transcription_ocr_inventory.json"
    blocker_paths = list(
        (
            workspace
            / "storage"
            / "workflow-runs"
            / "test-transcription-ocr-missing-download-receipt"
            / "blocker-receipts"
        ).rglob("*.json")
    )

    assert summary["status"] == "blocked"
    assert summary["halt_stage_id"] == "transcription-ocr"
    assert receipt["status"] == "blocked"
    assert receipt["checks"]["download_corpus_stage_receipt_present"] is False
    assert not inventory_path.exists()
    assert len(blocker_paths) == 1
    blocker = json.loads(blocker_paths[0].read_text(encoding="utf-8"))
    assert blocker["blocker_class"] == "missing_receipt"
    assert "download-corpus stage receipt" in blocker["reason"]


def test_transcription_ocr_failed_download_corpus_receipt_writes_blocker(tmp_path: Path) -> None:
    workspace = write_workspace(tmp_path)
    write_transcription_ocr_download_fixture(
        workspace,
        "test-transcription-ocr-failed-download-receipt",
        stage_status="blocked",
    )

    result = run_runner(
        [
            "--workspace",
            str(workspace),
            "--run-id",
            "test-transcription-ocr-failed-download-receipt",
            "--stage",
            "transcription-ocr",
            "--mode",
            "live",
            "--resume",
            "--no-env-file",
            "--summary",
        ],
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
    )

    assert result.returncode == 1
    summary = json.loads(result.stdout)
    receipt_path = (
        workspace
        / "storage"
        / "workflow-runs"
        / "test-transcription-ocr-failed-download-receipt"
        / "stage-05-transcription-ocr.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    inventory_path = workspace / "storage" / "transcription-ocr" / "transcription_ocr_inventory.json"
    blocker_paths = list(
        (
            workspace
            / "storage"
            / "workflow-runs"
            / "test-transcription-ocr-failed-download-receipt"
            / "blocker-receipts"
        ).rglob("*.json")
    )

    assert summary["status"] == "blocked"
    assert summary["halt_stage_id"] == "transcription-ocr"
    assert receipt["status"] == "blocked"
    assert receipt["checks"]["download_corpus_stage_status"] == "blocked"
    assert not inventory_path.exists()
    assert len(blocker_paths) == 1
    blocker = json.loads(blocker_paths[0].read_text(encoding="utf-8"))
    assert blocker["blocker_class"] == "stage_blocked"
    assert "succeeded download-corpus stage receipt" in blocker["reason"]


def test_corpus_inventory_index_builds_source_aware_inventory_and_index(tmp_path: Path) -> None:
    workspace = write_workspace(tmp_path)
    paths = write_corpus_inventory_index_fixture(workspace, "test-corpus-inventory-index")

    result = run_runner(
        [
            "--workspace",
            str(workspace),
            "--run-id",
            "test-corpus-inventory-index",
            "--stage",
            "corpus-inventory-index",
            "--mode",
            "live",
            "--resume",
            "--no-env-file",
            "--summary",
        ],
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "EMBEDDING_PROVIDER": "fake"},
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    receipt_path = (
        workspace
        / "storage"
        / "workflow-runs"
        / "test-corpus-inventory-index"
        / "stage-06-corpus-inventory-index.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    inventory_path = workspace / "storage" / "corpus" / "corpus_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    index_digest_path = workspace / "storage" / "vector-store" / "test-corpus-inventory-index" / "index_digest.json"
    index_digest = json.loads(index_digest_path.read_text(encoding="utf-8"))
    papers_chunks = (
        workspace
        / "storage"
        / "vector-store"
        / "test-corpus-inventory-index"
        / "stores"
        / "papers"
        / "chunks.jsonl"
    )
    artifacts_chunks = (
        workspace
        / "storage"
        / "vector-store"
        / "test-corpus-inventory-index"
        / "stores"
        / "artifacts"
        / "chunks.jsonl"
    )
    serialized = json.dumps(
        {"summary": summary, "receipt": receipt, "inventory": inventory, "index": index_digest},
        sort_keys=True,
    )

    assert summary["status"] == "succeeded"
    assert receipt["status"] == "succeeded"
    assert receipt["stage_contract"]["status"] == "pass"
    assert "indexing.token_bounded_chunking" in receipt["stage_contract"]["required_script_ids"]
    assert "indexing.local_embedding_materialization_policy" in receipt["stage_contract"]["required_script_ids"]
    assert receipt["checks"]["provider_calls_made"] is False
    assert receipt["checks"]["live_provider_calls_made"] is False
    assert receipt["checks"]["secrets_copied"] is False
    assert receipt["checks"]["download_corpus_stage_receipt_sha256"] == file_sha256(paths["stage4_receipt"])
    assert receipt["checks"]["download_corpus_inventory_sha256"] == file_sha256(paths["download_inventory"])
    assert receipt["checks"]["transcription_ocr_stage_receipt_sha256"] == file_sha256(paths["stage5_receipt"])
    assert receipt["checks"]["transcription_ocr_inventory_sha256"] == file_sha256(paths["transcription_inventory"])
    assert receipt["checks"]["chunk_length_stats"]["snippet_only_rejected"] is True
    assert receipt["checks"]["token_budget_policy"]["policy_digest"]
    assert receipt["checks"]["token_budget_policy_digest"] == receipt["checks"]["token_budget_policy"]["policy_digest"]
    assert receipt["checks"]["embedding_policy_digest"]
    assert receipt["checks"]["embedding_max_length"] is None
    assert receipt["checks"]["materialization_batch_size"] == 8
    assert receipt["checks"]["embedding_readiness"]["attempted"] is True
    assert receipt["checks"]["embedding_readiness"]["client_start_or_load_attempted"] is True
    assert receipt["checks"]["embedding_readiness"]["status"] == "available"
    assert receipt["checks"]["embedding_status"] == "vector_embeddings_written_local_embedding_client_ready"
    assert receipt["checks"]["vector_embeddings_written"] is True
    assert receipt["checks"]["embedded_chunk_counts"] == receipt["checks"]["chunk_counts"]
    assert receipt["checks"]["vector_materialization"]["status"] == "succeeded"
    for store in receipt["checks"]["vector_materialization"]["stores"].values():
        assert store["embedding_sidecar_manifest_path"]
        assert store["embedding_sidecar_manifest_sha256"]
        assert store["embedding_sidecar_format"] == "npz-buckets-v2"
        assert store["embedding_sidecar_shard_count"] >= 1
        assert store["chroma_insert_batch_size"] >= store["chunk_count"]
    assert receipt["checks"]["provider_cost_posture"]["embedding_provider"] == "fake"
    assert "stores/papers" in receipt["checks"]["store_paths"]["papers"]
    assert "stores/artifacts" in receipt["checks"]["store_paths"]["artifacts"]
    assert inventory["receipt_type"] == "conference_pipeline.iclr2026_corpus_inventory"
    assert inventory["upstream"]["download_corpus_inventory_sha256"] == file_sha256(paths["download_inventory"])
    assert inventory["upstream"]["transcription_ocr_inventory_sha256"] == file_sha256(paths["transcription_inventory"])
    assert inventory["artifact_counts"]["indexable_text"] == 3
    assert inventory["artifact_counts"]["audit_only"] == 2
    assert {"paper_text", "workshop_paper_text", "speech_transcript", "media_asset"} <= {
        record["source_family_id"] for record in inventory["artifact_records"]
    }
    assert index_digest["stores"]["papers"]["chunk_count"] >= 2
    assert index_digest["stores"]["artifacts"]["chunk_count"] >= 1
    assert index_digest["token_budget_policy_digest"] == receipt["checks"]["token_budget_policy_digest"]
    assert index_digest["embedding_policy_digest"] == receipt["checks"]["embedding_policy_digest"]
    assert index_digest["materialization_batch_size"] == 8
    assert index_digest["stores"]["papers"]["embedded_chunk_count"] == index_digest["stores"]["papers"]["chunk_count"]
    assert index_digest["stores"]["artifacts"]["embedded_chunk_count"] == index_digest["stores"]["artifacts"]["chunk_count"]
    assert (index_digest_path.parent / "stores" / "papers" / "v15_index_digest.json").exists()
    assert (index_digest_path.parent / "stores" / "artifacts" / "v15_index_digest.json").exists()
    assert index_digest["embedding_readiness"]["status"] == "available"
    assert index_digest["vector_embeddings_written"] is True
    assert index_digest["embedded_chunk_counts"] == receipt["checks"]["chunk_counts"]
    assert "chunk_length_stats" in index_digest
    assert papers_chunks.exists()
    assert artifacts_chunks.exists()
    papers_chunk_text = papers_chunks.read_text(encoding="utf-8")
    artifact_chunk_lines = [
        json.loads(line)
        for line in artifacts_chunks.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert "Main Paper Title" in papers_chunk_text
    assert any(
        "artifacts, multimodal tool use, and evaluation" in chunk["text"]
        for chunk in artifact_chunk_lines
    )
    transcript_chunks = [
        chunk for chunk in artifact_chunk_lines if chunk["source_family_id"] == "speech_transcript"
    ]
    assert transcript_chunks
    assert transcript_chunks[0]["text_path"].endswith(
        "storage/transcription-ocr/workshop-talks/talk_annotated.json"
    )
    assert transcript_chunks[0]["indexed_text_kind"] == "annotated_transcript_json"
    assert "Workshop: Workshop Demo" in transcript_chunks[0]["text"]
    assert "Talk: Talk Demo" in transcript_chunks[0]["text"]
    assert "Speaker: Dr Demo" in transcript_chunks[0]["text"]
    source_context = transcript_chunks[0]["source_context"]
    assert source_context["slideslive_id"] == "39000001"
    assert source_context["source_title"] == "Source Demo"
    assert source_context["session_title"] == "Session Demo"
    assert source_context["workshop_id"] == "workshop_demo"
    assert source_context["workshop_title"] == "Workshop Demo"
    assert source_context["talk_id"] == "talk_001"
    assert source_context["talk_title"] == "Talk Demo"
    assert source_context["talk_speaker"] == "Dr Demo"
    assert transcript_chunks[0]["token_budget_policy_digest"] == index_digest["token_budget_policy_digest"]
    assert transcript_chunks[0]["embedding_policy_digest"] == index_digest["embedding_policy_digest"]
    assert "SECRET_VALUE" not in serialized
    assert SECRET_VALUE not in serialized


def test_corpus_inventory_index_resolves_main_paper_authors_via_shared_enrichment(
    tmp_path: Path,
) -> None:
    """Plan 10.1-07: a main-track lane record whose download artifact carries
    an OpenReview-normalized author list (normalize_openreview_2026_note()'s
    own output shape) resolves through the shared
    conference_pipeline.indexing.paper_source_metadata.enrich_paper_records
    entry point -- the same shared module CVPR2026's runner uses -- with a
    populated authors list, a populated paper_title, and a provenance label
    imported from the shared module (never a runner-local string literal).
    """

    runner = load_runner_module()
    workspace = write_workspace(tmp_path)
    run_id = "test-corpus-inventory-index-authors"
    authors = ["Ada Lovelace", "Alan Turing"]
    write_corpus_inventory_index_fixture(workspace, run_id, main_paper_authors=authors)

    result = run_runner(
        [
            "--workspace",
            str(workspace),
            "--run-id",
            run_id,
            "--stage",
            "corpus-inventory-index",
            "--mode",
            "live",
            "--resume",
            "--no-env-file",
            "--summary",
        ],
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "EMBEDDING_PROVIDER": "fake"},
    )
    assert result.returncode == 0, result.stderr

    receipt_path = (
        workspace
        / "storage"
        / "workflow-runs"
        / run_id
        / "stage-06-corpus-inventory-index.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    inventory_path = workspace / "storage" / "corpus" / "corpus_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))

    main_records = [
        record
        for record in inventory["artifact_records"]
        if record["source_family_id"] == "paper_text"
    ]
    assert len(main_records) == 1
    main_record = main_records[0]
    assert main_record["authors"] == authors
    assert main_record["paper_title"] == "Main Paper Title"
    assert (
        main_record["source_metadata_provenance"]["authors"]
        == runner.platform_paper_source_metadata.PROVENANCE_SUBMISSION_SYSTEM_METADATA
    )
    assert (
        main_record["source_metadata_provenance"]["title"]
        == runner.platform_paper_source_metadata.PROVENANCE_SUBMISSION_SYSTEM_METADATA
    )

    coverage = receipt["checks"]["source_metadata_coverage"]
    assert set(coverage.keys()) == {"papers_main", "papers_workshop", "artifacts"}
    assert coverage["papers_main"]["total"] == 1
    assert coverage["papers_main"]["authors_resolved"] == 1
    assert coverage["papers_main"]["title_resolved"] == 1
    assert (
        coverage["papers_main"]["provenance"]["authors"][
            runner.platform_paper_source_metadata.PROVENANCE_SUBMISSION_SYSTEM_METADATA
        ]
        == 1
    )


def test_corpus_inventory_index_no_author_data_stays_honestly_unresolved(
    tmp_path: Path,
) -> None:
    """Plan 10.1-07: a lane record with no author data anywhere upstream
    (the default fixture's download artifact carries no ``authors`` key)
    produces a corpus record with unresolved authors provenance and no
    fabricated author list -- matching the CVPR2026 behavior exactly (an
    individual paper's unresolved metadata is a coverage number, never a
    fabricated value and never a stage blocker).
    """

    runner = load_runner_module()
    workspace = write_workspace(tmp_path)
    run_id = "test-corpus-inventory-index-no-authors"
    write_corpus_inventory_index_fixture(workspace, run_id)

    result = run_runner(
        [
            "--workspace",
            str(workspace),
            "--run-id",
            run_id,
            "--stage",
            "corpus-inventory-index",
            "--mode",
            "live",
            "--resume",
            "--no-env-file",
            "--summary",
        ],
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "EMBEDDING_PROVIDER": "fake"},
    )
    assert result.returncode == 0, result.stderr

    inventory_path = workspace / "storage" / "corpus" / "corpus_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    main_records = [
        record
        for record in inventory["artifact_records"]
        if record["source_family_id"] == "paper_text"
    ]
    assert len(main_records) == 1
    main_record = main_records[0]
    assert "authors" not in main_record
    assert (
        main_record["source_metadata_provenance"]["authors"]
        == runner.platform_paper_source_metadata.PROVENANCE_UNRESOLVED
    )


def test_iclr2026_runner_defines_no_local_title_or_author_extraction_function() -> None:
    """Plan 10.1-07: the sibling runner contributes zero new extraction logic
    -- every title/author resolution is delegated to the shared
    conference_pipeline.indexing.paper_source_metadata module, never a
    function defined inside this runner.
    """

    runner = load_runner_module()
    suspicious_substrings = ("extract_title", "extract_author", "parse_title", "parse_author")
    offending = [
        name
        for name in dir(runner)
        if callable(getattr(runner, name, None))
        and any(substring in name.lower() for substring in suspicious_substrings)
    ]
    assert offending == []


def test_corpus_inventory_index_batch_size_change_preserves_embedding_identity(
    tmp_path: Path,
) -> None:
    workspace = write_workspace(tmp_path)
    run_id = "test-corpus-inventory-policy-digest"
    write_corpus_inventory_index_fixture(workspace, run_id)

    command = [
        "--workspace",
        str(workspace),
        "--run-id",
        run_id,
        "--stage",
        "corpus-inventory-index",
        "--mode",
        "live",
        "--resume",
        "--no-env-file",
        "--summary",
    ]
    receipt_path = (
        workspace / "storage" / "workflow-runs" / run_id / "stage-06-corpus-inventory-index.json"
    )

    first = run_runner(
        command,
        env={
            "PYTHONPATH": str(REPO_ROOT / "src"),
            "EMBEDDING_PROVIDER": "fake",
            "CORPUS_VECTOR_EMBED_BATCH_SIZE": "8",
        },
    )
    assert first.returncode == 0, first.stderr
    first_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    first_policy_digest = first_receipt["checks"]["embedding_policy_digest"]

    second = run_runner(
        command,
        env={
            "PYTHONPATH": str(REPO_ROOT / "src"),
            "EMBEDDING_PROVIDER": "fake",
            "CORPUS_VECTOR_EMBED_BATCH_SIZE": "7",
        },
    )
    assert second.returncode == 0, second.stderr
    second_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    materialized_stores = second_receipt["checks"]["vector_materialization"]["stores"]

    assert second_receipt["checks"]["embedding_policy_digest"] == first_policy_digest
    assert second_receipt["checks"]["materialization_batch_size"] == 7
    for store in materialized_stores.values():
        assert store["embedding_resume_count"] == store["chunk_count"]
        assert store["embedding_upsert_count"] == 0
        assert store["embedding_sidecar_resume_count"] == store["chunk_count"]


def test_corpus_inventory_index_reuses_valid_sidecars_on_resume(
    tmp_path: Path,
) -> None:
    workspace = write_workspace(tmp_path)
    run_id = "test-corpus-inventory-reuse-sidecars"
    write_corpus_inventory_index_fixture(workspace, run_id)

    command = [
        "--workspace",
        str(workspace),
        "--run-id",
        run_id,
        "--stage",
        "corpus-inventory-index",
        "--mode",
        "live",
        "--resume",
        "--no-env-file",
        "--summary",
    ]
    env = {
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "EMBEDDING_PROVIDER": "fake",
        "CORPUS_VECTOR_EMBED_BATCH_SIZE": "8",
    }
    first = run_runner(command, env=env)
    assert first.returncode == 0, first.stderr

    corpus_inventory = workspace / "storage" / "corpus" / "corpus_inventory.json"
    vector_root = workspace / "storage" / "vector-store" / run_id
    paper_chunks = vector_root / "stores" / "papers" / "chunks.jsonl"
    artifact_chunks = vector_root / "stores" / "artifacts" / "chunks.jsonl"
    old_timestamp = 1_700_000_000
    os.utime(corpus_inventory, (old_timestamp, old_timestamp))
    os.utime(paper_chunks, (old_timestamp, old_timestamp))
    os.utime(artifact_chunks, (old_timestamp, old_timestamp))

    second = run_runner(command, env=env)
    assert second.returncode == 0, second.stderr
    receipt = json.loads(
        (
            workspace
            / "storage"
            / "workflow-runs"
            / run_id
            / "stage-06-corpus-inventory-index.json"
        ).read_text(encoding="utf-8")
    )
    index_digest = json.loads((vector_root / "index_digest.json").read_text(encoding="utf-8"))

    assert receipt["checks"]["sidecar_resume_reused"] is True
    assert index_digest["sidecar_resume_reused"] is True
    assert int(corpus_inventory.stat().st_mtime) == old_timestamp
    assert int(paper_chunks.stat().st_mtime) == old_timestamp
    assert int(artifact_chunks.stat().st_mtime) == old_timestamp


def test_corpus_index_sidecar_resume_rejects_inventory_digest_mismatch(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    workspace = write_workspace(tmp_path)
    run_id = "test-corpus-sidecar-digest-mismatch"
    vector_root = workspace / "storage" / "vector-store" / run_id
    paper_store = vector_root / "stores" / "papers"
    artifact_store = vector_root / "stores" / "artifacts"
    paper_store.mkdir(parents=True)
    artifact_store.mkdir(parents=True)

    for store_dir in (paper_store, artifact_store):
        (store_dir / "chunks.jsonl").write_text(
            json.dumps({"chunk_id": f"{store_dir.name}:0", "text": "old"}) + "\n",
            encoding="utf-8",
        )
        (store_dir / "metadata.json").write_text("{}\n", encoding="utf-8")

    embedding_readiness = {
        "embedding_policy_digest": "fake-digest",
        "materialization_batch_size": 8,
    }
    digest = {
        "receipt_type": "conference_pipeline.iclr2026_corpus_index_digest",
        "corpus_inventory_sha256": "old-inventory-digest",
        "token_budget_policy_digest": runner.policy_metadata(runner.default_token_budget_policy())[
            "policy_digest"
        ],
        "embedding_policy_digest": "fake-digest",
        "materialization_batch_size": 8,
        "stores": {
            "papers": {
                "chunks_path": runner.workspace_relpath(workspace, paper_store / "chunks.jsonl"),
                "chunks_sha256": file_sha256(paper_store / "chunks.jsonl"),
                "metadata_path": runner.workspace_relpath(workspace, paper_store / "metadata.json"),
                "metadata_sha256": file_sha256(paper_store / "metadata.json"),
                "chunk_count": 1,
            },
            "artifacts": {
                "chunks_path": runner.workspace_relpath(workspace, artifact_store / "chunks.jsonl"),
                "chunks_sha256": file_sha256(artifact_store / "chunks.jsonl"),
                "metadata_path": runner.workspace_relpath(workspace, artifact_store / "metadata.json"),
                "metadata_sha256": file_sha256(artifact_store / "metadata.json"),
                "chunk_count": 1,
            },
        },
    }
    (vector_root / "index_digest.json").write_text(
        json.dumps(digest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    assert (
        runner.load_reusable_corpus_index_sidecars(
            workspace=workspace,
            run_id=run_id,
            corpus_inventory_sha256="new-inventory-digest",
            embedding_readiness=embedding_readiness,
        )
        is None
    )


def test_corpus_sidecar_rebuild_preserves_previous_chunks_on_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = load_runner_module()
    workspace = write_workspace(tmp_path)
    run_id = "test-sidecar-atomic-rebuild"
    text_dir = workspace / "storage" / "transcription-ocr" / "text-extraction"
    first_text = text_dir / "first.txt"
    second_text = text_dir / "second.txt"
    first_text.parent.mkdir(parents=True, exist_ok=True)
    first_text.write_text("first replacement chunk source\n", encoding="utf-8")
    second_text.write_text("second replacement chunk source\n", encoding="utf-8")
    corpus_inventory_path = workspace / "storage" / "corpus" / "corpus_inventory.json"
    corpus_inventory_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_inventory_path.write_text("{}\n", encoding="utf-8")
    chunks_path = (
        workspace
        / "storage"
        / "vector-store"
        / run_id
        / "stores"
        / "papers"
        / "chunks.jsonl"
    )
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.write_text("previous-complete-sidecar\n", encoding="utf-8")
    artifact_records = [
        {
            "artifact_id": "paper-1",
            "indexable_text": True,
            "source_family_id": "paper_text",
            "indexed_text_path": str(first_text.relative_to(workspace)),
            "source_path": "papers/first.pdf",
            "source_sha256": "source-1",
            "indexed_text_sha256": "text-1",
            "indexed_text_kind": "pdf_text",
            "metadata": {"title": "First Paper"},
        },
        {
            "artifact_id": "paper-2",
            "indexable_text": True,
            "source_family_id": "paper_text",
            "indexed_text_path": str(second_text.relative_to(workspace)),
            "source_path": "papers/second.pdf",
            "source_sha256": "source-2",
            "indexed_text_sha256": "text-2",
            "indexed_text_kind": "pdf_text",
            "metadata": {"title": "Second Paper"},
        },
    ]
    calls = 0

    def fake_chunks_for_record(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("fixture-sidecar-rebuild-failure")
        return [
            SimpleNamespace(
                text="replacement chunk text",
                chunk_index=0,
                chunk_count=1,
                strategy="fixture",
                char_offsets={"start": 0, "end": 22},
                locator={"kind": "fixture"},
            )
        ]

    monkeypatch.setattr(runner, "chunks_for_record", fake_chunks_for_record)
    monkeypatch.setattr(runner, "approximate_embedding_token_count", lambda _text: 3)

    try:
        runner.build_corpus_index_sidecars(
            workspace=workspace,
            run_id=run_id,
            artifact_records=artifact_records,
            corpus_inventory_path=corpus_inventory_path,
            corpus_inventory_sha256="fixture-corpus-sha",
            embedding_readiness={
                "status": "available",
                "embedding_status": "chunk_sidecars_written_local_embedding_client_ready",
                "embedding_policy": {"provider": "fake"},
                "embedding_policy_digest": "fixture-embedding-policy",
                "embedding_max_length": None,
                "materialization_batch_size": 8,
            },
        )
    except RuntimeError as exc:
        assert "fixture-sidecar-rebuild-failure" in str(exc)
    else:  # pragma: no cover - this branch should never be reachable
        raise AssertionError("sidecar rebuild failure was not raised")

    assert chunks_path.read_text(encoding="utf-8") == "previous-complete-sidecar\n"
    assert not list(chunks_path.parent.glob(".chunks.jsonl.*.tmp"))


def test_embedding_readiness_probe_attempts_start_before_failure(monkeypatch) -> None:
    runner = load_runner_module()
    attempts: list[object] = []

    class FailingEmbeddingClient:
        provider = "mlx"
        model = "fixture-bge"
        embedding_dim = 1024

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            attempts.append(texts)
            raise RuntimeError("startup failed api_key=fixture-secret")

    def fake_build_embedding_client(config: dict[str, object]) -> FailingEmbeddingClient:
        attempts.append(config)
        return FailingEmbeddingClient()

    monkeypatch.setenv("EMBEDDING_PROVIDER", "mlx")
    monkeypatch.setattr(runner, "build_embedding_client", fake_build_embedding_client)

    readiness = runner.probe_local_embedding_readiness()

    assert readiness["status"] == "failed"
    assert readiness["client_start_or_load_attempted"] is True
    assert readiness["embedding_status"] == "blocked_local_embedding_client_start_failed"
    assert attempts
    assert "fixture-secret" not in json.dumps(readiness)
    assert "api_key=REDACTED" in readiness["error"]


def test_corpus_inventory_index_blocks_on_missing_transcription_receipt(tmp_path: Path) -> None:
    workspace = write_workspace(tmp_path)
    write_corpus_inventory_index_fixture(
        workspace,
        "test-corpus-inventory-missing-transcription",
        write_transcription_receipt=False,
    )

    result = run_runner(
        [
            "--workspace",
            str(workspace),
            "--run-id",
            "test-corpus-inventory-missing-transcription",
            "--stage",
            "corpus-inventory-index",
            "--mode",
            "live",
            "--resume",
            "--no-env-file",
            "--summary",
        ],
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
    )

    assert result.returncode == 1
    summary = json.loads(result.stdout)
    receipt_path = (
        workspace
        / "storage"
        / "workflow-runs"
        / "test-corpus-inventory-missing-transcription"
        / "stage-06-corpus-inventory-index.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    blocker_paths = list(
        (
            workspace
            / "storage"
            / "workflow-runs"
            / "test-corpus-inventory-missing-transcription"
            / "blocker-receipts"
        ).rglob("*.json")
    )

    assert summary["status"] == "blocked"
    assert summary["halt_stage_id"] == "corpus-inventory-index"
    assert receipt["status"] == "blocked"
    assert receipt["checks"]["blocker_code"] == "transcription_ocr_stage_not_succeeded"
    assert receipt["checks"]["transcription_ocr_stage_receipt_present"] is False
    assert not (workspace / "storage" / "corpus" / "corpus_inventory.json").exists()
    assert len(blocker_paths) == 1


def test_corpus_inventory_index_blocks_on_stale_transcription_inventory_digest(tmp_path: Path) -> None:
    workspace = write_workspace(tmp_path)
    paths = write_corpus_inventory_index_fixture(
        workspace,
        "test-corpus-inventory-stale-transcription",
        stale_transcription_digest=True,
    )

    result = run_runner(
        [
            "--workspace",
            str(workspace),
            "--run-id",
            "test-corpus-inventory-stale-transcription",
            "--stage",
            "corpus-inventory-index",
            "--mode",
            "live",
            "--resume",
            "--no-env-file",
            "--summary",
        ],
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
    )

    assert result.returncode == 1
    receipt_path = (
        workspace
        / "storage"
        / "workflow-runs"
        / "test-corpus-inventory-stale-transcription"
        / "stage-06-corpus-inventory-index.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert receipt["status"] == "blocked"
    assert receipt["checks"]["blocker_code"] == "transcription_ocr_inventory_digest_mismatch"
    assert receipt["checks"]["transcription_ocr_output_inventory_sha256"] != file_sha256(
        paths["transcription_inventory"]
    )
    assert receipt["checks"]["transcription_ocr_inventory_sha256"] == file_sha256(
        paths["transcription_inventory"]
    )
    assert not (workspace / "storage" / "vector-store" / "test-corpus-inventory-stale-transcription").exists()


def test_corpus_inventory_index_blocks_workshop_speech_without_talk_annotation(
    tmp_path: Path,
) -> None:
    workspace = write_workspace(tmp_path)
    run_id = "test-corpus-inventory-missing-workshop-talks"
    paths = write_corpus_inventory_index_fixture(workspace, run_id)
    transcription_inventory_path = paths["transcription_inventory"]
    transcription_inventory = json.loads(transcription_inventory_path.read_text(encoding="utf-8"))
    speech_record = transcription_inventory["lanes"]["session_speech_transcription"]["records"][0]
    speech_record.pop("annotated_transcript_json_path", None)
    speech_record.pop("workshop_talk_discovery_receipt_path", None)
    speech_record.pop("workshop_talks_json_path", None)
    transcription_inventory_path.write_text(
        json.dumps(transcription_inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    stage5_receipt_path = paths["stage5_receipt"]
    stage5_receipt = json.loads(stage5_receipt_path.read_text(encoding="utf-8"))
    stage5_receipt["checks"]["output_inventory"]["sha256"] = file_sha256(
        transcription_inventory_path
    )
    stage5_receipt_path.write_text(
        json.dumps(stage5_receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = run_runner(
        [
            "--workspace",
            str(workspace),
            "--run-id",
            run_id,
            "--stage",
            "corpus-inventory-index",
            "--mode",
            "live",
            "--resume",
            "--no-env-file",
            "--summary",
        ],
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "EMBEDDING_PROVIDER": "fake"},
    )

    assert result.returncode == 1
    receipt = json.loads(
        (
            workspace
            / "storage"
            / "workflow-runs"
            / run_id
            / "stage-06-corpus-inventory-index.json"
        ).read_text(encoding="utf-8")
    )

    assert receipt["status"] == "blocked"
    assert receipt["checks"]["blocker_code"] == "workshop_talk_segmentation_required"
    assert receipt["checks"]["blockers"][0]["code"] == "workshop_talk_segmentation_required"
    assert "transcription_ocr.workshop_talk_boundary_discovery" in receipt["checks"]["blockers"][0][
        "required_owner_ids"
    ]
    assert not (workspace / "storage" / "vector-store" / run_id).exists()


def test_corpus_inventory_index_blocks_when_workshop_talk_media_acquisition_missing(
    tmp_path: Path,
) -> None:
    workspace = write_workspace(tmp_path)
    run_id = "test-corpus-inventory-missing-workshop-talk-media"
    paths = write_corpus_inventory_index_fixture(workspace, run_id)

    download_inventory_path = paths["download_inventory"]
    download_inventory = json.loads(download_inventory_path.read_text(encoding="utf-8"))
    for artifact in download_inventory["local_artifacts"] + download_inventory["direct_download_artifacts"]:
        if artifact.get("artifact_kind") == "media_asset":
            for key in (
                "workshop_id",
                "workshop_title",
                "talk_id",
                "talk_title",
                "talk_speaker",
            ):
                artifact.pop(key, None)
    download_inventory_path.write_text(
        json.dumps(download_inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    transcription_inventory_path = paths["transcription_inventory"]
    transcription_inventory = json.loads(transcription_inventory_path.read_text(encoding="utf-8"))
    speech_record = transcription_inventory["lanes"]["session_speech_transcription"]["records"][0]
    for key in (
        "annotated_transcript_json_path",
        "workshop_talk_discovery_receipt_path",
        "workshop_talks_json_path",
        "workshop_id",
        "workshop_title",
        "talk_id",
        "talk_title",
        "talk_speaker",
    ):
        speech_record.pop(key, None)
    transcription_inventory_path.write_text(
        json.dumps(transcription_inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    stage4_receipt = json.loads(paths["stage4_receipt"].read_text(encoding="utf-8"))
    stage4_receipt["checks"]["download_corpus_inventory_sha256"] = file_sha256(
        download_inventory_path
    )
    paths["stage4_receipt"].write_text(
        json.dumps(stage4_receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    stage5_receipt = json.loads(paths["stage5_receipt"].read_text(encoding="utf-8"))
    stage5_receipt["checks"]["download_corpus_inventory_sha256"] = file_sha256(
        download_inventory_path
    )
    stage5_receipt["checks"]["output_inventory"]["sha256"] = file_sha256(
        transcription_inventory_path
    )
    paths["stage5_receipt"].write_text(
        json.dumps(stage5_receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = run_runner(
        [
            "--workspace",
            str(workspace),
            "--run-id",
            run_id,
            "--stage",
            "corpus-inventory-index",
            "--mode",
            "live",
            "--resume",
            "--no-env-file",
            "--summary",
        ],
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "EMBEDDING_PROVIDER": "fake"},
    )

    assert result.returncode == 1
    receipt = json.loads(
        (
            workspace
            / "storage"
            / "workflow-runs"
            / run_id
            / "stage-06-corpus-inventory-index.json"
        ).read_text(encoding="utf-8")
    )

    assert receipt["status"] == "blocked"
    assert receipt["checks"]["blocker_code"] == "workshop_talk_media_acquisition_required"
    scope = receipt["checks"]["workshop_talk_media_scope"]
    assert scope["workshop_paper_pdf_count"] == 1
    assert scope["workshop_speech_count"] == 0
    assert scope["workshop_talk_media_required"] is True
    assert scope["workshop_talk_media_proven_or_waived"] is False
    assert not (workspace / "storage" / "vector-store" / run_id).exists()


def test_analysis_stage_blocks_gracefully_after_stage6_gate(
    tmp_path: Path,
) -> None:
    workspace = write_workspace(tmp_path)
    run_id = "test-analysis-exhaustive-extraction-block"
    write_corpus_inventory_index_fixture(workspace, run_id)
    write_topic_strategy(workspace, run_id)

    stage6 = run_runner(
        [
            "--workspace",
            str(workspace),
            "--run-id",
            run_id,
            "--stage",
            "corpus-inventory-index",
            "--mode",
            "live",
            "--resume",
            "--no-env-file",
            "--summary",
        ],
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "EMBEDDING_PROVIDER": "fake"},
    )
    assert stage6.returncode == 0, stage6.stderr

    result = run_runner(
        [
            "--workspace",
            str(workspace),
            "--run-id",
            run_id,
            "--stage",
            "analysis",
            "--mode",
            "live",
            "--resume",
            "--no-env-file",
            "--summary",
        ],
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
    )

    assert result.returncode == 1
    receipt = json.loads(
        (
            workspace
            / "storage"
            / "workflow-runs"
            / run_id
            / "stage-07-analysis.json"
        ).read_text(encoding="utf-8")
    )

    assert receipt["status"] == "blocked"
    assert receipt["checks"]["blocker_code"] == "analysis_exhaustive_extraction_required"
    assert receipt["checks"]["stage6_vector_embeddings_written"] is True
    assert receipt["checks"]["exhaustive_extraction_receipt_present"] is False
    assert receipt["checks"]["exhaustive_extraction_by_topic_count"] == 0
    assert receipt["checks"]["exhaustive_extraction_by_artifact_count"] == 0
    assert not (workspace / "analysis" / "analysis_manifest.json").exists()


def test_analysis_source_packs_include_llm_full_read_evidence(tmp_path: Path) -> None:
    workspace = write_workspace(tmp_path)
    run_id = "test-analysis-full-read-source-pack"
    runner = load_runner_module()
    vector_root = workspace / "storage" / "vector-store" / run_id
    paper_store = vector_root / "stores" / "papers"
    artifact_store = vector_root / "stores" / "artifacts"
    paper_store.mkdir(parents=True)
    artifact_store.mkdir(parents=True)
    paper_chunk = {
        "chunk_id": "paper:reasoning:0",
        "artifact_id": "paper_abstract:paper-1",
        "source_path": "https://openreview.net/forum?id=paper-1",
        "text": "Reasoning and test-time compute paper evidence.",
    }
    artifact_chunk = {
        "chunk_id": "speech:reasoning:0",
        "artifact_id": "speech_transcript:talk-1:segment:0001",
        "source_path": "cache/media/talk.mp3",
        "text": "A talk about reasoning traces and efficient inference.",
    }
    (paper_store / "chunks.jsonl").write_text(json.dumps(paper_chunk) + "\n", encoding="utf-8")
    (artifact_store / "chunks.jsonl").write_text(
        json.dumps(artifact_chunk) + "\n",
        encoding="utf-8",
    )
    index_digest = {
        "stores": {
            "papers": {
                "chunks_path": runner.workspace_relpath(workspace, paper_store / "chunks.jsonl")
            },
            "artifacts": {
                "chunks_path": runner.workspace_relpath(workspace, artifact_store / "chunks.jsonl")
            },
        }
    }
    (vector_root / "index_digest.json").write_text(
        json.dumps(index_digest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt_path = (
        workspace
        / "storage"
        / "workflow-runs"
        / run_id
        / "14A-llm-full-read-extraction-receipt.json"
    )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(
            {
                "status": "succeeded",
                "scope": "all",
                "model": "mlx-community/Qwen3.5-35B-A3B-4bit",
                "checks": {
                    "completion_state": "complete",
                    "expected_unit_count": 2,
                    "completed_unit_count": 2,
                    "failed_count": 0,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    topic_dir = workspace / "storage" / "extractions" / run_id / "llm-full-read" / "by-topic"
    topic_dir.mkdir(parents=True)
    topic_dir.joinpath("01_llm_reasoning_test_time.json").write_text(
        json.dumps(
            {
                "topic_id": "01_llm_reasoning_test_time",
                "unit_count": 2,
                "items": [
                    {
                        "unit_id": "speech_transcript:talk-1:segment:0001",
                        "input_kind": "transcript_segment",
                        "important": True,
                        "confidence": 0.92,
                        "summary": "A talk segment links latent reasoning to efficient inference.",
                        "metadata": {
                            "talk_title": "Latent Reasoning",
                            "workshop_id": "reasoning-workshop",
                        },
                    },
                    {
                        "unit_id": "paper_abstract:paper-1",
                        "input_kind": "paper_abstract",
                        "important": True,
                        "confidence": 0.88,
                        "summary": "A paper abstract studies test-time reasoning compute.",
                        "metadata": {"title": "Reasoning Compute", "paper_id": "paper-1"},
                    },
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = runner.build_analysis_source_packs(
        workspace=workspace,
        run_id=run_id,
        stage6_receipt={"checks": {"output_inventory": {"sha256": "corpus-sha"}}},
        index_digest=index_digest,
        topic_specs=(
            {
                "id": "01_llm_reasoning_test_time",
                "title": "LLM reasoning and test-time compute",
                "keywords": ("reasoning",),
            },
        ),
    )

    assert result["source_pack_count"] == 1
    source_pack = json.loads(
        (workspace / "analysis" / "narrative_source_packs" / "01_llm_reasoning_test_time.json")
        .read_text(encoding="utf-8")
    )
    full_read = source_pack["llm_full_read_evidence"]
    assert full_read["status"] == "succeeded"
    assert full_read["unit_count"] == 2
    assert full_read["input_kind_counts"] == {
        "paper_abstract": 1,
        "transcript_segment": 1,
    }
    assert {
        item["unit_id"] for item in full_read["selected_items"]
    } == {"speech_transcript:talk-1:segment:0001", "paper_abstract:paper-1"}
    assert "speech_transcript:talk-1:segment:0001" in source_pack["extraction_record_ids"]
    query_plan = json.loads(
        (workspace / "analysis" / "provenance" / "query_plans" / "01_llm_reasoning_test_time.json")
        .read_text(encoding="utf-8")
    )
    assert query_plan["llm_full_read"]["selected_unit_ids"] == [
        "speech_transcript:talk-1:segment:0001",
        "paper_abstract:paper-1",
    ]
    source_pack["topic_reconciliation"] = {
        "status": "succeeded",
        "owner_id": "analysis.topic_reconciliation",
        "markdown_path": "analysis/topic_reconciliations/01_llm_reasoning_test_time.md",
        "reconciliation_text": "Reconciled source-pack evidence.",
    }
    (
        workspace / "analysis" / "narrative_source_packs" / "01_llm_reasoning_test_time.json"
    ).write_text(json.dumps(source_pack, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    runner.build_analysis_source_packs(
        workspace=workspace,
        run_id=run_id,
        stage6_receipt={"checks": {"output_inventory": {"sha256": "corpus-sha"}}},
        index_digest=index_digest,
        topic_specs=(
            {
                "id": "01_llm_reasoning_test_time",
                "title": "LLM reasoning and test-time compute",
                "keywords": ("reasoning",),
            },
        ),
    )
    rebuilt_source_pack = json.loads(
        (workspace / "analysis" / "narrative_source_packs" / "01_llm_reasoning_test_time.json")
        .read_text(encoding="utf-8")
    )
    assert rebuilt_source_pack["topic_reconciliation"]["reconciliation_text"] == (
        "Reconciled source-pack evidence."
    )


def test_analysis_source_packs_order_independence_main_conference_representation(
    tmp_path: Path,
) -> None:
    """Regression (Phase 10.1.1-04, Finding D on the sibling runner): the papers
    store's file order must not determine whether main-conference chunks are
    represented in a topic's source pack.

    Before this phase, ICLR2026 ran a single concatenated per-topic bucket scan
    with no per-source de-duplication and no family stratification at all --
    strictly older than even CVPR2026's pre-fix per-store scan. Workshop-scoped
    chunks are written FIRST here and alone supply enough distinct artifact_ids
    to exhaust ANALYSIS_SOURCE_PACK_MAX_CHUNKS, so this test cannot pass by the
    main-conference chunks merely happening to fit before the cap fills."""

    runner = load_runner_module()
    assert hasattr(runner, "select_topic_source_chunks"), (
        "ICLR2026 runner must delegate to the shared select_topic_source_chunks"
    )
    workspace = write_workspace(tmp_path)
    run_id = "test-analysis-source-pack-order-independence"
    vector_root = workspace / "storage" / "vector-store" / run_id
    paper_store = vector_root / "stores" / "papers"
    paper_store.mkdir(parents=True)

    workshop_chunks = [
        {
            "chunk_id": f"paper:workshop-{index}:0",
            "artifact_id": f"workshop_paper_abstract:workshop-{index}",
            "source_path": f"https://openreview.net/forum?id=workshop-{index}",
            "text": f"Reasoning workshop evidence, distinct source {index}.",
            "conference_scope": "workshop",
        }
        for index in range(8)
    ]
    main_conference_chunks = [
        {
            "chunk_id": f"paper:main-{index}:0",
            "artifact_id": f"paper_abstract:main-{index}",
            "source_path": f"https://openreview.net/forum?id=main-{index}",
            "text": f"Reasoning main-conference evidence, distinct source {index}.",
            "conference_scope": "main_conference",
        }
        for index in range(5)
    ]
    with (paper_store / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in workshop_chunks + main_conference_chunks:
            handle.write(json.dumps(chunk) + "\n")

    index_digest = {
        "stores": {
            "papers": {
                "chunks_path": runner.workspace_relpath(workspace, paper_store / "chunks.jsonl")
            },
        }
    }
    (vector_root / "index_digest.json").write_text(
        json.dumps(index_digest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    topic_specs = (
        {
            "id": "01_llm_reasoning_test_time",
            "title": "LLM reasoning and test-time compute",
            "keywords": ("reasoning",),
        },
    )

    runner.build_analysis_source_packs(
        workspace=workspace,
        run_id=run_id,
        stage6_receipt={"checks": {"output_inventory": {"sha256": "a" * 64}}},
        index_digest=index_digest,
        topic_specs=topic_specs,
    )

    source_pack = json.loads(
        (workspace / "analysis" / "narrative_source_packs" / "01_llm_reasoning_test_time.json")
        .read_text(encoding="utf-8")
    )
    chunk_ids = [chunk["chunk_id"] for chunk in source_pack["chunks"]]
    main_ids = [cid for cid in chunk_ids if cid.startswith("paper:main-")]
    workshop_ids = [cid for cid in chunk_ids if cid.startswith("paper:workshop-")]

    assert len(main_ids) >= 5, (
        f"expected at least 5 main-conference chunks despite workshop chunks "
        f"occupying every earlier file position, got {chunk_ids}"
    )
    assert workshop_ids, "workshop chunks should still be represented, not excluded entirely"
    assert len(chunk_ids) == runner.ANALYSIS_SOURCE_PACK_MAX_CHUNKS


def test_analysis_source_packs_preserve_iclr2026_schema_after_shared_selector_wiring(
    tmp_path: Path,
) -> None:
    """Schema-preservation regression: delegating chunk SELECTION to the shared
    module must not import CVPR2026's/ICML2026's serialization -- ICLR2026's
    query plan still omits paper_matching and its source pack still hardcodes
    an empty matched_papers list."""

    runner = load_runner_module()
    workspace = write_workspace(tmp_path)
    run_id = "test-analysis-source-pack-schema-preservation"
    vector_root = workspace / "storage" / "vector-store" / run_id
    paper_store = vector_root / "stores" / "papers"
    paper_store.mkdir(parents=True)
    paper_chunk = {
        "chunk_id": "paper:reasoning:0",
        "artifact_id": "paper_abstract:paper-1",
        "source_path": "https://openreview.net/forum?id=paper-1",
        "text": "Reasoning and test-time compute paper evidence.",
        "conference_scope": "main_conference",
    }
    (paper_store / "chunks.jsonl").write_text(json.dumps(paper_chunk) + "\n", encoding="utf-8")
    index_digest = {
        "stores": {
            "papers": {
                "chunks_path": runner.workspace_relpath(workspace, paper_store / "chunks.jsonl")
            },
        }
    }
    (vector_root / "index_digest.json").write_text(
        json.dumps(index_digest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    topic_specs = (
        {
            "id": "01_llm_reasoning_test_time",
            "title": "LLM reasoning and test-time compute",
            "keywords": ("reasoning",),
        },
    )

    runner.build_analysis_source_packs(
        workspace=workspace,
        run_id=run_id,
        stage6_receipt={"checks": {"output_inventory": {"sha256": "a" * 64}}},
        index_digest=index_digest,
        topic_specs=topic_specs,
    )

    source_pack = json.loads(
        (workspace / "analysis" / "narrative_source_packs" / "01_llm_reasoning_test_time.json")
        .read_text(encoding="utf-8")
    )
    query_plan = json.loads(
        (workspace / "analysis" / "provenance" / "query_plans" / "01_llm_reasoning_test_time.json")
        .read_text(encoding="utf-8")
    )
    assert source_pack["matched_papers"] == []
    assert "paper_matching" not in query_plan
    assert "paper_matching_evidence" not in source_pack


def test_exhaustive_topic_extraction_stage_writes_contract_evidence(
    tmp_path: Path,
) -> None:
    workspace = write_workspace(tmp_path)
    run_id = "test-exhaustive-topic-extraction-stage"
    write_corpus_inventory_index_fixture(workspace, run_id)
    write_topic_strategy(workspace, run_id)

    stage6 = run_runner(
        [
            "--workspace",
            str(workspace),
            "--run-id",
            run_id,
            "--stage",
            "corpus-inventory-index",
            "--mode",
            "live",
            "--resume",
            "--no-env-file",
            "--summary",
        ],
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "EMBEDDING_PROVIDER": "fake"},
    )
    assert stage6.returncode == 0, stage6.stderr

    result = run_runner(
        [
            "--workspace",
            str(workspace),
            "--run-id",
            run_id,
            "--stage",
            "exhaustive-topic-extraction",
            "--mode",
            "live",
            "--resume",
            "--no-env-file",
            "--summary",
        ],
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
    )

    assert result.returncode == 0, result.stderr
    receipt_path = (
        workspace
        / "storage"
        / "workflow-runs"
        / run_id
        / "14A-exhaustive-topic-extraction-receipt.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert receipt["status"] == "succeeded"
    assert receipt["stage_contract"]["status"] == "pass"
    assert receipt["checks"]["by_topic_count"] >= 1
    assert receipt["checks"]["by_artifact_count"] >= 1
    assert "analysis.exhaustive_topic_extraction" in receipt["required_script_ids"]


def test_corpus_inventory_index_excludes_skip_ocr_slides_from_text_index(tmp_path: Path) -> None:
    workspace = write_workspace(tmp_path)
    paths = write_corpus_inventory_index_fixture(workspace, "test-corpus-inventory-skip-ocr")

    result = run_runner(
        [
            "--workspace",
            str(workspace),
            "--run-id",
            "test-corpus-inventory-skip-ocr",
            "--stage",
            "corpus-inventory-index",
            "--mode",
            "live",
            "--resume",
            "--no-env-file",
            "--summary",
        ],
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "EMBEDDING_PROVIDER": "fake"},
    )

    assert result.returncode == 0, result.stderr
    inventory = json.loads(
        (workspace / "storage" / "corpus" / "corpus_inventory.json").read_text(encoding="utf-8")
    )
    artifact_chunks_path = (
        workspace
        / "storage"
        / "vector-store"
        / "test-corpus-inventory-skip-ocr"
        / "stores"
        / "artifacts"
        / "chunks.jsonl"
    )
    chunk_text = artifact_chunks_path.read_text(encoding="utf-8")
    skipped_records = [
        record
        for record in inventory["artifact_records"]
        if record["source_path"] == str(paths["skipped_slide"].relative_to(workspace))
    ]

    assert skipped_records
    assert skipped_records[0]["skip_ocr"] is True
    assert skipped_records[0]["skip_reason"] == "reference_or_duplicate_slide"
    assert skipped_records[0]["indexable_text"] is False
    assert skipped_records[0]["artifact_id"] not in chunk_text
    assert "skip_ocr" in json.dumps(skipped_records[0], sort_keys=True)


def write_stage8_fixture(workspace: Path, run_id: str) -> None:
    topics = [
        ("01_reasoning", "Reasoning"),
        ("02_agents", "Agents"),
        ("03_evals", "Evaluation"),
    ]
    analysis_dir = workspace / "analysis"
    (analysis_dir / "provenance").mkdir(parents=True, exist_ok=True)
    (analysis_dir / "slide_briefs").mkdir(parents=True, exist_ok=True)
    (analysis_dir / "narrative_source_packs").mkdir(parents=True, exist_ok=True)
    manifest_topics = []
    citation_topics = []
    for index, (topic_id, title) in enumerate(topics, start=1):
        markdown_rel = f"analysis/{topic_id}.md"
        markdown_path = workspace / markdown_rel
        markdown_path.write_text(
            f"# {title}\n\nSubstantial topic report body with cited evidence.\n" * 80,
            encoding="utf-8",
        )
        provenance_json_rel = f"analysis/provenance/{topic_id}.json"
        provenance_md_rel = f"analysis/provenance/{topic_id}.md"
        slide_brief_rel = f"analysis/slide_briefs/{topic_id}.md"
        (workspace / provenance_json_rel).write_text('{"schema_version": 1}\n', encoding="utf-8")
        (workspace / provenance_md_rel).write_text(f"# Provenance {topic_id}\n", encoding="utf-8")
        (workspace / slide_brief_rel).write_text(f"# Slide brief {topic_id}\n", encoding="utf-8")
        extraction_ids = [f"paper_abstract:{topic_id}:{item}" for item in range(6)]
        chunk_ids = [f"chunk:{topic_id}:{item}" for item in range(2)]
        research_ids = [f"{run_id}:{topic_id}:research"]
        manifest_topics.append(
            {
                "id": topic_id,
                "number": index,
                "slug": topic_id,
                "title": title,
                "markdown_path": markdown_rel,
                "deep_analysis_path": markdown_rel,
                "markdown_sha256": file_sha256(markdown_path),
                "provenance_json_path": provenance_json_rel,
                "provenance_path": provenance_md_rel,
                "slide_brief_path": slide_brief_rel,
                "citation_counts": {
                    "extraction_record_ids": len(extraction_ids),
                    "vector_store_chunk_ids": len(chunk_ids),
                    "research_ids": len(research_ids),
                },
                "deep_validation": {
                    "status": "pass",
                    "word_count": 2500,
                    "has_executive_summary": True,
                    "has_key_findings": True,
                    "has_references_and_sources": True,
                },
                "analysis_profile": "deep-narrative-v1",
                "narrative_model": DEFAULT_DEEP_NARRATIVE_MODEL,
            }
        )
        citation_topics.append(
            {
                "topic_id": topic_id,
                "markdown_path": markdown_rel,
                "provenance_json_path": provenance_json_rel,
                "provenance_path": provenance_md_rel,
                "slide_brief_path": slide_brief_rel,
                "extraction_record_ids": extraction_ids,
                "vector_store_chunk_ids": chunk_ids,
                "research_ids": research_ids,
                "deep_validation": {"status": "pass"},
                "narrative_model": DEFAULT_DEEP_NARRATIVE_MODEL,
            }
        )
        source_pack = {
            "schema_version": 1,
            "run_id": run_id,
            "topic_id": topic_id,
            "title": title,
            "topic_reconciliation": {
                "status": "succeeded",
                "model": DEFAULT_DEEP_NARRATIVE_MODEL,
                "markdown_path": f"analysis/topic_reconciliations/{topic_id}.md",
                "reference_validation_status": "succeeded",
                "reference_validation_counts": {
                    "citation_count": 5,
                    "ledger_entry_count": 10,
                    "invalid_reference_count": 0,
                    "unknown_reference_count": 0,
                    "title_mismatch_count": 0,
                    "bare_reference_warning_count": 0,
                },
                "reference_validation": {
                    "status": "succeeded",
                    "citation_count": 5,
                    "ledger_entry_count": 10,
                    "invalid_reference_count": 0,
                    "unknown_reference_count": 0,
                    "title_mismatch_count": 0,
                    "bare_reference_warning_count": 0,
                },
            },
        }
        (
            analysis_dir / "narrative_source_packs" / f"{topic_id}.json"
        ).write_text(json.dumps(source_pack, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        extraction_path = (
            workspace / "storage" / "extractions" / run_id / "by-topic" / f"{topic_id}.json"
        )
        extraction_path.parent.mkdir(parents=True, exist_ok=True)
        extraction_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": run_id,
                    "topic_id": topic_id,
                    "records": [
                        {
                            "record_id": extraction_ids[0],
                            "title": title,
                            "summary": "Fixture topic extraction evidence.",
                        }
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    analysis_manifest = {
        "schema_version": 1,
        "generated_at": "2026-05-27T00:00:00Z",
        "workspace": str(workspace),
        "generator": "test",
        "content_sha256": "feedface" * 8,
        "topics": manifest_topics,
    }
    (analysis_dir / "analysis_manifest.json").write_text(
        json.dumps(analysis_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (analysis_dir / "analysis_citation_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_id,
                "generated_at": "2026-05-27T00:00:00Z",
                "evidence_chain_version": "test",
                "topics": citation_topics,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    run_dir = workspace / "storage" / "workflow-runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "07D-topic-reconciliation.json").write_text(
        json.dumps({"status": "succeeded", "topic_count": len(topics)}, indent=2) + "\n",
        encoding="utf-8",
    )
    reference_checks = {
        "topic_count": len(topics),
        "completed_topic_count": len(topics),
        "ledger_entry_count": 30,
        "citation_count": 15,
        "invalid_reference_count": 0,
        "unknown_reference_count": 0,
        "title_mismatch_count": 0,
        "bare_reference_warning_count": 0,
    }
    (run_dir / "07E-topic-reference-validation.json").write_text(
        json.dumps({"status": "succeeded", "topic_count": len(topics), "checks": reference_checks})
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "07H-topic-reconciliation-reference-workflow.json").write_text(
        json.dumps(
            {
                "status": "succeeded",
                "topic_count": len(topics),
                "checks": {
                    "final_validation_status": "succeeded",
                    "final_topic_count": len(topics),
                    "final_invalid_reference_count": 0,
                    "final_unknown_reference_count": 0,
                    "final_title_mismatch_count": 0,
                    "final_bare_reference_warning_count": 0,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "stage-07-analysis.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage_id": "analysis",
                "mode": "live",
                "status": "succeeded",
                "script_registry_sha256": "test",
                "required_script_ids": [],
                "stage_contract": {"status": "pass"},
                "checks": {
                    "analysis_manifest_content_sha256": analysis_manifest["content_sha256"],
                    "analysis_manifest_topic_count": len(topics),
                    "narrative_model": DEFAULT_DEEP_NARRATIVE_MODEL,
                    "narrative_status": "succeeded",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def write_stage8_success_receipt_fixture(workspace: Path, run_id: str) -> None:
    run_dir = workspace / "storage" / "workflow-runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (workspace / "analysis" / "executive_summary.md").write_text(
        "# Executive summary\n\n" + ("summary evidence\n" * 200),
        encoding="utf-8",
    )
    (workspace / "analysis" / "trend_report.md").write_text(
        "# Trend report\n\n" + ("trend evidence\n" * 200),
        encoding="utf-8",
    )
    (run_dir / "stage-08-qa-gap-fill-synthesis.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage_id": "qa-gap-fill-synthesis",
                "mode": "live",
                "status": "succeeded",
                "script_registry_sha256": "test",
                "required_script_ids": [],
                "stage_contract": {"status": "pass"},
                "checks": {
                    "executive_summary_status": "pass",
                    "trend_report_status": "pass",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    stamp_upstream_stage_receipts(run_dir)


def write_stage9_success_receipt_fixture(workspace: Path, run_id: str) -> None:
    run_dir = workspace / "storage" / "workflow-runs" / run_id
    output_dir = workspace / "audio" / "iclr2026_overview" / "focused_podcast"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "script.json").write_text('{"segments": [{"text": "ok"}]}\n', encoding="utf-8")
    (output_dir / "script.md").write_text("# Podcast\n", encoding="utf-8")
    (output_dir / "show_notes.md").write_text("# Notes\n", encoding="utf-8")
    (output_dir / "audio_manifest.json").write_text("{}\n", encoding="utf-8")
    (output_dir / "audio_ready_package.json").write_text("{}\n", encoding="utf-8")
    (output_dir / "podcast.mp3").write_bytes(b"fake podcast")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "stage-09-podcast.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage_id": "podcast",
                "mode": "live",
                "status": "succeeded",
                "script_registry_sha256": "test",
                "required_script_ids": [],
                "stage_contract": {"status": "pass"},
                "checks": {
                    "podcast_script_status": "pass",
                    "podcast_audio_status": "pass",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    backup_dir = run_dir / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    (backup_dir / "podcast-backup-write-receipt.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "receipt_type": "conference_pipeline.gdrive_sync",
                "status": "succeeded",
                "run_id": run_id,
                "secrets_copied": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    stamp_upstream_stage_receipts(run_dir)


class FakeStage8SynthesisClient:
    provider = "anthropic"
    model = "claude-opus-4-8"

    def synthesize(self, *, kind: str, system_prompt: str, user_prompt: str) -> SimpleNamespace:
        assert kind in {"executive_summary", "trend_report"}
        assert "Citation pool" in user_prompt
        return SimpleNamespace(
            text=(f"# {kind}\n\nValidated synthesis paragraph.\n" * 120),
            input_tokens=100,
            output_tokens=50,
        )


def test_qa_gap_fill_synthesis_writes_contract_evidence(tmp_path: Path) -> None:
    runner = load_runner_module()
    workspace = write_workspace(tmp_path)
    run_id = "test-stage8"
    write_stage8_fixture(workspace, run_id)
    args = SimpleNamespace(
        workspace=workspace,
        run_id=run_id,
        mode="live",
        resume=False,
        env_file=None,
        provider_cost_approved=True,
    )

    stamp_upstream_stage_receipts(workspace / "storage" / "workflow-runs" / run_id)
    exit_code, summary = runner.run_qa_gap_fill_synthesis(
        args,
        runner.STAGES["qa-gap-fill-synthesis"],
        llm_client=FakeStage8SynthesisClient(),
    )

    assert exit_code == 0
    assert summary["status"] == "succeeded"
    receipt = json.loads(
        (
            workspace
            / "storage"
            / "workflow-runs"
            / run_id
            / "stage-08-qa-gap-fill-synthesis.json"
        ).read_text(encoding="utf-8")
    )
    assert receipt["status"] == "succeeded"
    assert receipt["stage_contract"]["status"] == "pass"
    assert "analysis.coverage_gap_fill" in receipt["required_script_ids"]
    assert "analysis.synthesis_trend_report" in receipt["required_script_ids"]
    assert receipt["checks"]["traceability_report_status"] == "pass"
    assert receipt["checks"]["coverage_gap_report_status"] == "pass"
    assert receipt["checks"]["executive_summary_status"] == "pass"
    assert receipt["checks"]["trend_report_status"] == "pass"
    assert (workspace / "analysis" / "qa" / "traceability_report.json").is_file()
    assert (workspace / "analysis" / "qa" / "coverage_gap_report.json").is_file()
    assert (workspace / "analysis" / "executive_summary.md").is_file()
    assert (workspace / "analysis" / "trend_report.md").is_file()


def test_qa_gap_fill_synthesis_blocks_missing_stage7_receipt(tmp_path: Path) -> None:
    runner = load_runner_module()
    workspace = write_workspace(tmp_path)
    run_id = "test-stage8-missing"
    write_stage8_fixture(workspace, run_id)
    (
        workspace
        / "storage"
        / "workflow-runs"
        / run_id
        / "stage-07-analysis.json"
    ).unlink()
    args = SimpleNamespace(
        workspace=workspace,
        run_id=run_id,
        mode="live",
        resume=False,
        env_file=None,
        provider_cost_approved=True,
    )

    exit_code, summary = runner.run_qa_gap_fill_synthesis(
        args,
        runner.STAGES["qa-gap-fill-synthesis"],
        llm_client=FakeStage8SynthesisClient(),
    )

    assert exit_code == 1
    assert summary["status"] == "blocked"
    receipt = json.loads(
        (
            workspace
            / "storage"
            / "workflow-runs"
            / run_id
            / "stage-08-qa-gap-fill-synthesis.json"
        ).read_text(encoding="utf-8")
    )
    assert receipt["status"] == "blocked"
    assert receipt["checks"]["blocker_code"] == "stage8_required_inputs_blocked"


def test_podcast_stage_writes_script_audio_and_contract_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    runner = load_runner_module()
    workspace = write_workspace(tmp_path)
    run_id = "test-stage9"
    write_stage8_fixture(workspace, run_id)
    args = SimpleNamespace(
        workspace=workspace,
        run_id=run_id,
        mode="live",
        resume=False,
        env_file=None,
        provider_cost_approved=True,
    )
    stamp_upstream_stage_receipts(workspace / "storage" / "workflow-runs" / run_id)
    exit_code, summary = runner.run_qa_gap_fill_synthesis(
        args,
        runner.STAGES["qa-gap-fill-synthesis"],
        llm_client=FakeStage8SynthesisClient(),
    )
    assert exit_code == 0, summary

    def fake_command_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        if "scripts/generate_focused_podcast.py" in command:
            output_dir = Path(command[command.index("--output-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            script = {
                "title": "Test podcast",
                "summary": "A source-backed test podcast.",
                "segments": [
                    {
                        "speaker": "Maya",
                        "text": " ".join(["This is a source-backed podcast segment."] * 30),
                    },
                    {
                        "speaker": "Leo",
                        "text": " ".join(["This segment continues the verified analysis."] * 30),
                    },
                ],
                "show_notes": "Generated from verified reports.",
            }
            (output_dir / "script.json").write_text(
                json.dumps(script, indent=2) + "\n", encoding="utf-8"
            )
            (output_dir / "script.md").write_text("# Test podcast\n", encoding="utf-8")
            (output_dir / "show_notes.md").write_text("# Show notes\n", encoding="utf-8")
            (output_dir / "metadata.json").write_text("{}\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "script ok", "")
        if "scripts/render_podcast_audio.py" in command:
            output_dir = Path(command[command.index("--output-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            audio_path = output_dir / "podcast.mp3"
            audio_path.write_bytes(b"fake-mp3-bytes")
            audio_manifest = {
                "schema_version": 1,
                "script_json_path": str(output_dir / "script.json"),
                "podcast_path": str(audio_path),
                "podcast_sha256": file_sha256(audio_path),
                "podcast_byte_count": audio_path.stat().st_size,
                "segment_count": 2,
                "audio_file_count": 2,
                "provider_call_count": 2,
                "estimated_cost_usd": 0.01,
                "secrets_copied": False,
            }
            (output_dir / "audio_manifest.json").write_text(
                json.dumps(audio_manifest, indent=2) + "\n", encoding="utf-8"
            )
            return subprocess.CompletedProcess(command, 0, "audio ok", "")
        raise AssertionError(command)

    monkeypatch.setattr(
        runner,
        "provider_receipt_summary",
        lambda workspace, run_id=None, stage_id=None: {
            "canonical_receipt_files_scanned": 2,
            "canonical_total_estimated_cost_usd": 0.02,
            "canonical_totals_by_provider": {"openai": 0.02},
            "canonical_totals_by_model": {"openai:gpt-4o-mini-tts": 0.02},
            "canonical_totals_by_operation": {"tts": 0.02},
            "failed_receipts": 0,
            "unknown_cost_receipts": 0,
            "receipt_paths": ["provider-call-1.json", "provider-call-2.json"],
        },
    )

    exit_code, summary = runner.run_podcast(
        args,
        runner.STAGES["podcast"],
        command_runner=fake_command_runner,
    )

    assert exit_code == 0
    assert summary["status"] == "succeeded"
    receipt_path = (
        workspace
        / "storage"
        / "workflow-runs"
        / run_id
        / "stage-09-podcast.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "succeeded"
    assert receipt["stage_contract"]["status"] == "pass"
    assert "podcast.focused_script" in receipt["required_script_ids"]
    assert "podcast.render_audio" in receipt["required_script_ids"]
    assert receipt["checks"]["podcast_script_status"] == "pass"
    assert receipt["checks"]["podcast_audio_status"] == "pass"
    assert receipt["checks"]["provider_calls_made"] is True
    assert (workspace / "audio" / "iclr2026_overview" / "focused_podcast" / "podcast.mp3").is_file()


def test_podcast_stage_blocks_missing_stage8_receipt(tmp_path: Path) -> None:
    runner = load_runner_module()
    workspace = write_workspace(tmp_path)
    args = SimpleNamespace(
        workspace=workspace,
        run_id="test-stage9-missing",
        mode="live",
        resume=False,
        env_file=None,
        provider_cost_approved=True,
    )

    exit_code, summary = runner.run_podcast(args, runner.STAGES["podcast"])

    assert exit_code == 1
    assert summary["status"] == "blocked"
    receipt = json.loads(
        (
            workspace
            / "storage"
            / "workflow-runs"
            / "test-stage9-missing"
            / "stage-09-podcast.json"
        ).read_text(encoding="utf-8")
    )
    assert receipt["status"] == "blocked"
    assert receipt["checks"]["blocker_code"] == "podcast_required_inputs_blocked"


def test_presentation_stage_writes_manifest_and_contract_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    runner = load_runner_module()
    workspace = write_workspace(tmp_path)
    run_id = "test-stage10"
    write_stage8_fixture(workspace, run_id)
    write_stage8_success_receipt_fixture(workspace, run_id)
    write_stage9_success_receipt_fixture(workspace, run_id)
    args = SimpleNamespace(
        workspace=workspace,
        run_id=run_id,
        mode="live",
        resume=False,
        env_file=None,
        provider_cost_approved=True,
    )

    def fake_paper_matching_runner(
        *, workspace_root: Path, run_id: str, topic_id: str
    ) -> dict[str, object]:
        output_dir = workspace_root / "storage" / "paper-matching" / run_id / topic_id
        output_dir.mkdir(parents=True, exist_ok=True)
        paper = {
            "rank": 1,
            "title": f"Matched Paper for {topic_id}",
            "paper_id": f"paper-{topic_id}",
            "url": "https://openreview.net/forum?id=fixture",
            "evidence_snippet": "Fixture match",
        }
        (output_dir / "artifact_gpt_researcher_findings.md").write_text(
            "# Artifact findings\n", encoding="utf-8"
        )
        (output_dir / "keypoints.json").write_text(
            json.dumps({"keypoints": ["fixture keypoint"]}) + "\n",
            encoding="utf-8",
        )
        (output_dir / "paper_gpt_researcher_findings.md").write_text(
            "# Paper findings\n", encoding="utf-8"
        )
        (output_dir / "matched_papers.json").write_text(
            json.dumps({"schema_version": 1, "topic_id": topic_id, "papers": [paper]})
            + "\n",
            encoding="utf-8",
        )
        deck_dir = workspace_root / "decks" / topic_id
        deck_dir.mkdir(parents=True, exist_ok=True)
        (workspace_root / "analysis" / "exhaustive_summaries").mkdir(
            parents=True,
            exist_ok=True,
        )
        (workspace_root / "analysis" / "exhaustive_summaries" / f"{topic_id}__papers.json").write_text(
            json.dumps({"schema_version": 1, "papers": [paper]}) + "\n",
            encoding="utf-8",
        )
        (deck_dir / "presentation_frame.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "presentation_id": topic_id,
                    "topic_id": topic_id,
                    "retrieval_title": f"Fixture retrieval title for {topic_id}",
                    "audience_title": f"Fixture audience title for {topic_id}",
                    "field_thesis": "Fixture field thesis for command planning.",
                    "source_analysis_path": f"analysis/{topic_id}.md",
                    "general_questions": [
                        "What changed in the field?",
                        "Why does it matter for technical leaders?",
                        "What evidence should shape the next decision?",
                    ],
                    "opening_anchor_policy": {
                        "rules": ["Start with the field-level frame."]
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (deck_dir / "speaker_author_work_ledger.json").write_text(
            json.dumps({"schema_version": 1, "entries": []}) + "\n",
            encoding="utf-8",
        )
        for name in (
            "source_corpus_inventory.json",
            "source_repair_findings.json",
            "research_cluster_plan.json",
            "artifact_truth_report.json",
            "presentation_review.json",
        ):
            (deck_dir / name).write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "pass",
                        "presentation_id": topic_id,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        (deck_dir / "ranked_papers.json").write_text(
            json.dumps({"schema_version": 1, "papers": [paper]}) + "\n",
            encoding="utf-8",
        )
        return {"status": "succeeded", "topics": [topic_id]}

    def arg_path(command: list[str], flag: str) -> Path:
        value = Path(command[command.index(flag) + 1])
        if value.is_absolute() or "--workspace" not in command:
            return value
        return Path(command[command.index("--workspace") + 1]) / value

    def write_slides(path: Path, *, links: int = 0, svgs: int = 0) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        link_text = "\n".join(
            f"- [Reference {index}](https://example.com/{index})"
            for index in range(links)
        )
        svg_text = "\n".join(
            f"![Figure {index}](assets/generated/figure_{index:02d}.svg)"
            for index in range(svgs)
        )
        path.write_text(
            "\n\n".join(
                [
                    "## Slide 1: Claim\n\nEvidence-backed claim.",
                    "## Slide 2: Contrast\n\nEvidence-backed contrast.",
                    "## Slide 3: Implication\n\nEvidence-backed implication.",
                    link_text,
                    svg_text,
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    def fake_command_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        script = Path(command[1]).as_posix()
        if script.endswith("generate_executive_overview_decks.py"):
            workspace_arg = arg_path(command, "--workspace")
            for presentation_id in ("executive_summary_short", "executive_summary_long"):
                deck_dir = workspace_arg / "decks" / presentation_id
                (deck_dir / "assets" / "generated").mkdir(parents=True, exist_ok=True)
                (deck_dir / "assets" / "generated" / "slide_001.svg").write_text(
                    "<svg xmlns='http://www.w3.org/2000/svg'><text font-size='20'>Executive</text></svg>\n",
                    encoding="utf-8",
                )
                (deck_dir / "6_presenter_script.md").write_text("# Script\n", encoding="utf-8")
                (deck_dir / "6_presenter_script_notebooklm_2host.md").write_text(
                    "# Script\n",
                    encoding="utf-8",
                )
                for name in (
                    "presentation_frame.json",
                    "presentation_blueprint.json",
                    "presentation_blueprint_receipt.json",
                    "blueprint_review.json",
                    "presentation_deck.json",
                    "presentation_deck_receipt.json",
                    "source_binding_report.json",
                    "visual_plan.json",
                    "render_spec.json",
                    "delivery_single_speaker.json",
                    "delivery_notebooklm.json",
                    "asset_inventory.json",
                    "figure_specs.json",
                    "figure_generation_model_receipt.json",
                    "figure_code_review_report.json",
                    "figure_svg_validation_report.json",
                    "figure_semantic_review.json",
                ):
                    (deck_dir / name).write_text(
                        json.dumps(
                            {
                                "schema_version": 1,
                                "status": "pass",
                                "presentation_id": presentation_id,
                                "slides": [{"slide_id": "slide-001"}],
                            }
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                (deck_dir / "source_attributions.json").write_text("{}\n", encoding="utf-8")
                (deck_dir / "source_pack_receipt.json").write_text(
                    json.dumps({"status": "succeeded"}) + "\n",
                    encoding="utf-8",
                )
            arg_path(command, "--receipt-output").write_text(
                json.dumps({"status": "succeeded"}) + "\n",
                encoding="utf-8",
            )
        elif script.endswith("generate_editorial_blueprint.py") or script.endswith("generate_presentation_blueprint.py"):
            if "--output-dir" in command:
                output_dir = arg_path(command, "--output-dir")
                blueprint_path = output_dir / "presentation_blueprint.json"
                receipt_path = output_dir / "presentation_blueprint_receipt.json"
                md_path = output_dir / "presentation_blueprint.md"
            else:
                blueprint_path = arg_path(command, "--output-json")
                receipt_path = arg_path(command, "--receipt-path")
                md_path = arg_path(command, "--output-md")
                output_dir = blueprint_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            blueprint_payload = json.loads(
                (Path("tests/fixtures/presentation_blueprint/valid_blueprint.json")).read_text(
                    encoding="utf-8"
                )
            )
            blueprint_payload["blueprint_id"] = command[command.index("--presentation-id") + 1]
            blueprint_path.write_text(
                json.dumps(blueprint_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            blueprint_sha = file_sha256(blueprint_path)
            md_path.write_text("# Presentation Blueprint\n", encoding="utf-8")
            receipt_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "mode": "live",
                        "production_evidence": True,
                        "blueprint_sha256": blueprint_sha,
                        "provider_receipt_paths": ["storage/provider-calls/fixture.json"],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        elif script.endswith("review_presentation_blueprint.py"):
            blueprint_path = arg_path(command, "--presentation-blueprint")
            blueprint_sha = file_sha256(blueprint_path)
            arg_path(command, "--blueprint-review").write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "approval": {"status": "approved_for_draft_generation"},
                        "blueprint_sha256": blueprint_sha,
                        "approved_at": "2026-06-06T00:00:00Z",
                        "approved_by": "fixture-reviewer",
                        "review_artifact": "blueprint_review.md",
                        "gate_types": {
                            "deterministic": "pass",
                            "LLM-judge": "pass",
                            "human": "pass",
                        },
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            arg_path(command, "--output-dir").joinpath("blueprint_review.md").write_text(
                "# Blueprint Review\n", encoding="utf-8"
            )
        elif script.endswith("generate_slides_agent.py"):
            write_slides(arg_path(command, "--output-slides-md"))
        elif script.endswith("generate_deck_contract.py"):
            arg_path(command, "--output").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "presentation_id": command[command.index("--presentation-id") + 1],
                        "slides": [{"slide_id": "slide-001", "title": "Claim"}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            arg_path(command, "--receipt-output").write_text(
                json.dumps({"status": "succeeded", "production_evidence": True}) + "\n",
                encoding="utf-8",
            )
        elif script.endswith("build_deck_source_pack.py"):
            output_dir = arg_path(command, "--output-dir")
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "extracted_content.json").write_text("{}\n", encoding="utf-8")
            (output_dir / "source_attributions.json").write_text("{}\n", encoding="utf-8")
            (output_dir / "ranked_papers.json").write_text(
                json.dumps({"schema_version": 1, "papers": []}) + "\n",
                encoding="utf-8",
            )
            (output_dir / "source_pack_receipt.json").write_text(
                json.dumps({"status": "succeeded"}) + "\n", encoding="utf-8"
            )
        elif script.endswith("bind_presentation_sources.py"):
            arg_path(command, "--output").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "presentation_id": command[command.index("--presentation-id") + 1],
                        "slides": [{"slide_id": "slide-001", "visible_references": []}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif script.endswith("enrich_slides_with_papers.py"):
            write_slides(arg_path(command, "--output-slides"), links=3)
        elif script.endswith("reflow_slides.py"):
            write_slides(arg_path(command, "--output-slides"), links=3)
        elif script.endswith("sanity_check_slides.py"):
            write_slides(arg_path(command, "--output-slides"), links=3)
            arg_path(command, "--report-json").write_text("{}\n", encoding="utf-8")
        elif script.endswith("enrich_hyperlinks_hybrid.py"):
            write_slides(arg_path(command, "--output-slides"), links=6)
        elif script.endswith("consolidate_slides.py"):
            write_slides(arg_path(command, "--output-slides"), links=6)
        elif script.endswith("generate_programmatic_figures.py"):
            generated_dir = arg_path(command, "--generated-dir")
            generated_dir.mkdir(parents=True, exist_ok=True)
            for index in range(3):
                (generated_dir / f"figure_{index:02d}.svg").write_text(
                    "<svg xmlns='http://www.w3.org/2000/svg'></svg>\n",
                    encoding="utf-8",
                )
            if "--output-slides" in command:
                write_slides(arg_path(command, "--output-slides"), links=6, svgs=3)
            if "--visual-plan" in command:
                arg_path(command, "--visual-plan").write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "presentation_id": command[command.index("--presentation-id") + 1],
                            "slides": [{"slide_id": "slide-001", "figures": []}],
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
            if "--figure-specs-output" in command:
                arg_path(command, "--figure-specs-output").write_text(
                    json.dumps({"schema_version": 1, "figures": []}) + "\n",
                    encoding="utf-8",
                )
            arg_path(command, "--asset-inventory").write_text(
                json.dumps({"assets": []}) + "\n", encoding="utf-8"
            )
        elif script.endswith("generate_render_spec.py"):
            arg_path(command, "--output").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "presentation_id": command[command.index("--presentation-id") + 1],
                        "slides": [{"slide_id": "slide-001", "speaker_notes": "Presenter notes."}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif script.endswith("validate_svgs.py"):
            arg_path(command, "--report-output").write_text("# SVG validation\n", encoding="utf-8")
        elif script.endswith("add_title_slide_to_slides_md.py"):
            pass
        elif script.endswith("md_to_revealjs.py"):
            arg_path(command, "--output-html").write_text(
                "<html><body><section data-markdown-slide='0'></section></body></html>\n",
                encoding="utf-8",
            )
            arg_path(command, "--slide-mapping").write_text(
                json.dumps({"slides": [{"slide_number": 1}]}) + "\n",
                encoding="utf-8",
            )
        elif script.endswith("inspect_rendered_slides.py"):
            screenshots_dir = arg_path(command, "--screenshots-dir")
            screenshots_dir.mkdir(parents=True, exist_ok=True)
            (screenshots_dir / "slide_001.png").write_bytes(b"png")
            arg_path(command, "--report-output").write_text("# Visual QC\n", encoding="utf-8")
        elif script.endswith("inspect_rendered_slides_llm.py"):
            arg_path(command, "--report-output").write_text(
                json.dumps({"status": "passed", "errors": []}) + "\n",
                encoding="utf-8",
            )
        elif script.endswith("run_presentation_repair_loop.py"):
            arg_path(command, "--receipt-output").write_text(
                json.dumps({"status": "passed"}) + "\n", encoding="utf-8"
            )
            arg_path(command, "--repair-feedback-output").write_text("{}\n", encoding="utf-8")
        elif script.endswith("generate_presenter_script.py"):
            if "--delivery-single-speaker" in command:
                arg_path(command, "--delivery-single-speaker").write_text(
                    json.dumps({"schema_version": 1, "delivery_variant": "single_speaker"})
                    + "\n",
                    encoding="utf-8",
                )
                arg_path(command, "--delivery-notebooklm").write_text(
                    json.dumps({"schema_version": 1, "delivery_variant": "notebooklm"})
                    + "\n",
                    encoding="utf-8",
                )
                if "--debug-script-md" in command:
                    arg_path(command, "--debug-script-md").write_text(
                        "### [SLIDE 1]\nPresenter notes.\n", encoding="utf-8"
                    )
                if "--debug-notebooklm-script-md" in command:
                    arg_path(command, "--debug-notebooklm-script-md").write_text(
                        "### [SLIDE 1]\nNotebook notes.\n", encoding="utf-8"
                    )
            else:
                arg_path(command, "--output-script-md").write_text(
                    "### [SLIDE 1]\nPresenter notes.\n", encoding="utf-8"
                )
                arg_path(command, "--output-notebooklm-script-md").write_text(
                    "### [SLIDE 1]\nNotebook notes.\n", encoding="utf-8"
                )
        elif script.endswith("build_presenter_video.py"):
            arg_path(command, "--build-dir").mkdir(parents=True, exist_ok=True)
            (arg_path(command, "--build-dir") / "script.json").write_text("{}\n", encoding="utf-8")
            arg_path(command, "--output").write_bytes(b"video")
        elif script.endswith("build_notebooklm_style_presenter_video.py"):
            arg_path(command, "--build-dir").mkdir(parents=True, exist_ok=True)
            (arg_path(command, "--build-dir") / "script.json").write_text("{}\n", encoding="utf-8")
            arg_path(command, "--output").write_bytes(b"video")
            if "--podcast-script-out" in command:
                arg_path(command, "--podcast-script-out").write_text("Notebook notes.\n", encoding="utf-8")
        elif script.endswith("validate_video_sync.py"):
            arg_path(command, "--single-speaker-report").write_text(
                json.dumps({"schema_version": 1, "status": "passed"}) + "\n",
                encoding="utf-8",
            )
            arg_path(command, "--notebooklm-report").write_text(
                json.dumps({"schema_version": 1, "status": "passed"}) + "\n",
                encoding="utf-8",
            )
        elif script.endswith("validate_video_delivery.py"):
            arg_path(command, "--report-out").write_text(
                json.dumps({"schema_version": 1, "status": "pass", "check_counts": {"pass": 4}, "checks": []})
                + "\n",
                encoding="utf-8",
            )
        elif script.endswith("md_to_pptx.py"):
            arg_path(command, "--output-pdf").write_bytes(b"%PDF-1.4\n/Type /Pages /Count 1\n%%EOF\n")
        elif script.endswith("export_revealjs_pdf.py"):
            arg_path(command, "--output-pdf").write_bytes(b"%PDF-1.4\n/Type /Pages /Count 1\n%%EOF\n")
            if "--frame-manifest" in command:
                arg_path(command, "--frame-manifest").write_text(
                    json.dumps({"schema_version": 1, "frames": []}) + "\n",
                    encoding="utf-8",
                )
            if "--contact-sheet" in command:
                arg_path(command, "--contact-sheet").write_bytes(b"png")
            if "--receipt-output" in command:
                arg_path(command, "--receipt-output").write_text(
                    json.dumps({"schema_version": 1, "status": "pass"}) + "\n",
                    encoding="utf-8",
                )
        elif script.endswith("audit_presentation_links.py"):
            arg_path(command, "--receipt-output").write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "artifact_count": 4,
                        "issue_count": 0,
                        "issues": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            raise AssertionError(command)
        return subprocess.CompletedProcess(command, 0, "ok", "")

    def fake_validation_runner(
        *, workspace: Path, run_id: str, topics: list[dict[str, str]]
    ) -> dict[str, object]:
        report_path = workspace / "storage" / "workflow-runs" / run_id / "presentation-validation-report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps({"status": "pass", "topics": [topic["id"] for topic in topics]})
            + "\n",
            encoding="utf-8",
        )
        return {
            "status": "pass",
            "report_path": str(report_path.relative_to(workspace)),
            "report_sha256": file_sha256(report_path),
            "record_count": len(topics),
            "totals": {"pass": len(topics)},
        }

    monkeypatch.setattr(
        runner,
        "provider_receipt_summary",
        lambda workspace, run_id=None, stage_id=None: {
            "canonical_receipt_files_scanned": 1,
            "canonical_total_estimated_cost_usd": 0.05,
            "failed_receipts": 0,
            "unknown_cost_receipts": 0,
            "receipt_paths": ["provider-call.json"],
        },
    )

    exit_code, summary = runner.run_presentation(
        args,
        runner.STAGES["presentation"],
        command_runner=fake_command_runner,
        paper_matching_runner=fake_paper_matching_runner,
        validation_runner=fake_validation_runner,
    )

    assert exit_code == 0
    assert summary["status"] == "succeeded"
    receipt = json.loads(
        (
            workspace
            / "storage"
            / "workflow-runs"
            / run_id
            / "stage-10-presentation.json"
        ).read_text(encoding="utf-8")
    )
    assert receipt["status"] == "succeeded"
    assert receipt["stage_contract"]["status"] == "pass"
    assert "presentation.generate_deck_contract" in receipt["required_script_ids"]
    assert "presentation.bind_presentation_sources" in receipt["required_script_ids"]
    assert "presentation.generate_render_spec" in receipt["required_script_ids"]
    assert receipt["checks"]["link_audit_status"] == "pass"
    assert receipt["checks"]["link_audit"]["issue_count"] == 0
    assert "presentation.executive_overview_decks" in receipt["required_script_ids"]
    assert "presentation.repair_loop" in receipt["required_script_ids"]
    assert "presentation.export_pdf" in receipt["required_script_ids"]
    assert receipt["checks"]["presentation_manifest_status"] == "pass"
    assert receipt["checks"]["paper_matching"]["status"] == "pass"
    assert receipt["checks"]["deck_generation_status"] == "pass"
    assert "presentation.deck_contract_generation" in receipt["checks"]["model_governance_task_ids"]
    assert "presentation.source_binding" in receipt["checks"]["model_governance_task_ids"]
    assert receipt["checks"]["validation_status"] == "pass"
    assert receipt["checks"]["completed_topic_count"] == receipt["checks"]["topic_count"]
    assert (
        receipt["checks"]["completed_executive_overview_count"]
        == receipt["checks"]["executive_overview_count"]
    )
    assert (workspace / "decks" / "presentation_manifest.json").is_file()
    assert (workspace / "decks" / "01_reasoning" / "presentation.pdf").is_file()
    assert (workspace / "decks" / "executive_summary_short" / "presentation.pdf").is_file()
    assert (workspace / "decks" / "executive_summary_long" / "presentation.pdf").is_file()


def test_presentation_stage_blocks_missing_stage9_receipt(tmp_path: Path) -> None:
    runner = load_runner_module()
    workspace = write_workspace(tmp_path)
    run_id = "test-stage10-missing-stage9"
    write_stage8_fixture(workspace, run_id)
    write_stage8_success_receipt_fixture(workspace, run_id)
    args = SimpleNamespace(
        workspace=workspace,
        run_id=run_id,
        mode="live",
        resume=False,
        env_file=None,
        provider_cost_approved=True,
    )

    exit_code, summary = runner.run_presentation(args, runner.STAGES["presentation"])

    assert exit_code == 1
    assert summary["status"] == "blocked"
    receipt = json.loads(
        (
            workspace
            / "storage"
            / "workflow-runs"
            / run_id
            / "stage-10-presentation.json"
        ).read_text(encoding="utf-8")
    )
    assert receipt["status"] == "blocked"
    assert receipt["checks"]["blocker_code"] == "presentation_required_inputs_blocked"
    assert {
        blocker["code"] for blocker in receipt["checks"]["input_blockers"]
    } >= {
        "presentation_stage9_receipt_missing_or_malformed",
        "presentation_podcast_backup_missing_or_not_succeeded",
    }


def test_presentation_resume_command_skip_requires_material_outputs(tmp_path: Path) -> None:
    runner = load_runner_module()
    workspace = write_workspace(tmp_path)
    output = workspace / "decks" / "01_demo_topic" / "1_draft_slides.md"
    command = [
        sys.executable,
        "scripts/presentations/generate_slides_agent.py",
        "--workspace",
        str(workspace),
        "--output-slides-md",
        str(output),
    ]

    complete, relpaths = runner.presentation_command_is_resume_complete(
        command,
        workspace=workspace,
    )
    assert complete is False
    assert relpaths == ["decks/01_demo_topic/1_draft_slides.md"]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")
    complete, _relpaths = runner.presentation_command_is_resume_complete(
        command,
        workspace=workspace,
    )
    assert complete is False

    output.write_text("## Slide 1: Demo\n", encoding="utf-8")
    # An output that exists is not evidence it is current: only a ledger
    # record of the run that produced it is (v5.1 step lineage).
    complete, relpaths = runner.presentation_command_is_resume_complete(
        command,
        workspace=workspace,
    )
    assert complete is False
    runner.presentation_record_command(command, workspace=workspace)
    assert (workspace / "decks" / "01_demo_topic" / "step_lineage.json").is_file()
    complete, relpaths = runner.presentation_command_is_resume_complete(
        command,
        workspace=workspace,
    )
    assert complete is True
    assert relpaths == ["decks/01_demo_topic/1_draft_slides.md"]

    input_path = workspace / "analysis" / "slide_briefs" / "01_demo_topic.md"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text("# newer input\n", encoding="utf-8")
    stale_command = [
        *command,
        "--analysis-md",
        str(input_path),
    ]
    complete, relpaths = runner.presentation_command_is_resume_complete(
        stale_command,
        workspace=workspace,
    )
    assert complete is False
    assert relpaths == ["decks/01_demo_topic/1_draft_slides.md"]


def test_presentation_resume_command_skip_rejects_changed_input_digest_not_mtime(tmp_path: Path) -> None:
    """The ledger binds input CONTENT: touching an input's mtime keeps the
    command reusable, changing its bytes does not; a hand-edited output also
    re-runs the command."""

    import os

    runner = load_runner_module()
    workspace = write_workspace(tmp_path)
    input_path = workspace / "analysis" / "slide_briefs" / "01_demo_topic.md"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text("# brief\n", encoding="utf-8")
    output = workspace / "decks" / "01_demo_topic" / "1_draft_slides.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "scripts/presentations/generate_slides_agent.py",
        "--workspace",
        str(workspace),
        "--analysis-md",
        str(input_path),
        "--output-slides-md",
        str(output),
    ]
    output.write_text("## Slide 1: Demo\n", encoding="utf-8")
    runner.presentation_record_command(command, workspace=workspace)
    assert runner.presentation_command_is_resume_complete(command, workspace=workspace)[0] is True

    stat = input_path.stat()
    os.utime(input_path, ns=(stat.st_atime_ns + 10_000_000_000, stat.st_mtime_ns + 10_000_000_000))
    assert runner.presentation_command_is_resume_complete(command, workspace=workspace)[0] is True

    input_path.write_text("# brief, revised\n", encoding="utf-8")
    assert runner.presentation_command_is_resume_complete(command, workspace=workspace)[0] is False

    runner.presentation_record_command(command, workspace=workspace)
    assert runner.presentation_command_is_resume_complete(command, workspace=workspace)[0] is True
    output.write_text("## Slide 1: hand edited\n", encoding="utf-8")
    assert runner.presentation_command_is_resume_complete(command, workspace=workspace)[0] is False


def test_qa_gap_fill_synthesis_resume_reruns_when_stage7_output_changed(tmp_path: Path) -> None:
    """v5.1: an unchanged upstream reuses the receipt; a moved upstream
    output_digest re-runs the stage and preserves the prior receipt."""

    runner = load_runner_module()
    workspace = write_workspace(tmp_path)
    run_id = "test-stage8-lineage"
    write_stage8_fixture(workspace, run_id)
    run_dir = workspace / "storage" / "workflow-runs" / run_id
    stamp_upstream_stage_receipts(run_dir)
    args = SimpleNamespace(workspace=workspace, run_id=run_id, mode="live", resume=False, env_file=None, provider_cost_approved=True)
    exit_code, _summary = runner.run_qa_gap_fill_synthesis(args, runner.STAGES["qa-gap-fill-synthesis"], llm_client=FakeStage8SynthesisClient())
    assert exit_code == 0
    receipt_path = run_dir / "stage-08-qa-gap-fill-synthesis.json"
    first = json.loads(receipt_path.read_text(encoding="utf-8"))
    stage7_key = f"storage/workflow-runs/{run_id}/stage-07-analysis.json"
    assert first["lineage"]["upstream_receipts"][stage7_key]["digest_kind"] == "output_digest"

    resume_args = SimpleNamespace(**{**vars(args), "resume": True})
    exit_code, _summary = runner.run_qa_gap_fill_synthesis(resume_args, runner.STAGES["qa-gap-fill-synthesis"], llm_client=FakeStage8SynthesisClient())
    assert exit_code == 0
    assert json.loads(receipt_path.read_text(encoding="utf-8"))["started_at"] == first["started_at"], "unchanged upstream: reused"

    stage7 = json.loads((run_dir / "stage-07-analysis.json").read_text(encoding="utf-8"))
    stage7["output_digest"] = "4" * 64
    (run_dir / "stage-07-analysis.json").write_text(json.dumps(stage7, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    exit_code, _summary = runner.run_qa_gap_fill_synthesis(resume_args, runner.STAGES["qa-gap-fill-synthesis"], llm_client=FakeStage8SynthesisClient())
    assert exit_code == 0
    third = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert third["lineage"]["upstream_receipts"][stage7_key]["digest"] == "4" * 64, "changed upstream: re-ran"
    assert third["lineage"]["lineage_digest"] != first["lineage"]["lineage_digest"]
    assert list(run_dir.glob("stage-08-qa-gap-fill-synthesis.superseded-*.json"))


def test_qa_gap_fill_synthesis_blocks_before_spend_when_stage7_is_itself_stale(tmp_path: Path) -> None:
    """v5.1 pre-spend gate: Stage 7's own upstream moved, so Stage 8 blocks
    before its paid client runs, and without a run-wide blocker receipt that
    would also halt the Stage 7 re-run that clears it."""

    runner = load_runner_module()
    workspace = write_workspace(tmp_path)
    run_id = "test-stage8-upstream-stale"
    write_stage8_fixture(workspace, run_id)
    run_dir = workspace / "storage" / "workflow-runs" / run_id
    stamp_upstream_stage_receipts(run_dir)
    stage7_path = run_dir / "stage-07-analysis.json"
    stage7 = json.loads(stage7_path.read_text(encoding="utf-8"))
    stage7["lineage"] = {
        **stage7["lineage"],
        "upstream_receipts": {
            f"storage/workflow-runs/{run_id}/14A-exhaustive-topic-extraction-receipt.json": {
                "stage_id": "exhaustive-topic-extraction",
                "digest_kind": "output_digest",
                "digest": "0" * 64,
            }
        },
    }
    stage7_path.write_text(json.dumps(stage7, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    class _MustNotBeCalled(FakeStage8SynthesisClient):
        def __getattribute__(self, name: str):
            if name in {"provider", "model"}:
                return object.__getattribute__(self, name)
            raise AssertionError(f"paid client used over a stale upstream: {name}")

    args = SimpleNamespace(workspace=workspace, run_id=run_id, mode="live", resume=True, env_file=None, provider_cost_approved=True)
    exit_code, summary = runner.run_qa_gap_fill_synthesis(args, runner.STAGES["qa-gap-fill-synthesis"], llm_client=_MustNotBeCalled())

    assert exit_code == 1
    receipt = json.loads((run_dir / "stage-08-qa-gap-fill-synthesis.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "blocked"
    assert receipt["checks"]["blocker_code"] == "upstream_lineage_not_fresh"
    gate = receipt["checks"]["upstream_lineage_blockers"]
    assert [(item["code"], item["upstream_stage_id"]) for item in gate] == [("stage_contract_upstream_stale", "analysis")]
    assert receipt["blocker_receipt_path"] is None
    assert runner.active_blockers(workspace, run_id=run_id) == [], "no sticky run-wide blocker"
    assert summary["halt_stage_id"] == "qa-gap-fill-synthesis"
