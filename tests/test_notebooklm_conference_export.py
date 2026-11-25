import hashlib
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/package_conference_for_notebook_lm_export.py"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_conference_notebooklm_export_uses_clean_markdown_without_dereferencing(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "conferences/demo2026"

    write_json(
        workspace / "storage/download-corpus/openreview_accepted_papers.json",
        {
            "papers": [
                {
                    "id": "PAPER123",
                    "title": "Sparse Reward Reasoning",
                    "abstract": "A useful abstract about verifier-guided reasoning.",
                    "authors": ["A. Author", "B. Author"],
                    "forum_url": "https://openreview.net/forum?id=PAPER123",
                    "primary_area": "reasoning",
                    "keywords": ["rlvr", "reasoning"],
                }
            ]
        },
    )

    write_json(
        workspace / "storage/workshops/talk_media_inventory.json",
        {
            "records": [
                {
                    "talk_id": "talk_media_001_abc123",
                    "talk_title": "Scaling Test-Time Exploration",
                    "talk_speaker": "Dr. Ada",
                    "workshop_id": "ICLR.cc/2026/Workshop/Demo",
                    "workshop_title": "Demo Workshop",
                    "local_path": "storage/workshops/talk-media/Demo/embedded-audio/talk_media_001_abc123-raw.mp3",
                }
            ]
        },
    )
    write_json(
        workspace / "storage/workshops/inventory.json",
        {
            "workshops": [
                {
                    "venueid": "ICLR.cc/2026/Workshop/Demo",
                    "title": "Demo Workshop",
                }
            ]
        },
    )
    write_json(
        workspace
        / "storage/workshops/papers/demo/receipts/WORKSHOP456.json",
        {
            "note_id": "WORKSHOP456",
            "venueid": "ICLR.cc/2026/Workshop/Demo",
            "title": "Sparse Reward Reasoning",
            "authors": ["Workshop Author"],
            "pdf_path": "storage/workshops/papers/demo/WORKSHOP456.pdf",
            "status": "succeeded",
        },
    )

    transcript = (
        workspace
        / "storage/transcription-ocr/speech-transcripts/speech-talk_media_001_abc123-raw-feedface.txt"
    )
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text("This is the workshop transcript.", encoding="utf-8")
    write_json(
        transcript.with_suffix(".json"),
        {
            "provider": "openai",
            "provider_model": "gpt-4o-transcribe-diarize",
            "source_path": "storage/workshops/talk-media/Demo/embedded-audio/talk_media_001_abc123-raw.mp3",
            "source_sha256": "a" * 64,
            "transcript_sha256": "b" * 64,
        },
    )

    main_transcript = (
        workspace
        / "storage/transcription-ocr/speech-transcripts/speech-raw-mainposter.txt"
    )
    main_transcript.write_text("This is the main conference transcript.", encoding="utf-8")
    write_json(
        main_transcript.with_suffix(".json"),
        {
            "source_path": "cache/acquisition-downloads/media_asset/poster-10000999/slideslive_audio/raw.mp3",
            "source_sha256": "c" * 64,
            "transcript_sha256": "d" * 64,
        },
    )
    page_probe = (
        workspace
        / "cache/acquisition-downloads/item-page-probes/virtual-2026-poster-10000999-demo.html"
    )
    page_probe.parent.mkdir(parents=True, exist_ok=True)
    page_probe.write_text(
        """
        <script type="application/ld+json">
        {
          "@type": "CreativeWork",
          "name": "Main Conference Talk",
          "author": [{"name": "Prof. Turing"}, {"name": "Dr. Hopper"}]
        }
        </script>
        """,
        encoding="utf-8",
    )

    speech_hash = "abcdefabcdefabcdefabcdef"
    analysis = workspace / "analysis/01_reasoning.md"
    analysis.parent.mkdir(parents=True, exist_ok=True)
    analysis.write_text(
        "\n".join(
            [
                "# Reasoning",
                "",
                "Paper ref: `paper_abstract:PAPER123`.",
                f"Talk ref: `speech_transcript:{speech_hash}:segment:00000`, `00001`.",
                "Segment shorthand ref: `segment:00001`.",
                f"Bare talk ref: `{speech_hash}:segment:00001`.",
            ]
        ),
        encoding="utf-8",
    )
    write_json(
        workspace / "analysis/topic_reference_ledgers/01_reasoning.json",
        {
            "entries": [
                {
                    "canonical_id": f"speech_transcript:{speech_hash}:segment:00000",
                    "bare_id": f"{speech_hash}:segment:00000",
                    "source_labels": ["Scaling Test-Time Exploration"],
                    "titles": [],
                },
                {
                    "canonical_id": f"speech_transcript:{speech_hash}:segment:00001",
                    "bare_id": f"{speech_hash}:segment:00001",
                    "source_labels": ["Scaling Test-Time Exploration"],
                    "titles": [],
                },
            ]
        },
    )

    output_dir = tmp_path / "export"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--workspace",
            str(workspace),
            "--output-dir",
            str(output_dir),
            "--skip-corpus-inventory",
        ],
        check=True,
        cwd=REPO_ROOT,
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["speech_transcripts"] == 2
    assert manifest["counts"]["main_conference_papers"] == 1
    assert manifest["counts"]["workshop_papers"] == 1
    assert manifest["counts"]["analysis_documents"] == 1
    assert manifest["counts"]["reference_map_entries"] == 0

    transcript_export = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((output_dir / "notebooklm").glob("transcripts_part_*.md"))
    )
    assert "# Scaling Test-Time Exploration" in transcript_export
    assert "Author: Dr. Ada" in transcript_export
    assert "# Main Conference Talk" in transcript_export
    assert "Author: Prof. Turing, Dr. Hopper" in transcript_export
    assert "This is the workshop transcript." in transcript_export
    assert "This is the main conference transcript." in transcript_export
    assert "Provider:" not in transcript_export
    assert "Transcript path:" not in transcript_export

    paper_export = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((output_dir / "notebooklm").glob("paper_abstracts_part_*.md"))
    )
    assert "Source kind: main_conference" in paper_export
    assert "Source kind: workshop" in paper_export
    assert "Workshop: Demo Workshop" in paper_export
    assert "A useful abstract about verifier-guided reasoning." in paper_export

    analysis_export = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((output_dir / "notebooklm").glob("analysis_docs_part_*.md"))
    )
    assert "## Dereferenced References" not in analysis_export
    assert "`paper_abstract:PAPER123`" in analysis_export
    assert f"`speech_transcript:{speech_hash}:segment:00000`" in analysis_export
    assert "`00001`" in analysis_export
    assert "`segment:00001`" in analysis_export
    assert f"`{speech_hash}:segment:00001`" in analysis_export


def test_conference_notebooklm_export_full_text_mode_groups_by_topic(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "conferences/demo2026"
    pdf_hash = "a" * 40

    write_json(
        workspace / "storage/download-corpus/openreview_accepted_papers.json",
        {
            "papers": [
                {
                    "id": "PAPER123",
                    "title": "Sparse Reward Reasoning",
                    "abstract": "An abstract.",
                    "authors": ["A. Author"],
                    "forum_url": "https://openreview.net/forum?id=PAPER123",
                    "primary_area": "reasoning",
                    "keywords": ["rlvr"],
                    "venue": "ICLR 2026",
                    "pdf_url": f"https://openreview.net/pdf/{pdf_hash}.pdf",
                }
            ]
        },
    )

    # Extracted full text with boilerplate, a hyphenated break, a references
    # list (should be dropped), and an appendix (should be kept).
    text_path = (
        workspace
        / "storage/transcription-ocr/text-extraction/main/paper_text_extraction-demo.txt"
    )
    text_path.parent.mkdir(parents=True, exist_ok=True)
    paper_text = "\n".join(
        [
            "Published as a conference paper at ICLR 2026",
            "Sparse Reward Reasoning",
            "",
            "We study verifier-guided reason-",
            "ing with sparse rewards in detail.",
            "",
            "References",
            "[1] Some Author. A cited paper. 2024.",
            "[2] Another Author. Another cited paper. 2023.",
            "",
            "Appendix A: Extra Proofs",
            "The appendix retains the supplementary derivations.",
        ]
    )
    text_path.write_bytes(paper_text.encode("utf-8"))
    # The on-disk path is retention-redacted in real artifacts, so the export
    # joins extracted text to its record by content hash (text_sha256).
    text_sha = hashlib.sha256(paper_text.encode("utf-8")).hexdigest()
    write_json(
        workspace
        / "storage/extractions/demo-run/by-artifact/paper_text_demo.json",
        {
            "artifact_id": "paper_text:demo",
            "source_artifact_id": "paper_text:demo",
            "source_artifact_path": (
                f"cache/acquisition-downloads/paper_pdf/openreview-venue/{pdf_hash}-abc.pdf"
            ),
            "topic_ids": ["01_reasoning_post_training", "03_evaluation_benchmarks_data"],
            "quality_metadata": {
                "source_text_path": (
                    "storage/transcription-ocr/text-extraction/main/REDACTED.txt"
                ),
                "text_sha256": text_sha,
                "topic_scores": {
                    "01_reasoning_post_training": 153.0,
                    "03_evaluation_benchmarks_data": 12.0,
                },
            },
        },
    )

    output_dir = tmp_path / "export"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--workspace",
            str(workspace),
            "--output-dir",
            str(output_dir),
            "--paper-mode",
            "full",
        ],
        check=True,
        cwd=REPO_ROOT,
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["paper_mode"] == "full"
    assert manifest["counts"]["full_text_main_papers"] == 1
    assert manifest["counts"]["papers_full_source_files"] == 1
    # Abstract parts must not be produced in full-only mode.
    assert not list((output_dir / "notebooklm").glob("paper_abstracts_part_*.md"))

    full_parts = list(
        (output_dir / "notebooklm").glob("papers_full_01_reasoning_post_training_part_*.md")
    )
    assert len(full_parts) == 1
    full_export = full_parts[0].read_text(encoding="utf-8")

    # Clean metadata title is used (joined via PDF hash).
    assert "# Paper: Sparse Reward Reasoning" in full_export
    # No LLM full-read record here, so the confidence-ranked heuristic primary
    # (topic_ids[0]) is used.
    assert "Primary topic: Reasoning & Post-training" in full_export
    assert "Topic source: heuristic_confidence" in full_export
    # Hyphenated line break is rejoined.
    assert "verifier-guided reasoning with sparse rewards" in full_export
    # Boilerplate header and references list are stripped.
    assert "Published as a conference paper" not in full_export
    assert "A cited paper" not in full_export
    # Appendix is retained.
    assert "The appendix retains the supplementary derivations." in full_export


def test_full_text_mode_prefers_llm_full_read_topic(tmp_path: Path) -> None:
    workspace = tmp_path / "conferences/demo2026"
    pdf_hash = "b" * 40

    write_json(
        workspace / "storage/download-corpus/openreview_accepted_papers.json",
        {
            "papers": [
                {
                    "id": "PAPERLLM",
                    "forum": "PAPERLLM",
                    "title": "Diffusion Planning",
                    "abstract": "An abstract.",
                    "pdf_url": f"https://openreview.net/pdf/{pdf_hash}.pdf",
                }
            ]
        },
    )

    text_path = (
        workspace
        / "storage/transcription-ocr/text-extraction/main/paper_text_extraction-llm.txt"
    )
    text_path.parent.mkdir(parents=True, exist_ok=True)
    paper_text = "Diffusion Planning\n\nWe study generative planning in detail."
    text_path.write_bytes(paper_text.encode("utf-8"))
    text_sha = hashlib.sha256(paper_text.encode("utf-8")).hexdigest()

    # Heuristic record ranks reasoning first; the LLM full-read record assigns
    # generative models. The LLM assignment must win.
    write_json(
        workspace / "storage/extractions/demo-run/by-artifact/paper_text_llm.json",
        {
            "artifact_id": "paper_text:llm",
            "source_artifact_id": "paper_text:llm",
            "source_artifact_path": (
                f"cache/acquisition-downloads/paper_pdf/openreview-venue/{pdf_hash}-abc.pdf"
            ),
            "topic_ids": ["01_reasoning_post_training"],
            "quality_metadata": {
                "text_sha256": text_sha,
                "topic_scores": {"01_reasoning_post_training": 99.0},
            },
        },
    )
    write_json(
        workspace
        / "storage/extractions/demo-run/llm-full-read/by-unit/paper_abstract_PAPERLLM.json",
        {
            "unit_id": "paper_abstract:PAPERLLM",
            "lane": "llm_full_read",
            "topic_ids": ["05_generative_models"],
            "confidence": 0.93,
            "metadata": {"paper_id": "PAPERLLM", "title": "Diffusion Planning"},
        },
    )

    output_dir = tmp_path / "export"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--workspace",
            str(workspace),
            "--output-dir",
            str(output_dir),
            "--paper-mode",
            "full",
        ],
        check=True,
        cwd=REPO_ROOT,
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["full_text_topic_from_llm_full_read"] == 1
    assert manifest["counts"]["full_text_topic_from_heuristic"] == 0

    nb = output_dir / "notebooklm"
    # The paper lands under the LLM topic, not the heuristic reasoning topic.
    assert list(nb.glob("papers_full_05_generative_models_part_*.md"))
    assert not list(nb.glob("papers_full_01_reasoning_post_training_part_*.md"))
    full_export = next(nb.glob("papers_full_05_generative_models_part_*.md")).read_text(
        encoding="utf-8"
    )
    assert "Primary topic: Generative Models" in full_export
    assert "Topic source: llm_full_read" in full_export


def test_export_emits_long_and_short_presentation_prompts(tmp_path: Path) -> None:
    workspace = tmp_path / "conferences/demo2026"
    write_json(
        workspace / "storage/download-corpus/openreview_accepted_papers.json",
        {"papers": []},
    )
    write_json(
        workspace / "analysis/topic_strategy.json",
        {
            "conference_label": "ICLR 2026",
            "final_topics": [
                {"topic_id": "02_agentic_systems", "title": "Agentic Systems and Autonomy"}
            ],
        },
    )
    summary = workspace / "analysis/exhaustive_summaries/02_agentic_systems__papers.md"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        "\n".join(
            [
                "# Agentic systems - papers",
                "",
                "## Key Points",
                "",
                "- Tool use is a core agentic capability.",
                "",
                "## Tensions Or Contrasts",
                "",
                "- Single-agent simplicity versus multi-agent synergy.",
                "",
                "## Open Questions",
                "",
                "- How should agent memory be governed?",
                "",
                "## Analysis Feed",
                "- ignored",
            ]
        ),
        encoding="utf-8",
    )
    (workspace / "analysis/executive_summary.md").write_text(
        "Executive synthesis\n\n1) Systems, not models, are the unit of progress.\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "export"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--workspace",
            str(workspace),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=REPO_ROOT,
    )

    prompts_dir = output_dir / "presentation_prompts"
    files = {p.name for p in prompts_dir.glob("*.txt")}
    # 12 topics x {long, short} + global {long, short} = 26.
    assert len(files) == 26
    for name in (
        "02_agentic_systems_long.txt",
        "02_agentic_systems_short.txt",
        "00_global_conference_long.txt",
        "00_global_conference_short.txt",
    ):
        assert name in files

    long_prompt = (prompts_dir / "02_agentic_systems_long.txt").read_text(encoding="utf-8")
    short_prompt = (prompts_dir / "02_agentic_systems_short.txt").read_text(encoding="utf-8")
    assert "~60 minutes" in long_prompt and "~20 minutes" in short_prompt
    assert "Agentic Systems and Autonomy" in long_prompt  # rich title from strategy
    assert "Tool use is a core agentic capability." in long_prompt  # grounded key point
    assert "slide deck" in long_prompt  # references the slide source

    global_long = (prompts_dir / "00_global_conference_long.txt").read_text(encoding="utf-8")
    assert "conference-wide overview" in global_long
    assert "Systems, not models, are the unit of progress." in global_long

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["presentation_prompts"]) == 26


def test_export_includes_topic_slide_decks(tmp_path: Path) -> None:
    workspace = tmp_path / "conferences/demo2026"
    write_json(
        workspace / "storage/download-corpus/openreview_accepted_papers.json",
        {"papers": []},
    )
    write_json(
        workspace / "analysis/topic_strategy.json",
        {
            "conference_label": "ICLR 2026",
            "final_topics": [
                {"topic_id": "02_agentic_systems", "title": "Agentic Systems and Autonomy"}
            ],
        },
    )
    slide_path = workspace / "decks/02_agentic_systems/slides.md"
    slide_path.parent.mkdir(parents=True, exist_ok=True)
    slide_path.write_text(
        "# Agentic Systems Slide Brief\n\n## Slide 1\n\n- Agents move from demos to systems.\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "export"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--workspace",
            str(workspace),
            "--output-dir",
            str(output_dir),
            "--skip-presentation-prompts",
        ],
        check=True,
        cwd=REPO_ROOT,
    )

    nb = output_dir / "notebooklm"
    slide_parts = list(nb.glob("slides_02_agentic_systems_part_*.md"))
    assert len(slide_parts) == 1
    slide_export = slide_parts[0].read_text(encoding="utf-8")
    assert "Agentic Systems Slide Brief" in slide_export
    assert "Agents move from demos to systems." in slide_export

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["topic_slide_decks"] == 1
    assert manifest["counts"]["slides_source_files"] == 1
    assert "02_agentic_systems" in manifest["parts"]["slides"]


def test_export_emits_conference_overview_source(tmp_path: Path) -> None:
    workspace = tmp_path / "conferences/demo2026"
    write_json(
        workspace / "storage/download-corpus/openreview_accepted_papers.json",
        {"papers": []},
    )
    write_json(
        workspace / "analysis/topic_strategy.json",
        {
            "conference_label": "ICLR 2026",
            "final_topics": [
                {"topic_id": "02_agentic_systems", "title": "Agentic Systems and Autonomy"}
            ],
        },
    )
    (workspace / "analysis/exhaustive_summaries").mkdir(parents=True, exist_ok=True)
    (workspace / "analysis/exhaustive_summaries/02_agentic_systems__papers.md").write_text(
        "# Agentic\n\n## Summary\n\nAgents are moving from demos to systems. More detail follows.\n",
        encoding="utf-8",
    )
    (workspace / "analysis/executive_summary.md").write_text(
        "Executive synthesis\n\nThe dominant, actionable thesis: systems are the unit of progress.\n\n"
        "1) Systems, not models, are the unit of progress.\n2) Evaluation is reliability infrastructure.\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "export"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--workspace",
            str(workspace),
            "--output-dir",
            str(output_dir),
            "--skip-podcast-prompts",
        ],
        check=True,
        cwd=REPO_ROOT,
    )

    overview_path = output_dir / "notebooklm" / "00_conference_overview.md"
    assert overview_path.exists()
    overview = overview_path.read_text(encoding="utf-8")
    assert "# ICLR 2026: Conference Overview" in overview
    assert "The dominant, actionable thesis" not in overview  # only the thesis text
    assert "Systems are the unit of progress." in overview  # extracted thesis, capitalized
    assert "Systems, not models, are the unit of progress." in overview  # cross-theme
    assert "Agentic Systems and Autonomy" in overview  # topic title from strategy
    assert "Agents are moving from demos to systems." in overview  # topic one-liner

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["conference_overview"] == "notebooklm/00_conference_overview.md"
    # Overview is the first index row.
    import csv as _csv

    with (output_dir / "index.csv").open(encoding="utf-8") as handle:
        first_row = next(_csv.DictReader(handle))
    assert first_row["kind"] == "conference_overview"
