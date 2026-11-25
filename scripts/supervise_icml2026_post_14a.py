#!/usr/bin/env python3
"""RETIRED: the ICML 2026 post-14A supervisor.

This script used to wait for the Stage 14A full-read extraction and then chain
the analysis sub-steps and presentation builds unattended. It is retired
because it:

* ran 07D-07H topic reconciliation *before* per-topic paper matching (the wrong
  order: reconciliation reads the source packs that matching feeds);
* forced ``--max-input-tokens 400000`` on the 07A summaries, overriding the
  tier-aware chunk default and billing most calls at the high price tier;
* hardcoded model literals instead of reading the model-governance registry;
* continued into presentation builds without an operator gate.

It now refuses to run and prints the correct post-full-read order. Run each
step by hand as documented in docs/workflows/pipeline-workflow.md
("Post-Full-Read Analysis Chain").
"""

from __future__ import annotations

import sys

DOCS_REFERENCE = "docs/workflows/pipeline-workflow.md (section: Post-Full-Read Analysis Chain)"

POST_FULL_READ_ORDER = (
    "07A exhaustive topic summaries (scripts/summarize_llm_full_read_topics.py; "
    "STOP POINT: model from workflow.exhaustive_summary_model in config.yaml or "
    "--provider, else it refuses and asks; tier-aware chunk default -- do not pass "
    "--max-input-tokens)",
    "07B deep-dive question selection (scripts/select_deep_dive_questions.py)",
    "07C dual deep-dive research lanes (scripts/run_dual_deep_dive_research.py; "
    "engine from workflow.research_engine in config.yaml)",
    "per-topic paper matching (scripts/match_papers_from_artifact_keypoints.py --topic-id <topic>)",
    "Stage 7 analysis pass 1 WITHOUT --provider-cost-approved ($0; rebuilds source packs; "
    "expected to stop on the lineage check)",
    "07D-07H reconciliation / reference workflow (scripts/run_topic_reconciliation_reference_workflow.py)",
    "Stage 7 analysis pass 2 WITH --provider-cost-approved",
    "Stage 8 QA and gap-fill synthesis",
)


def refusal_message() -> str:
    steps = "\n".join(f"  {index}. {step}" for index, step in enumerate(POST_FULL_READ_ORDER, start=1))
    return (
        "scripts/supervise_icml2026_post_14a.py is retired and runs nothing.\n"
        "It used the wrong step order (reconciliation before paper matching), forced\n"
        "--max-input-tokens 400000, hardcoded model literals, and continued into presentation.\n"
        "Run the post-full-read chain step by hand, with operator cost approval before each\n"
        "paid step and `conference-pipeline workflow stage-status` between steps:\n"
        f"{steps}\n"
        f"See {DOCS_REFERENCE}."
    )


def main(argv: list[str] | None = None) -> int:
    del argv  # every argument is ignored: the supervisor never runs anything
    print(refusal_message(), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
