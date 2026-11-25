"""NeurIPS-specific analysis constants (D-17).

`NEURIPS_TOPIC_IDS` is the canonical list of topic ids for the NeurIPS 2025
workspace ONLY. It is NeurIPS-specific and must NOT be used as a
cross-conference default: reusable analysis code discovers topics from the
workspace (analysis reports on disk or `analysis/analysis_manifest.json`) and
only falls back to this tuple for the neurips2025 workspace. Checked in as a
tuple - do NOT derive from CLAUDE.md or from filesystem globbing at import
time. The drift-guard test in `tests/test_analysis_constants_drift.py`
verifies the tuple stays in sync with `conferences/neurips2025/analysis/NN_*.md`
on disk.

Source of truth for drift-guard comparison: `conferences/neurips2025/analysis/`
directory. When a topic is added/removed there, update this tuple and re-run
`pytest tests/test_analysis_constants_drift.py -q`.
"""

from __future__ import annotations

NEURIPS_TOPIC_IDS: tuple[str, ...] = (
    "01_tools_and_open_source",
    "02_genai_architectures",
    "03_data_preparation",
    "04_data_attribution",
    "05_data_markets",
    "06_visual_models",
    "07_tabular_models",
    "08_language_models",
    "09_genai_in_science",
    "10_serendipitous_findings",
    "11_ethical_problems",
    "12_commercial_applications",
    "13_industry_insights",
    "14_synthetic_data",
    "15_whats_happening_at_neurips",
    "16_neurips_awards",
    "17_llm_scaling_laws_efficiency",
    "18_agentic_ai_autonomous_systems",
    "19_notable_quotes",
    "20_small_language_models",
)
