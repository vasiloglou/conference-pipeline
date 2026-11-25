"""
Load per-conference topic guardrails for prompts and QA.

Guardrails live next to the conference analysis plan:
  conferences/<conf>/analysis/topic_guardrails.json

Schema (per topic_id):
{
  "01": {
    "required_cues": ["benchmark", "framework", "infrastructure"],
    "min_quote_len": 12,
    "min_summary_len": 8,
    "allow_generic": false
  },
  ...
}
"""

import json
from pathlib import Path
from typing import Dict, Any


def load_topic_guardrails(conf_base_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load topic guardrails from analysis/topic_guardrails.json.

    Args:
        conf_base_dir: Conference base directory

    Returns:
        Dict mapping topic_id -> guardrail dict
    """
    guardrail_path = Path(conf_base_dir) / "analysis" / "topic_guardrails.json"
    if not guardrail_path.exists():
        return {}

    try:
        with open(guardrail_path) as f:
            data = json.load(f)
        # Normalize keys to strings (topic ids)
        return {str(k).zfill(2): v for k, v in data.items()}
    except Exception:
        # On malformed file, return empty to avoid breaking pipeline
        return {}
