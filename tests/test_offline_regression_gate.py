from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RECEIPT_PATH = REPO_ROOT / "docs" / "quality" / "quality_baseline_receipt.json"

REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "receipt_type",
        "source_git_sha",
        "review_date",
        "reviewed_modules",
        "findings_summary",
        "bugs_fixed",
        "tests_added",
        "deferred_findings",
        "offline_command",
        "offline_command_result",
        "offline_command_exit_code",
        "offline_command_timestamp",
        "surfaces_covered",
    }
)
SECRET_VALUE_RE = re.compile(r"(sk-[A-Za-z0-9]{8}|AIza[0-9A-Za-z_-]{10})")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_string_values(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from iter_string_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_string_values(child)


def test_quality_baseline_receipt_exists_and_has_required_fields():
    assert RECEIPT_PATH.exists(), f"Receipt missing at {RECEIPT_PATH}"
    receipt = load_json(RECEIPT_PATH)
    missing = REQUIRED_FIELDS - set(receipt)
    assert not missing, f"Receipt missing fields: {sorted(missing)}"


def test_quality_baseline_receipt_source_git_sha_is_nonempty():
    receipt = load_json(RECEIPT_PATH)
    assert receipt["source_git_sha"]


def test_quality_baseline_receipt_contains_no_obvious_secret_values():
    receipt = load_json(RECEIPT_PATH)
    leaked = [value for value in iter_string_values(receipt) if SECRET_VALUE_RE.search(value)]
    assert leaked == []
