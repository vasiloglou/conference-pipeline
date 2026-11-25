"""One-record, byte-exact compatibility seam for the historical r6 decision."""

from __future__ import annotations

import hashlib
import json
from types import MappingProxyType
from typing import Final


LEGACY_V1_BYTES: Final = (
    b'{\n  "audio_policy_digest": null,\n  "binding_state": "unavailable",\n'
    b'  "checkpoint_sha256": null,\n  "exit_code": 1,\n'
    b'  "historical_checkpoint_archive_sha256": null,\n'
    b'  "historical_checkpoint_lineage_sha256": null,\n'
    b'  "lineage_state": null,\n  "phase": "pre_admission",\n'
    b'  "reason_code": "admission_invalid",\n'
    b'  "receipt_type": "conference_pipeline.cvpr2026_r6_cli_decision",\n'
    b'  "record_count": null,\n  "recovery_inventory_sha256": null,\n'
    b'  "schema_version": 1,\n  "source_scope_sha256": null\n}\n'
)
LEGACY_V1_BYTE_LENGTH: Final = 483
LEGACY_V1_SHA256: Final = (
    "0e9c65de26caa096a8b68460ed43e8cd852367b6610963e6d6815a67b65945a8"
)
LEGACY_V1_DOCUMENT: Final = MappingProxyType(
    {
        "audio_policy_digest": None,
        "binding_state": "unavailable",
        "checkpoint_sha256": None,
        "exit_code": 1,
        "historical_checkpoint_archive_sha256": None,
        "historical_checkpoint_lineage_sha256": None,
        "lineage_state": None,
        "phase": "pre_admission",
        "reason_code": "admission_invalid",
        "receipt_type": "conference_pipeline.cvpr2026_r6_cli_decision",
        "record_count": None,
        "recovery_inventory_sha256": None,
        "schema_version": 1,
        "source_scope_sha256": None,
    }
)


def is_exact_legacy_v1(raw: bytes) -> bool:
    """Admit the one immutable predecessor, never a schema-v1 family."""

    if (
        type(raw) is not bytes
        or len(raw) != LEGACY_V1_BYTE_LENGTH
        or raw != LEGACY_V1_BYTES
        or hashlib.sha256(raw).hexdigest() != LEGACY_V1_SHA256
    ):
        return False
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return type(value) is dict and value == dict(LEGACY_V1_DOCUMENT) and all(
        type(value[key]) is type(expected)
        for key, expected in LEGACY_V1_DOCUMENT.items()
    )


if len(LEGACY_V1_BYTES) != LEGACY_V1_BYTE_LENGTH:
    raise RuntimeError("pinned legacy predecessor length drifted")
if hashlib.sha256(LEGACY_V1_BYTES).hexdigest() != LEGACY_V1_SHA256:
    raise RuntimeError("pinned legacy predecessor digest drifted")
