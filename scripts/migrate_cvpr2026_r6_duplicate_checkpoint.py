#!/usr/bin/env python3
"""Run the exact offline CVPR r6 duplicate-checkpoint migration once."""

from __future__ import annotations

import json
import sys

from conference_pipeline.acquisition.cvpr_r6_artifact_contract import R6ArtifactContract
from conference_pipeline.acquisition.cvpr_r6_checkpoint_migration import (
    R6CheckpointMigrationError,
    migrate_r6_duplicate_checkpoint,
)


def main(argv: list[str] | tuple[str, ...] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments:
        print(
            json.dumps(
                {"status": "blocked", "reason_code": "arguments_not_permitted"},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    try:
        result = migrate_r6_duplicate_checkpoint(contract=R6ArtifactContract.production())
    except (R6CheckpointMigrationError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "reason_code": "fixed_incident_validation_failed",
                    "detail": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "status": result.status,
                "changed": result.changed,
                "duplicate_count": result.duplicate_count,
                "unique_count": result.unique_count,
                "remaining_count": result.remaining_count,
                "canonical_checkpoint_sha256": result.canonical_checkpoint_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
