#!/usr/bin/env python3
"""Acquire CVPR r6 declared direct-public materials from fixed receipt paths.

This owner intentionally exposes no URL, header, cookie, session, host, or
credential option.  It must be invoked only after the separate live policy and
discovery gates have written the fixed r6 discovery receipt.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping

from conference_pipeline.acquisition.cvpr_r6_artifact_contract import R6ArtifactContract
from conference_pipeline.acquisition.cvpr_r6_media_recovery import (
    derive_r6_source_route_scope,
    validate_r6_source_route_scope,
)
from conference_pipeline.acquisition.public_material_acquisition import (
    PublicMaterialResponse,
    run_fixed_r6_material_acquisition,
)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


class UrllibDirectPublicTransport:
    """Minimal unauthenticated transport; redirect control stays in the module."""

    def open(self, url: str) -> PublicMaterialResponse:
        opener = urllib.request.build_opener(_NoRedirect())
        request = urllib.request.Request(url, headers={})
        try:
            response = opener.open(request)
        except urllib.error.HTTPError as error:
            return PublicMaterialResponse(
                status_code=error.code,
                headers={str(key): str(value) for key, value in error.headers.items()},
                body=(),
            )
        headers: Mapping[str, str] = {
            str(key): str(value) for key, value in response.headers.items()
        }

        def chunks() -> Iterable[bytes]:
            with response:
                while chunk := response.read(1024 * 1024):
                    yield chunk

        return PublicMaterialResponse(status_code=response.status, headers=headers, body=chunks())


def main() -> int:
    contract = R6ArtifactContract.production()
    recovery_path = contract.path("recovery_inventory")
    try:
        recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit("fixed r6 recovery inventory is required before material acquisition") from exc
    if not isinstance(recovery, dict):
        raise SystemExit("fixed r6 recovery inventory must be an object")
    # Recompute before transport construction, so a stale or reduced official
    # page scope cannot turn into a material-acquisition run.
    derive_r6_source_route_scope(recovery)
    validate_r6_source_route_scope(recovery, recovery)
    run_fixed_r6_material_acquisition(
        workspace=contract.workspace,
        transport=UrllibDirectPublicTransport(),
        artifact_contract=contract,
        recovery_inventory=recovery,
        require_source_scope=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
