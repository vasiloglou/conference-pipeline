from __future__ import annotations

import pytest

from conference_pipeline.extractions.classifier import check_local_llm_readiness
from conference_pipeline.provider_receipts import scan_local_llm_receipts

from tests.live_llm._helpers import assert_real_backend_kind, assert_real_endpoint_label


pytestmark = pytest.mark.live_llm


def test_readiness_returns_real_receipt(tmp_workspace, real_local_llm_backend):
    receipt = check_local_llm_readiness(
        workspace=tmp_workspace,
        run_id="live-readiness",
        stage_id="readiness",
    )

    assert_real_backend_kind(real_local_llm_backend.kind)
    assert receipt.preflight_mode is True
    assert receipt.paid_cloud_cost == 0.0
    assert receipt.model
    assert_real_endpoint_label(receipt.endpoint_label)


def test_readiness_records_real_endpoint_label_and_model(
    tmp_workspace, real_local_llm_backend
):
    receipt = check_local_llm_readiness(
        workspace=tmp_workspace,
        run_id="live-readiness-scan",
        stage_id="readiness",
    )

    scanned = scan_local_llm_receipts(tmp_workspace)
    assert any(item["endpoint_label"] == receipt.endpoint_label for item in scanned)
    assert any(item["model"] == receipt.model for item in scanned)
