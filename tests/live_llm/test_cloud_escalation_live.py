from __future__ import annotations

import os

import pytest

from conference_pipeline.extractions.classifier import run_quality_sample
from conference_pipeline.iclr2025_live import load_topic_families
from conference_pipeline.provider_receipts import scan_escalation_receipts


pytestmark = pytest.mark.live_llm


def _require_live_cloud_escalation() -> None:
    if os.environ.get("RUN_LIVE_CLOUD_ESCALATION") == "1":
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.fail(
                "RUN_LIVE_CLOUD_ESCALATION=1 requires OPENAI_API_KEY for the "
                "paid-cloud escalation proof lane"
            )
        return
    pytest.skip(
        "paid-cloud escalation proof is opt-in; set "
        "RUN_LIVE_CLOUD_ESCALATION=1 with provider credentials"
    )


@pytest.fixture
def paid_cloud_enabled() -> None:
    _require_live_cloud_escalation()


@pytest.fixture
def paid_cloud_local_backend(paid_cloud_enabled, real_local_llm_backend):
    del paid_cloud_enabled
    return real_local_llm_backend


def test_quality_sample_can_write_real_cloud_escalation_receipt(
    tmp_workspace,
    paid_cloud_local_backend,
    ambiguous_live_record,
):
    assert paid_cloud_local_backend is not None
    taxonomy = load_topic_families(tmp_workspace)

    result = run_quality_sample(
        [ambiguous_live_record],
        taxonomy,
        workspace=tmp_workspace,
        run_id="live-cloud-escalation",
        stage_id="quality-sample",
        sample_size=1,
        threshold_overrides={
            "cloud_escalation_trigger": 0.95,
            "min_mean_confidence": 0.0,
            "min_record_coverage": 0.0,
        },
    )

    assert result["recommendation"]["should_escalate"] is True
    assert result["recommendation"]["cloud_escalation_status"] == "succeeded"
    assert result["cloud_escalation"]["provider"] == "openai"
    assert result["cloud_escalation"]["escalation_receipt_path"]

    receipts = scan_escalation_receipts(tmp_workspace)
    assert receipts
    receipt = receipts[-1]
    assert receipt["wrapped_receipt"]["provider"] == "openai"
    assert receipt["wrapped_receipt"]["status"] == "succeeded"
    assert receipt["fail_closed_state"] is False
    assert receipt["affected_record_ids"] == ["live:ambiguous-1"]
