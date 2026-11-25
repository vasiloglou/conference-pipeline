from __future__ import annotations

import pytest

from conference_pipeline.extractions.classifier import run_quality_sample
from conference_pipeline.iclr2025_live import load_topic_families
from conference_pipeline.provider_receipts import scan_local_llm_receipts

from tests.live_llm._helpers import assert_real_endpoint_label


pytestmark = pytest.mark.live_llm


def test_quality_sample_writes_real_receipt(
    tmp_workspace,
    real_local_llm_backend,
    ambiguous_live_record,
):
    taxonomy = load_topic_families(tmp_workspace)
    result = run_quality_sample(
        [ambiguous_live_record],
        taxonomy,
        workspace=tmp_workspace,
        run_id="live-quality-sample",
        stage_id="quality-sample",
        sample_size=1,
    )

    assert result["quality_sample_mode"] is True
    assert result["sample_size"] == 1
    assert_real_endpoint_label(result["endpoint_label"])
    assert "recommendation" in result

    scanned = [
        item for item in scan_local_llm_receipts(tmp_workspace)
        if item.get("quality_sample_mode") is True
    ]
    assert scanned
    assert scanned[-1]["paid_cloud_cost"] == 0.0
    assert_real_endpoint_label(scanned[-1]["endpoint_label"])
