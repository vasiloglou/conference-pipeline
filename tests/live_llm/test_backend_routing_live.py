from __future__ import annotations

import pytest

from tests.live_llm._helpers import assert_real_backend_kind, assert_real_endpoint_label


pytestmark = pytest.mark.live_llm


def test_selected_backend_is_real_and_ready(real_local_llm_backend):
    assert_real_backend_kind(real_local_llm_backend.kind)
    assert_real_endpoint_label(real_local_llm_backend.endpoint_label)
