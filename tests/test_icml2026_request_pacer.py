"""Tests for RequestPacer, the item-page probe rate limiter.

Context: a Phase 13 ICML 2026 coverage-recovery item-page sweep (180 newly
discovered tutorial/expo/invited-talk pages) hit a ~100% server-side
"log in and register to view live content" gate rate, reproducible at both
--item-page-worker-count 8 and 2, while one-off manual requests with the same
session succeeded every time. Cookie freshness and header shape were ruled
out; the signature matches burst/velocity throttling. RequestPacer bounds the
aggregate request rate across the whole worker pool.
"""

from __future__ import annotations

import threading
import time

from scripts import run_icml2026_production as runner


class TestRequestPacer:
    def test_disabled_pacer_does_not_delay(self) -> None:
        pacer = runner.RequestPacer(0.0)
        started = time.monotonic()
        for _ in range(5):
            pacer.wait()
        assert time.monotonic() - started < 0.05

    def test_enforces_minimum_interval_single_thread(self) -> None:
        pacer = runner.RequestPacer(0.05)
        started = time.monotonic()
        for _ in range(4):
            pacer.wait()
        elapsed = time.monotonic() - started
        # 4 calls -> 3 enforced gaps of >= 0.05s each.
        assert elapsed >= 0.15

    def test_bounds_aggregate_rate_across_threads(self) -> None:
        """The limit is global across the pool, not per-worker."""

        pacer = runner.RequestPacer(0.05)
        call_times: list[float] = []
        lock = threading.Lock()

        def worker() -> None:
            pacer.wait()
            with lock:
                call_times.append(time.monotonic())

        threads = [threading.Thread(target=worker) for _ in range(6)]
        started = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.monotonic() - started
        # 6 calls sharing one pacer -> at least 5 enforced gaps of 0.05s,
        # regardless of how many threads dispatched concurrently.
        assert elapsed >= 0.25
        assert len(call_times) == 6
