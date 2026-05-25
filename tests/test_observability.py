import json
import logging

from rag_engine.observability.counters import Counters
from rag_engine.observability.logging import JsonFormatter, configure_logging, log_event


def test_json_formatter_emits_required_fields() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="rag_engine.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.trace_id = "abc"
    record.event = "test.event"
    record.duration_ms = 12.3

    line = formatter.format(record)
    payload = json.loads(line)
    assert payload["level"] == "INFO"
    assert payload["logger"] == "rag_engine.test"
    assert payload["trace_id"] == "abc"
    assert payload["event"] == "test.event"
    assert payload["duration_ms"] == 12.3


def test_log_event_threads_extras_through(caplog) -> None:
    configure_logging(log_format="json")
    logger = logging.getLogger("rag_engine.test")
    logger.propagate = True
    with caplog.at_level(logging.INFO, logger="rag_engine.test"):
        log_event(logger, "graph.retrieve", trace_id="t1", n_candidates=20, n_reranked=6)
    record = next(r for r in caplog.records if getattr(r, "event", None) == "graph.retrieve")
    assert record.trace_id == "t1"
    assert record.n_candidates == 20
    assert record.n_reranked == 6


def test_counters_increment_and_snapshot() -> None:
    counters = Counters()
    counters.increment("queries.total")
    counters.increment("queries.total", amount=4)
    counters.increment("queries.errors")
    snap = counters.snapshot()
    assert snap["totals"] == {"queries.total": 5, "queries.errors": 1}
    assert snap["samples"] == {}


def test_counters_observe_produces_percentile_summary() -> None:
    counters = Counters()
    for value in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        counters.observe("latency_ms", value)
    sample = counters.snapshot()["samples"]["latency_ms"]
    assert sample["count"] == 10.0
    assert sample["mean"] == 55.0
    assert sample["max"] == 100.0
    # snapshot uses statistics.median (midpoint of the two middle values).
    assert sample["p50"] == 55.0
    assert sample["p95"] in (90.0, 100.0)


def test_counters_reset_clears_state() -> None:
    counters = Counters()
    counters.increment("a")
    counters.observe("b", 1.0)
    counters.reset()
    assert counters.snapshot() == {"totals": {}, "samples": {}}
