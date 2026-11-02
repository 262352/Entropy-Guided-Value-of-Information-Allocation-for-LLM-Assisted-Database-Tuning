from __future__ import annotations

import json

from cognitune.evaluator.postgres_benchbase import _extract_metric


def test_goodput_is_the_primary_metric(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({
        "Throughput (requests/second)": 1166.9,
        "Goodput (requests/second)": 211.6,
        "Measured Requests": 141195,
    }))
    metric, details = _extract_metric(summary, 1166.9)
    assert metric == 211.6
    assert details["metric"] == "goodput"
    assert details["throughput_requests_per_second"] == 1166.9
    assert details["measured_requests"] == 141195


def test_throughput_fallback_when_goodput_missing(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"Throughput (requests/second)": 356.7}))
    metric, details = _extract_metric(summary, 356.7)
    assert metric == 356.7
    assert details["metric"] == "throughput"
