from __future__ import annotations

import vtsserving


def test_metrics_initialization():
    o = vtsserving.metrics.Gauge(name="test_metrics", documentation="test")
    assert isinstance(o, vtsserving.metrics._LazyMetric)
    assert o._proxy is None
    o = vtsserving.metrics.Histogram(name="test_metrics", documentation="test")
    assert isinstance(o, vtsserving.metrics._LazyMetric)
    assert o._proxy is None
    o = vtsserving.metrics.Counter(name="test_metrics", documentation="test")
    assert isinstance(o, vtsserving.metrics._LazyMetric)
    assert o._proxy is None
    o = vtsserving.metrics.Summary(name="test_metrics", documentation="test")
    assert isinstance(o, vtsserving.metrics._LazyMetric)
    assert o._proxy is None
