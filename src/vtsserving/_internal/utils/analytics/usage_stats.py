from __future__ import annotations

import os
import typing as t
import logging
import secrets
import threading
import contextlib
from typing import TYPE_CHECKING
from datetime import datetime
from datetime import timezone
from functools import wraps
from functools import lru_cache

import attr
import requests
from simple_di import inject
from simple_di import Provide

from ...utils import compose
from .schemas import EventMeta
from .schemas import ServeInitEvent
from .schemas import TrackingPayload
from .schemas import CommonProperties
from .schemas import ServeUpdateEvent
from ...configuration import get_debug_mode
from ...configuration.containers import VtsServingContainer

if TYPE_CHECKING:
    P = t.ParamSpec("P")
    T = t.TypeVar("T")
    AsyncFunc = t.Callable[P, t.Coroutine[t.Any, t.Any, t.Any]]

    from prometheus_client.samples import Sample

    from vtsserving import Service

    from ...server.metrics.prometheus import PrometheusClient

logger = logging.getLogger(__name__)

VTSSERVING_DO_NOT_TRACK = "VTSSERVING_DO_NOT_TRACK"
USAGE_TRACKING_URL = "https://t.vtsserving.com"
SERVE_USAGE_TRACKING_INTERVAL_SECONDS = int(12 * 60 * 60)  # every 12 hours
USAGE_REQUEST_TIMEOUT_SECONDS = 1


@lru_cache(maxsize=1)
def do_not_track() -> bool:  # pragma: no cover
    # Returns True if and only if the environment variable is defined and has value True.
    # The function is cached for better performance.
    return os.environ.get(VTSSERVING_DO_NOT_TRACK, str(False)).lower() == "true"


@lru_cache(maxsize=1)
def _usage_event_debugging() -> bool:
    # For VtsServing developers only - debug and print event payload if turned on
    return os.environ.get("__VTSSERVING_DEBUG_USAGE", str(False)).lower() == "true"


def silent(func: t.Callable[P, T]) -> t.Callable[P, T]:  # pragma: no cover
    # Silent errors when tracking
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> t.Any:
        try:
            return func(*args, **kwargs)
        except Exception as err:  # pylint: disable=broad-except
            if _usage_event_debugging():
                if get_debug_mode():
                    logger.error(
                        "Tracking Error: %s", err, stack_info=True, stacklevel=3
                    )
                else:
                    logger.info("Tracking Error: %s", err)
            else:
                logger.debug("Tracking Error: %s", err)

    return wrapper


@attr.define
class ServeInfo:
    serve_id: str
    serve_started_timestamp: datetime


def get_serve_info() -> ServeInfo:  # pragma: no cover
    # Returns a safe token for serve as well as timestamp of creating this token
    return ServeInfo(
        serve_id=secrets.token_urlsafe(32),
        serve_started_timestamp=datetime.now(timezone.utc),
    )


@inject
def get_payload(
    event_properties: EventMeta,
    session_id: str = Provide[VtsServingContainer.session_id],
) -> t.Dict[str, t.Any]:
    return TrackingPayload(
        session_id=session_id,
        common_properties=CommonProperties(),
        event_properties=event_properties,
        event_type=event_properties.event_name,
    ).to_dict()


@silent
def track(event_properties: EventMeta):
    if do_not_track():
        return
    payload = get_payload(event_properties=event_properties)

    if _usage_event_debugging():
        # For internal debugging purpose
        logger.info("Tracking Payload: %s", payload)
        return

    requests.post(
        USAGE_TRACKING_URL, json=payload, timeout=USAGE_REQUEST_TIMEOUT_SECONDS
    )


@inject
def _track_serve_init(
    svc: Service,
    production: bool,
    serve_kind: str,
    serve_info: ServeInfo = Provide[VtsServingContainer.serve_info],
):
    if svc.vts is not None:
        vts = svc.vts
        event_properties = ServeInitEvent(
            serve_id=serve_info.serve_id,
            serve_from_vts=True,
            production=production,
            serve_kind=serve_kind,
            vts_creation_timestamp=vts.info.creation_time,
            num_of_models=len(vts.info.models),
            num_of_runners=len(svc.runners),
            num_of_apis=len(vts.info.apis),
            model_types=[m.module for m in vts.info.models],
            runnable_types=[r.runnable_type for r in vts.info.runners],
            api_input_types=[api.input_type for api in vts.info.apis],
            api_output_types=[api.output_type for api in vts.info.apis],
        )
    else:
        event_properties = ServeInitEvent(
            serve_id=serve_info.serve_id,
            serve_from_vts=False,
            production=production,
            serve_kind=serve_kind,
            vts_creation_timestamp=None,
            num_of_models=len(
                set(
                    svc.models
                    + [model for runner in svc.runners for model in runner.models]
                )
            ),
            num_of_runners=len(svc.runners),
            num_of_apis=len(svc.apis.keys()),
            runnable_types=[r.runnable_class.__name__ for r in svc.runners],
            api_input_types=[api.input.__class__.__name__ for api in svc.apis.values()],
            api_output_types=[
                api.output.__class__.__name__ for api in svc.apis.values()
            ],
        )

    track(event_properties)


EXCLUDE_PATHS = {"/docs.json", "/livez", "/healthz", "/readyz"}


def filter_metrics(
    samples: list[Sample], *filters: t.Callable[[list[Sample]], list[Sample]]
):
    return [
        {**sample.labels, "value": sample.value}
        for sample in compose(*filters)(samples)
    ]


def get_metrics_report(
    metrics_client: PrometheusClient,
    serve_kind: str,
) -> list[dict[str, str | float]]:
    """
    Get Prometheus metrics reports from the metrics client. This will be used to determine tracking events.
    If the return metrics are legacy metrics, the metrics will have prefix VTSSERVING_, otherwise they will have prefix vtsserving_

    Args:
        metrics_client: Instance of vtsserving._internal.server.metrics.prometheus.PrometheusClient
        grpc: Whether the metrics are for gRPC server.

    Returns:
        A tuple of a list of metrics and an optional boolean to determine whether the return metrics are legacy metrics.
    """
    for metric in metrics_client.text_string_to_metric_families():
        metric_type = t.cast("str", metric.type)  # type: ignore (we need to cast due to no prometheus types)
        metric_name = t.cast("str", metric.name)  # type: ignore (we need to cast due to no prometheus types)
        metric_samples = t.cast("list[Sample]", metric.samples)  # type: ignore (we need to cast due to no prometheus types)
        if metric_type != "counter":
            continue
        # We only care about the counter metrics.
        assert metric_type == "counter"
        if serve_kind == "grpc":
            _filters: list[t.Callable[[list[Sample]], list[Sample]]] = [
                lambda samples: [s for s in samples if "api_name" in s.labels]
            ]
        elif serve_kind == "http":
            _filters = [
                lambda samples: [
                    s
                    for s in samples
                    if not s.labels["endpoint"].startswith("/static_content/")
                ],
                lambda samples: [
                    s for s in samples if s.labels["endpoint"] not in EXCLUDE_PATHS
                ],
                lambda samples: [s for s in samples if "endpoint" in s.labels],
            ]
        else:
            raise NotImplementedError("Unknown serve kind %s" % serve_kind)
        # If metrics prefix is VTSSERVING_, this is legacy metrics
        if metric_name.endswith("_request") and (
            metric_name.startswith("vtsserving_") or metric_name.startswith("VTSSERVING_")
        ):
            return filter_metrics(metric_samples, *_filters)

    return []


@inject
@contextlib.contextmanager
def track_serve(
    svc: Service,
    *,
    production: bool = False,
    serve_kind: str = "http",
    component: str = "standalone",
    metrics_client: PrometheusClient = Provide[VtsServingContainer.metrics_client],
    serve_info: ServeInfo = Provide[VtsServingContainer.serve_info],
) -> t.Generator[None, None, None]:
    if do_not_track():
        yield
        return

    _track_serve_init(svc=svc, production=production, serve_kind=serve_kind)

    if _usage_event_debugging():
        tracking_interval = 5
    else:
        tracking_interval = SERVE_USAGE_TRACKING_INTERVAL_SECONDS  # pragma: no cover

    stop_event = threading.Event()

    @silent
    def loop() -> t.NoReturn:  # type: ignore
        last_tracked_timestamp: datetime = serve_info.serve_started_timestamp
        while not stop_event.wait(tracking_interval):  # pragma: no cover
            now = datetime.now(timezone.utc)
            event_properties = ServeUpdateEvent(
                serve_id=serve_info.serve_id,
                production=production,
                # Note that we are currently only have two tracking jobs: http and grpc
                serve_kind=serve_kind,
                # Current accept components are "standalone", "api_server" and "runner"
                component=component,
                triggered_at=now,
                duration_in_seconds=int((now - last_tracked_timestamp).total_seconds()),
                metrics=get_metrics_report(metrics_client, serve_kind=serve_kind),
            )
            last_tracked_timestamp = now
            track(event_properties)

    tracking_thread = threading.Thread(target=loop, daemon=True)
    try:
        tracking_thread.start()
        yield
    finally:
        stop_event.set()
        tracking_thread.join()