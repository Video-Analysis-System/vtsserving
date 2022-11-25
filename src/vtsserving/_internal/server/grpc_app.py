from __future__ import annotations

import typing as t
import asyncio
import logging
from typing import TYPE_CHECKING
from functools import partial

from simple_di import inject
from simple_di import Provide

from ..configuration.containers import VtsServingContainer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:

    from vtsserving.grpc.types import Interceptors

    from ..service import Service
    from .grpc.servicer import Servicer

    OnStartup = list[t.Callable[[], t.Union[None, t.Coroutine[t.Any, t.Any, None]]]]


class GRPCAppFactory:
    """
    GRPCApp creates an async gRPC API server based on APIs defined with a VtsService via VtsService#apis.
    This is a light wrapper around GRPCServer with addition to `on_startup` and `on_shutdown` hooks.

    Note that even though the code are similar with BaseAppFactory, gRPC protocol is different from ASGI.
    """

    @inject
    def __init__(
        self,
        vts_service: Service,
        *,
        enable_metrics: bool = Provide[
            VtsServingContainer.api_server_config.metrics.enabled
        ],
    ) -> None:
        self.vts_service = vts_service
        self.enable_metrics = enable_metrics

    @inject
    async def wait_for_runner_ready(
        self,
        *,
        check_interval: int = Provide[
            VtsServingContainer.api_server_config.runner_probe.period
        ],
    ):
        if VtsServingContainer.api_server_config.runner_probe.enabled.get():
            logger.info("Waiting for runners to be ready...")
            logger.debug("Current runners: %r", self.vts_service.runners)

            while True:
                try:
                    runner_statuses = (
                        runner.runner_handle_is_ready()
                        for runner in self.vts_service.runners
                    )
                    runners_ready = all(await asyncio.gather(*runner_statuses))

                    if runners_ready:
                        break
                except ConnectionError as e:
                    logger.debug("[%s] Retrying ...", e)

                await asyncio.sleep(check_interval)

            logger.info("All runners ready.")

    @property
    def on_startup(self) -> OnStartup:
        on_startup: OnStartup = [self.vts_service.on_grpc_server_startup]
        if VtsServingContainer.development_mode.get():
            for runner in self.vts_service.runners:
                on_startup.append(partial(runner.init_local, quiet=True))
        else:
            for runner in self.vts_service.runners:
                on_startup.append(runner.init_client)

        on_startup.append(self.wait_for_runner_ready)
        return on_startup

    @property
    def on_shutdown(self) -> list[t.Callable[[], None]]:
        on_shutdown = [self.vts_service.on_grpc_server_shutdown]
        for runner in self.vts_service.runners:
            on_shutdown.append(runner.destroy)

        return on_shutdown

    def __call__(self) -> Servicer:
        from .grpc import Servicer

        return Servicer(
            self.vts_service,
            on_startup=self.on_startup,
            on_shutdown=self.on_shutdown,
            mount_servicers=self.vts_service.mount_servicers,
            interceptors=self.interceptors,
        )

    @property
    def interceptors(self) -> Interceptors:
        # Note that order of interceptors is important here.

        from vtsserving.grpc.interceptors.opentelemetry import (
            AsyncOpenTelemetryServerInterceptor,
        )

        interceptors: Interceptors = [AsyncOpenTelemetryServerInterceptor]

        if self.enable_metrics:
            from vtsserving.grpc.interceptors.prometheus import PrometheusServerInterceptor

            interceptors.append(PrometheusServerInterceptor)

        if VtsServingContainer.api_server_config.logging.access.enabled.get():
            from vtsserving.grpc.interceptors.access import AccessLogServerInterceptor

            access_logger = logging.getLogger("vtsserving.access")
            if access_logger.getEffectiveLevel() <= logging.INFO:
                interceptors.append(AccessLogServerInterceptor)

        # add users-defined interceptors.
        interceptors.extend(self.vts_service.interceptors)

        return interceptors
