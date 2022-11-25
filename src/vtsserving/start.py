from __future__ import annotations

import os
import sys
import json
import typing as t
import logging
import contextlib

from simple_di import inject
from simple_di import Provide

from ._internal.configuration.containers import VtsServingContainer

logger = logging.getLogger(__name__)

SCRIPT_RUNNER = "vtsserving_cli.worker.runner"
SCRIPT_API_SERVER = "vtsserving_cli.worker.http_api_server"
SCRIPT_GRPC_API_SERVER = "vtsserving_cli.worker.grpc_api_server"
SCRIPT_GRPC_PROMETHEUS_SERVER = "vtsserving_cli.worker.grpc_prometheus_server"

API_SERVER = "api_server"
RUNNER = "runner"


@inject
def start_runner_server(
    vts_identifier: str,
    working_dir: str,
    runner_name: str,
    port: int | None = None,
    host: str | None = None,
    backlog: int = Provide[VtsServingContainer.api_server_config.backlog],
) -> None:
    """
    Experimental API for serving a VtsServing runner.
    """
    from .serve import ensure_prometheus_dir

    prometheus_dir = ensure_prometheus_dir()

    from vtsserving import load

    from ._internal.utils import reserve_free_port
    from ._internal.utils.circus import create_standalone_arbiter
    from ._internal.utils.analytics import track_serve

    working_dir = os.path.realpath(os.path.expanduser(working_dir))
    svc = load(vts_identifier, working_dir=working_dir, standalone_load=True)

    from circus.sockets import CircusSocket  # type: ignore
    from circus.watcher import Watcher  # type: ignore

    watchers: t.List[Watcher] = []
    circus_socket_map: t.Dict[str, CircusSocket] = {}

    with contextlib.ExitStack() as port_stack:
        for runner in svc.runners:
            if runner.name == runner_name:
                if port is None:
                    port = port_stack.enter_context(reserve_free_port())
                if host is None:
                    host = "127.0.0.1"
                circus_socket_map[runner.name] = CircusSocket(
                    name=runner.name,
                    host=host,
                    port=port,
                    backlog=backlog,
                )

                watchers.append(
                    Watcher(
                        name=f"{RUNNER}_{runner.name}",
                        cmd=sys.executable,
                        args=[
                            "-m",
                            SCRIPT_RUNNER,
                            vts_identifier,
                            "--runner-name",
                            runner.name,
                            "--fd",
                            f"$(circus.sockets.{runner.name})",
                            "--working-dir",
                            working_dir,
                            "--no-access-log",
                            "--worker-id",
                            "$(circus.wid)",
                            "--prometheus-dir",
                            prometheus_dir,
                        ],
                        copy_env=True,
                        stop_children=True,
                        use_sockets=True,
                        working_dir=working_dir,
                        numprocesses=runner.scheduled_worker_count,
                    )
                )
                break
        else:
            raise ValueError(
                f"Runner {runner_name} not found in the service: `{vts_identifier}`, "
                f"available runners: {[r.name for r in svc.runners]}"
            )
    arbiter = create_standalone_arbiter(
        watchers=watchers,
        sockets=list(circus_socket_map.values()),
    )
    with track_serve(svc, production=True, component=RUNNER):
        arbiter.start(
            cb=lambda _: logger.info(  # type: ignore
                'Starting RunnerServer from "%s" running on http://%s:%s (Press CTRL+C to quit)',
                vts_identifier,
                host,
                port,
            ),
        )


@inject
def start_http_server(
    vts_identifier: str,
    runner_map: t.Dict[str, str],
    working_dir: str,
    port: int = Provide[VtsServingContainer.api_server_config.port],
    host: str = Provide[VtsServingContainer.api_server_config.host],
    backlog: int = Provide[VtsServingContainer.api_server_config.backlog],
    api_workers: int = Provide[VtsServingContainer.api_server_workers],
    ssl_certfile: str | None = Provide[VtsServingContainer.ssl.certfile],
    ssl_keyfile: str | None = Provide[VtsServingContainer.ssl.keyfile],
    ssl_keyfile_password: str | None = Provide[VtsServingContainer.ssl.keyfile_password],
    ssl_version: int | None = Provide[VtsServingContainer.ssl.version],
    ssl_cert_reqs: int | None = Provide[VtsServingContainer.ssl.cert_reqs],
    ssl_ca_certs: str | None = Provide[VtsServingContainer.ssl.ca_certs],
    ssl_ciphers: str | None = Provide[VtsServingContainer.ssl.ciphers],
) -> None:
    from .serve import ensure_prometheus_dir

    prometheus_dir = ensure_prometheus_dir()

    from circus.sockets import CircusSocket
    from circus.watcher import Watcher

    from vtsserving import load

    from .serve import create_watcher
    from .serve import API_SERVER_NAME
    from .serve import construct_ssl_args
    from .serve import PROMETHEUS_MESSAGE
    from ._internal.utils.circus import create_standalone_arbiter
    from ._internal.utils.analytics import track_serve

    working_dir = os.path.realpath(os.path.expanduser(working_dir))
    svc = load(vts_identifier, working_dir=working_dir, standalone_load=True)
    runner_requirements = {runner.name for runner in svc.runners}
    if not runner_requirements.issubset(set(runner_map)):
        raise ValueError(
            f"{vts_identifier} requires runners {runner_requirements}, but only {set(runner_map)} are provided."
        )
    watchers: t.List[Watcher] = []
    circus_socket_map: t.Dict[str, CircusSocket] = {}
    logger.debug("Runner map: %s", runner_map)
    circus_socket_map[API_SERVER_NAME] = CircusSocket(
        name=API_SERVER_NAME,
        host=host,
        port=port,
        backlog=backlog,
    )
    ssl_args = construct_ssl_args(
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
        ssl_keyfile_password=ssl_keyfile_password,
        ssl_version=ssl_version,
        ssl_cert_reqs=ssl_cert_reqs,
        ssl_ca_certs=ssl_ca_certs,
        ssl_ciphers=ssl_ciphers,
    )
    scheme = "https" if len(ssl_args) > 0 else "http"
    watchers.append(
        create_watcher(
            name="api_server",
            args=[
                "-m",
                SCRIPT_API_SERVER,
                vts_identifier,
                "--fd",
                f"$(circus.sockets.{API_SERVER_NAME})",
                "--runner-map",
                json.dumps(runner_map),
                "--working-dir",
                working_dir,
                "--backlog",
                f"{backlog}",
                "--worker-id",
                "$(CIRCUS.WID)",
                "--prometheus-dir",
                prometheus_dir,
                *ssl_args,
            ],
            working_dir=working_dir,
            numprocesses=api_workers,
        )
    )
    if VtsServingContainer.api_server_config.metrics.enabled.get():
        logger.info(
            PROMETHEUS_MESSAGE,
            scheme.upper(),
            vts_identifier,
            f"{scheme}://{host}:{port}/metrics",
        )

    arbiter = create_standalone_arbiter(
        watchers=watchers,
        sockets=list(circus_socket_map.values()),
    )
    with track_serve(svc, production=True, component=API_SERVER):
        arbiter.start(
            cb=lambda _: logger.info(  # type: ignore
                'Starting bare %s VtsServer from "%s" running on %s://%s:%d (Press CTRL+C to quit)',
                scheme.upper(),
                vts_identifier,
                scheme,
                host,
                port,
            ),
        )


@inject
def start_grpc_server(
    vts_identifier: str,
    runner_map: dict[str, str],
    working_dir: str,
    port: int = Provide[VtsServingContainer.grpc.port],
    host: str = Provide[VtsServingContainer.grpc.host],
    backlog: int = Provide[VtsServingContainer.api_server_config.backlog],
    api_workers: int = Provide[VtsServingContainer.api_server_workers],
    reflection: bool = Provide[VtsServingContainer.grpc.reflection.enabled],
    channelz: bool = Provide[VtsServingContainer.grpc.channelz.enabled],
    max_concurrent_streams: int
    | None = Provide[VtsServingContainer.grpc.max_concurrent_streams],
    ssl_certfile: str | None = Provide[VtsServingContainer.ssl.certfile],
    ssl_keyfile: str | None = Provide[VtsServingContainer.ssl.keyfile],
    ssl_ca_certs: str | None = Provide[VtsServingContainer.ssl.ca_certs],
) -> None:
    from .serve import ensure_prometheus_dir

    prometheus_dir = ensure_prometheus_dir()

    from circus.sockets import CircusSocket
    from circus.watcher import Watcher

    from vtsserving import load

    from .serve import create_watcher
    from .serve import construct_ssl_args
    from .serve import PROMETHEUS_MESSAGE
    from .serve import PROMETHEUS_SERVER_NAME
    from ._internal.utils import reserve_free_port
    from ._internal.utils.circus import create_standalone_arbiter
    from ._internal.utils.analytics import track_serve

    working_dir = os.path.realpath(os.path.expanduser(working_dir))
    svc = load(vts_identifier, working_dir=working_dir, standalone_load=True)
    runner_requirements = {runner.name for runner in svc.runners}
    if not runner_requirements.issubset(set(runner_map)):
        raise ValueError(
            f"{vts_identifier} requires runners {runner_requirements}, but only {set(runner_map)} are provided."
        )
    watchers: list[Watcher] = []
    circus_socket_map: dict[str, CircusSocket] = {}
    logger.debug("Runner map: %s", runner_map)
    ssl_args = construct_ssl_args(
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
        ssl_ca_certs=ssl_ca_certs,
    )
    scheme = "https" if len(ssl_args) > 0 else "http"
    with contextlib.ExitStack() as port_stack:
        api_port = port_stack.enter_context(
            reserve_free_port(host=host, port=port, enable_so_reuseport=True)
        )

        args = [
            "-m",
            SCRIPT_GRPC_API_SERVER,
            vts_identifier,
            "--host",
            host,
            "--port",
            str(api_port),
            "--runner-map",
            json.dumps(runner_map),
            "--working-dir",
            working_dir,
            "--prometheus-dir",
            prometheus_dir,
            "--worker-id",
            "$(CIRCUS.WID)",
            *ssl_args,
        ]
        if reflection:
            args.append("--enable-reflection")
        if channelz:
            args.append("--enable-channelz")
        if max_concurrent_streams:
            args.extend(
                [
                    "--max-concurrent-streams",
                    str(max_concurrent_streams),
                ]
            )

        watchers.append(
            create_watcher(
                name="grpc_api_server",
                args=args,
                use_sockets=False,
                working_dir=working_dir,
                numprocesses=api_workers,
            )
        )

    if VtsServingContainer.api_server_config.metrics.enabled.get():
        metrics_host = VtsServingContainer.grpc.metrics.host.get()
        metrics_port = VtsServingContainer.grpc.metrics.port.get()

        circus_socket_map[PROMETHEUS_SERVER_NAME] = CircusSocket(
            name=PROMETHEUS_SERVER_NAME,
            host=metrics_host,
            port=metrics_port,
            backlog=backlog,
        )

        watchers.append(
            create_watcher(
                name="prom_server",
                args=[
                    "-m",
                    SCRIPT_GRPC_PROMETHEUS_SERVER,
                    "--fd",
                    f"$(circus.sockets.{PROMETHEUS_SERVER_NAME})",
                    "--prometheus-dir",
                    prometheus_dir,
                    "--backlog",
                    f"{backlog}",
                ],
                working_dir=working_dir,
                numprocesses=1,
                singleton=True,
            )
        )

        logger.info(
            PROMETHEUS_MESSAGE,
            "gRPC",
            vts_identifier,
            f"http://{metrics_host}:{metrics_port}",
        )
    arbiter = create_standalone_arbiter(
        watchers=watchers, sockets=list(circus_socket_map.values())
    )
    with track_serve(svc, production=True, component=API_SERVER, serve_kind="grpc"):
        arbiter.start(
            cb=lambda _: logger.info(  # type: ignore
                'Starting bare %s VtsServer from "%s" running on %s://%s:%d (Press CTRL+C to quit)',
                "gRPC",
                vts_identifier,
                scheme,
                host,
                port,
            ),
        )
