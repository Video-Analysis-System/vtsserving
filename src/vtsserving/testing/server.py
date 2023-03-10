# pylint: disable=redefined-outer-name,not-context-manager
from __future__ import annotations

import os
import sys
import time
import socket
import typing as t
import urllib
import asyncio
import itertools
import contextlib
import subprocess
import urllib.error
import urllib.request
import multiprocessing
from typing import TYPE_CHECKING
from contextlib import contextmanager

import psutil

from vtsserving.grpc.utils import import_grpc
from vtsserving._internal.tag import Tag
from vtsserving._internal.utils import LazyLoader
from vtsserving._internal.utils import reserve_free_port
from vtsserving._internal.utils import cached_contextmanager

if TYPE_CHECKING:
    from grpc import aio
    from grpc_health.v1 import health_pb2 as pb_health
    from starlette.datastructures import Headers
    from starlette.datastructures import FormData

    from vtsserving._internal.vts.vts import Vts

else:
    pb_health = LazyLoader("pb_health", globals(), "grpc_health.v1.health_pb2")
    _, aio = import_grpc()


async def parse_multipart_form(headers: Headers, body: bytes) -> FormData:
    """
    parse starlette forms from headers and body
    """

    from starlette.formparsers import MultiPartParser

    async def async_bytesio(bytes_: bytes) -> t.AsyncGenerator[bytes, None]:
        yield bytes_
        yield b""
        return

    parser = MultiPartParser(headers=headers, stream=async_bytesio(body))
    return await parser.parse()


def kill_subprocess_tree(p: subprocess.Popen[t.Any]) -> None:
    """
    Tell the process to terminate and kill all of its children. Availabe both on Windows and Linux.
    Note: It will return immediately rather than wait for the process to terminate.

    Args:
        p: subprocess.Popen object
    """
    if psutil.WINDOWS:
        subprocess.call(["taskkill", "/F", "/T", "/PID", str(p.pid)])
    else:
        p.terminate()


async def server_warmup(
    host_url: str,
    timeout: float,
    grpc: bool = False,
    check_interval: float = 1,
    popen: subprocess.Popen[t.Any] | None = None,
    service_name: str | None = None,
    _internal_stubs_version: str = "v1",
) -> bool:
    start_time = time.time()
    proxy_handler = urllib.request.ProxyHandler({})
    opener = urllib.request.build_opener(proxy_handler)
    print("Waiting for host %s to be ready.." % host_url)
    while time.time() - start_time < timeout:
        if grpc:
            from vtsserving.testing.grpc import create_channel

            try:
                if service_name is None:
                    service_name = (
                        f"vtsserving.grpc.{_internal_stubs_version}.VtsService"
                    )
                async with create_channel(host_url) as channel:
                    Check = channel.unary_unary(
                        "/grpc.health.v1.Health/Check",
                        request_serializer=pb_health.HealthCheckRequest.SerializeToString,
                        response_deserializer=pb_health.HealthCheckResponse.FromString,
                    )
                    resp = await t.cast(
                        t.Awaitable[pb_health.HealthCheckResponse],
                        Check(
                            pb_health.HealthCheckRequest(service=service_name),
                            timeout=timeout,
                        ),
                    )
                    if resp.status == pb_health.HealthCheckResponse.SERVING:
                        return True
                    else:
                        await asyncio.sleep(check_interval)
            except aio.AioRpcError as e:
                print(f"[{e}] Retrying to connect to the host {host_url}...")
                await asyncio.sleep(check_interval)
        else:
            try:
                if popen and popen.poll() is not None:
                    return False
                elif opener.open(f"http://{host_url}/readyz", timeout=1).status == 200:
                    return True
                else:
                    await asyncio.sleep(check_interval)
            except (
                ConnectionError,
                urllib.error.URLError,
                socket.timeout,
            ) as e:
                print(f"[{e}] Retrying to connect to the host {host_url}...")
                await asyncio.sleep(check_interval)
    print(f"Timed out waiting {timeout} seconds for Server {host_url} to be ready.")
    return False


@cached_contextmanager("{project_path}, {cleanup}")
def build(project_path: str, cleanup: bool = True) -> t.Generator[Vts, None, None]:
    """
    Build a VtsServing project.
    """
    from vtsserving import vtss

    print(f"Building vts: {project_path}")
    vts = vtss.build_vtsfile(build_ctx=project_path)
    yield vts
    if cleanup:
        print(f"Deleting vts: {str(vts.tag)}")
        vtss.delete(vts.tag)


@cached_contextmanager("{vts_tag}, {image_tag}, {cleanup}, {use_grpc}")
def containerize(
    vts_tag: str | Tag,
    image_tag: str | None = None,
    cleanup: bool = True,
    use_grpc: bool = False,
    backend: str = "docker",
) -> t.Generator[str, None, None]:
    """
    Build the docker image from a saved vts, yield the docker image tag
    """
    from vtsserving import vtss

    vts_tag = Tag.from_taglike(vts_tag)
    if image_tag is None:
        image_tag = vts_tag.name
    try:
        print(f"Building vts server container: {vts_tag}")
        vtss.containerize(
            str(vts_tag),
            docker_image_tag=(image_tag,),
            progress="plain",
            features=["grpc", "grpc-reflection"] if use_grpc else None,
            labels={"testing": True, "module": __name__},
            backend=backend,
        )
        yield image_tag
    finally:
        if cleanup:
            print(f"Removing vts server container: {image_tag}")
            subprocess.call([backend, "rmi", image_tag])


@cached_contextmanager("{image_tag}, {config_file}, {use_grpc}")
def run_vts_server_container(
    image_tag: str,
    config_file: str | None = None,
    use_grpc: bool = False,
    timeout: float = 90,
    host: str = "127.0.0.1",
    backend: str = "docker",
):
    """
    Launch a vtsserving service container from a container, yield the host URL
    """
    from vtsserving._internal.configuration.containers import VtsServingContainer

    container_name = f"vtsserving-test-{image_tag}-{hash(config_file)}"
    with reserve_free_port(enable_so_reuseport=use_grpc) as port, reserve_free_port(
        enable_so_reuseport=use_grpc
    ) as prom_port:
        pass

    cmd = [
        backend,
        "run",
        "--rm",
        "--name",
        container_name,
        "--publish",
        f"{port}:3000",
    ]
    if config_file is not None:
        cmd.extend(["--env", "VTSSERVING_CONFIG=/home/vtsserving/vtsserving_config.yml"])
        cmd.extend(
            ["-v", f"{os.path.abspath(config_file)}:/home/vtsserving/vtsserving_config.yml"]
        )
    if use_grpc:
        cmd.extend(
            ["--publish", f"{prom_port}:{VtsServingContainer.grpc.metrics.port.get()}"]
        )
    cmd.append(image_tag)
    serve_cmd = "serve-grpc" if use_grpc else "serve-http"
    cmd.extend([serve_cmd, "--production"])
    print(f"Running API server in container: '{' '.join(cmd)}'")
    with subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        encoding="utf-8",
    ) as proc:
        try:
            host_url = f"{host}:{port}"
            if asyncio.run(
                server_warmup(host_url, timeout=timeout, popen=proc, grpc=use_grpc)
            ):
                yield host_url
            else:
                raise RuntimeError(
                    f"API server {host_url} failed to start within {timeout} seconds"
                ) from None
        finally:
            print(f"Stopping Vts container {container_name}...")
            subprocess.call([backend, "stop", container_name])
    time.sleep(1)


@contextmanager
def run_vts_server_standalone(
    vts: str,
    use_grpc: bool = False,
    config_file: str | None = None,
    timeout: float = 90,
    host: str = "127.0.0.1",
):
    """
    Launch a vtsserving service directly by the vtsserving CLI, yields the host URL.
    """
    copied = os.environ.copy()
    if config_file is not None:
        copied["VTSSERVING_CONFIG"] = os.path.abspath(config_file)
    with reserve_free_port(host=host, enable_so_reuseport=use_grpc) as server_port:
        cmd = [
            sys.executable,
            "-m",
            "vtsserving",
            "serve-grpc" if use_grpc else "serve",
            "--production",
            "--port",
            f"{server_port}",
        ]
        if use_grpc:
            cmd += ["--host", f"{host}"]
    cmd += [vts]
    print(f"Running command: '{' '.join(cmd)}'")
    p = subprocess.Popen(
        cmd,
        stderr=subprocess.STDOUT,
        env=copied,
        encoding="utf-8",
    )
    try:
        host_url = f"{host}:{server_port}"
        assert asyncio.run(
            server_warmup(host_url, timeout=timeout, popen=p, grpc=use_grpc)
        )
        yield host_url
    finally:
        print(f"Stopping process [{p.pid}]...")
        kill_subprocess_tree(p)
        p.communicate()


def start_mitm_proxy(port: int) -> None:
    import uvicorn

    from .utils import http_proxy_app

    print(f"Proxy server listen on {port}")
    uvicorn.run(http_proxy_app, port=port)  # type: ignore (not using ASGI3Application)


@contextmanager
def run_vts_server_distributed(
    vts_tag: str | Tag,
    config_file: str | None = None,
    use_grpc: bool = False,
    timeout: float = 90,
    host: str = "127.0.0.1",
):
    """
    Launch a vtsserving service as a simulated distributed environment(Yatai), yields the host URL.
    """
    import yaml

    import vtsserving

    with reserve_free_port(enable_so_reuseport=use_grpc) as proxy_port:
        pass
    print(f"Starting proxy on port {proxy_port}")
    proxy_process = multiprocessing.Process(
        target=start_mitm_proxy,
        args=(proxy_port,),
    )
    proxy_process.start()
    copied = os.environ.copy()
    # to ensure yatai specified headers BP100
    copied["YATAI_VTS_DEPLOYMENT_NAME"] = "test-deployment"
    copied["YATAI_VTS_DEPLOYMENT_NAMESPACE"] = "yatai"
    if use_grpc:
        copied["GPRC_PROXY"] = f"localhost:{proxy_port}"
    else:
        copied["HTTP_PROXY"] = f"http://127.0.0.1:{proxy_port}"
    if config_file is not None:
        copied["VTSSERVING_CONFIG"] = os.path.abspath(config_file)

    runner_map = {}
    processes: list[subprocess.Popen[str]] = []
    vts_service = vtsserving.vtss.get(vts_tag)
    path = vts_service.path
    with open(os.path.join(path, "vts.yaml"), "r", encoding="utf-8") as f:
        vtsfile = yaml.safe_load(f)
    for runner in vtsfile["runners"]:
        with reserve_free_port(enable_so_reuseport=use_grpc) as port:
            runner_map[runner["name"]] = f"tcp://127.0.0.1:{port}"
            cmd = [
                sys.executable,
                "-m",
                "vtsserving",
                "start-runner-server",
                str(vts_tag),
                "--runner-name",
                runner["name"],
                "--host",
                host,
                "--port",
                f"{port}",
                "--working-dir",
                path,
            ]
            print(f"Running command: '{' '.join(cmd)}'")
        processes.append(
            subprocess.Popen(
                cmd,
                encoding="utf-8",
                stderr=subprocess.STDOUT,
                env=copied,
            )
        )
    runner_args = [
        ("--remote-runner", f"{runner['name']}={runner_map[runner['name']]}")
        for runner in vtsfile["runners"]
    ]
    cmd = [
        sys.executable,
        "-m",
        "vtsserving",
        "start-http-server" if not use_grpc else "start-grpc-server",
        str(vts_tag),
        "--host",
        host,
        "--working-dir",
        path,
        *itertools.chain.from_iterable(runner_args),
    ]
    with reserve_free_port(host=host, enable_so_reuseport=use_grpc) as server_port:
        cmd.extend(["--port", f"{server_port}"])
    print(f"Running command: '{' '.join(cmd)}'")
    processes.append(
        subprocess.Popen(
            cmd,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            env=copied,
        )
    )
    try:
        host_url = f"{host}:{server_port}"
        asyncio.run(server_warmup(host_url, timeout=timeout, grpc=use_grpc))
        yield host_url
    finally:
        for p in processes:
            kill_subprocess_tree(p)
        for p in processes:
            p.communicate()
        if proxy_process is not None:
            proxy_process.terminate()
            proxy_process.join()


@cached_contextmanager(
    "{vts_name}, {project_path}, {config_file}, {deployment_mode}, {vtsserving_home}, {use_grpc}"
)
def host_vts(
    vts_name: str | Tag | None = None,
    project_path: str = ".",
    config_file: str | None = None,
    deployment_mode: t.Literal["standalone", "distributed", "container"] = "standalone",
    vtsserving_home: str | None = None,
    use_grpc: bool = False,
    clean_context: contextlib.ExitStack | None = None,
    host: str = "127.0.0.1",
    timeout: float = 120,
    backend: str = "docker",
) -> t.Generator[str, None, None]:
    """
    Host a vtsserving service, yields the host URL.

    Args:
        vts: a vts tag or :code:`module_path:service`
        project_path: the path to the project directory
        config_file: the path to the config file
        deployment_mode: the deployment mode, one of :code:`standalone`, :code:`docker` or :code:`distributed`
        clean_context: a contextlib.ExitStack to clean up the intermediate files,
                       like docker image and vtss. If None, it will be created. Used for reusing
                       those files in the same test session.
        vtsserving_home: if set, we will change the given VtsServing home folder to :code:`vtsserving_home`. Default
                      to :code:`$HOME/vtsserving`
        use_grpc: if True, running gRPC tests.
        host: set a given host for the vts, default to ``127.0.0.1``
        timeout: the timeout for the server to start
        backend: the backend to use for building container, default to ``docker``

    Returns:
        :obj:`str`: a generated host URL where we run the test vts.
    """
    import vtsserving

    # NOTE: clean_context here to ensure we can close all running context manager
    # to avoid dangling processes. When running pytest, we have a clean_context fixture
    # as a ExitStack.
    if clean_context is None:
        clean_context = contextlib.ExitStack()
        clean_on_exit = True
    else:
        clean_on_exit = False

    # NOTE: we need to set the VTSSERVING_HOME to a temporary folder to avoid
    # conflict with the user's VTSSERVING_HOME.
    if vtsserving_home:
        from vtsserving._internal.configuration.containers import VtsServingContainer

        VtsServingContainer.vtsserving_home.set(vtsserving_home)
    try:
        if vts_name is None or not vtsserving.list(vts_name):
            vts = clean_context.enter_context(
                build(project_path, cleanup=clean_on_exit)
            )
        else:
            vts = vtsserving.get(vts_name)
        print(
            f"Hosting VtsServer '{vts.tag}' in {deployment_mode} mode at '{project_path}'{' with config file '+config_file if config_file else ''}."
        )
        if deployment_mode == "standalone":
            with run_vts_server_standalone(
                vts.path,
                config_file=config_file,
                use_grpc=use_grpc,
                host=host,
                timeout=timeout,
            ) as host_url:
                yield host_url
        elif deployment_mode == "container":
            from vtsserving._internal.container import REGISTERED_BACKENDS

            if backend not in REGISTERED_BACKENDS:
                raise ValueError(
                    f"Unknown backend: {backend}. To register your custom backend, use 'vtsserving.container.register_backend()'"
                )
            container_tag = clean_context.enter_context(
                containerize(vts.tag, use_grpc=use_grpc, backend=backend)
            )
            with run_vts_server_container(
                container_tag,
                config_file=config_file,
                use_grpc=use_grpc,
                host=host,
                timeout=timeout,
                backend=backend,
            ) as host_url:
                yield host_url
        elif deployment_mode == "distributed":
            with run_vts_server_distributed(
                vts.tag,
                config_file=config_file,
                use_grpc=use_grpc,
                host=host,
                timeout=timeout,
            ) as host_url:
                yield host_url
        else:
            raise ValueError(f"Unknown deployment mode: {deployment_mode}") from None
    finally:
        print("Shutting down vts server...")
        if clean_on_exit:
            print("Cleaning on exit...")
            clean_context.close()
