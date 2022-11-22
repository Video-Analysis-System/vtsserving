from __future__ import annotations

import os
import sys
import logging

import click

logger = logging.getLogger(__name__)

DEFAULT_DEV_SERVER_HOST = "127.0.0.1"


def add_serve_command(cli: click.Group) -> None:

    from vtsserving._internal.log import configure_server_logging
    from vtsserving._internal.configuration.containers import VtsServingContainer

    @cli.command(aliases=["serve-http"])
    @click.argument("vts", type=click.STRING, default=".")
    @click.option(
        "--production",
        type=click.BOOL,
        help="Run the BentoServer in production mode",
        is_flag=True,
        default=False,
        show_default=True,
    )
    @click.option(
        "-p",
        "--port",
        type=click.INT,
        default=VtsServingContainer.http.port.get(),
        help="The port to listen on for the REST api server",
        envvar="VTSSERVING_PORT",
        show_default=True,
    )
    @click.option(
        "--host",
        type=click.STRING,
        default=VtsServingContainer.http.host.get(),
        help="The host to bind for the REST api server",
        envvar="VTSSERVING_HOST",
        show_default=True,
    )
    @click.option(
        "--api-workers",
        type=click.INT,
        default=VtsServingContainer.api_server_workers.get(),
        help="Specify the number of API server workers to start. Default to number of available CPU cores in production mode",
        envvar="VTSSERVING_API_WORKERS",
        show_default=True,
    )
    @click.option(
        "--backlog",
        type=click.INT,
        default=VtsServingContainer.api_server_config.backlog.get(),
        help="The maximum number of pending connections.",
        show_default=True,
    )
    @click.option(
        "--reload",
        type=click.BOOL,
        is_flag=True,
        help="Reload Service when code changes detected, this is only available in development mode",
        default=False,
        show_default=True,
    )
    @click.option(
        "--working-dir",
        type=click.Path(),
        help="When loading from source code, specify the directory to find the Service instance",
        default=None,
        show_default=True,
    )
    @click.option(
        "--ssl-certfile",
        type=str,
        default=None,
        help="SSL certificate file",
        show_default=True,
    )
    @click.option(
        "--ssl-keyfile",
        type=str,
        default=None,
        help="SSL key file",
        show_default=True,
    )
    @click.option(
        "--ssl-keyfile-password",
        type=str,
        default=None,
        help="SSL keyfile password",
        show_default=True,
    )
    @click.option(
        "--ssl-version",
        type=int,
        default=None,
        help="SSL version to use (see stdlib 'ssl' module)",
        show_default=True,
    )
    @click.option(
        "--ssl-cert-reqs",
        type=int,
        default=None,
        help="Whether client certificate is required (see stdlib 'ssl' module)",
        show_default=True,
    )
    @click.option(
        "--ssl-ca-certs",
        type=str,
        default=None,
        help="CA certificates file",
        show_default=True,
    )
    @click.option(
        "--ssl-ciphers",
        type=str,
        default=None,
        help="Ciphers to use (see stdlib 'ssl' module)",
        show_default=True,
    )
    def serve(  # type: ignore (unused warning)
        vts: str,
        production: bool,
        port: int,
        host: str,
        api_workers: int | None,
        backlog: int,
        reload: bool,
        working_dir: str,
        ssl_certfile: str | None,
        ssl_keyfile: str | None,
        ssl_keyfile_password: str | None,
        ssl_version: int | None,
        ssl_cert_reqs: int | None,
        ssl_ca_certs: str | None,
        ssl_ciphers: str | None,
    ) -> None:
        """Start a HTTP BentoServer from a given 🍱

        \b
        VTS is the serving target, it can be the import as:
        - the import path of a 'vtsserving.Service' instance
        - a tag to a Bento in local Bento store
        - a folder containing a valid 'vtsfile.yaml' build file with a 'service' field, which provides the import path of a 'vtsserving.Service' instance
        - a path to a built Bento (for internal & debug use only)

        e.g.:

        \b
        Serve from a vtsserving.Service instance source code (for development use only):
            'vtsserving serve fraud_detector.py:svc'

        \b
        Serve from a Bento built in local store:
            'vtsserving serve fraud_detector:4tht2icroji6zput3suqi5nl2'
            'vtsserving serve fraud_detector:latest'

        \b
        Serve from a Bento directory:
            'vtsserving serve ./fraud_detector_vts'

        \b
        If '--reload' is provided, VtsServing will detect code and model store changes during development, and restarts the service automatically.

        \b
        The '--reload' flag will:
        - be default, all file changes under '--working-dir' (default to current directory) will trigger a restart
        - when specified, respect 'include' and 'exclude' under 'vtsfile.yaml' as well as the '.vtsignore' file in '--working-dir', for code and file changes
        - all model store changes will also trigger a restart (new model saved or existing model removed)
        """
        configure_server_logging()
        if working_dir is None:
            if os.path.isdir(os.path.expanduser(vts)):
                working_dir = os.path.expanduser(vts)
            else:
                working_dir = "."
        if sys.path[0] != working_dir:
            sys.path.insert(0, working_dir)

        if production:
            if reload:
                logger.warning(
                    "'--reload' is not supported with '--production'; ignoring"
                )

            from vtsserving.serve import serve_http_production

            serve_http_production(
                vts,
                working_dir=working_dir,
                port=port,
                host=host,
                backlog=backlog,
                api_workers=api_workers,
                ssl_keyfile=ssl_keyfile,
                ssl_certfile=ssl_certfile,
                ssl_keyfile_password=ssl_keyfile_password,
                ssl_version=ssl_version,
                ssl_cert_reqs=ssl_cert_reqs,
                ssl_ca_certs=ssl_ca_certs,
                ssl_ciphers=ssl_ciphers,
            )
        else:
            from vtsserving.serve import serve_http_development

            serve_http_development(
                vts,
                working_dir=working_dir,
                port=port,
                host=DEFAULT_DEV_SERVER_HOST if not host else host,
                reload=reload,
                ssl_keyfile=ssl_keyfile,
                ssl_certfile=ssl_certfile,
                ssl_keyfile_password=ssl_keyfile_password,
                ssl_version=ssl_version,
                ssl_cert_reqs=ssl_cert_reqs,
                ssl_ca_certs=ssl_ca_certs,
                ssl_ciphers=ssl_ciphers,
            )

    from vtsserving._internal.utils import add_experimental_docstring

    @cli.command(name="serve-grpc")
    @click.argument("vts", type=click.STRING, default=".")
    @click.option(
        "--production",
        type=click.BOOL,
        help="Run the BentoServer in production mode",
        is_flag=True,
        default=False,
        show_default=True,
    )
    @click.option(
        "-p",
        "--port",
        type=click.INT,
        default=VtsServingContainer.grpc.port.get(),
        help="The port to listen on for the REST api server",
        envvar="VTSSERVING_PORT",
        show_default=True,
    )
    @click.option(
        "--host",
        type=click.STRING,
        default=VtsServingContainer.grpc.host.get(),
        help="The host to bind for the gRPC server",
        envvar="VTSSERVING_HOST",
        show_default=True,
    )
    @click.option(
        "--api-workers",
        type=click.INT,
        default=VtsServingContainer.api_server_workers.get(),
        help="Specify the number of API server workers to start. Default to number of available CPU cores in production mode",
        envvar="VTSSERVING_API_WORKERS",
        show_default=True,
    )
    @click.option(
        "--reload",
        type=click.BOOL,
        is_flag=True,
        help="Reload Service when code changes detected, this is only available in development mode",
        default=False,
        show_default=True,
    )
    @click.option(
        "--backlog",
        type=click.INT,
        default=VtsServingContainer.api_server_config.backlog.get(),
        help="The maximum number of pending connections.",
        show_default=True,
    )
    @click.option(
        "--working-dir",
        type=click.Path(),
        help="When loading from source code, specify the directory to find the Service instance",
        default=None,
        show_default=True,
    )
    @click.option(
        "--enable-reflection",
        is_flag=True,
        default=VtsServingContainer.grpc.reflection.enabled.get(),
        type=click.BOOL,
        help="Enable reflection.",
        show_default=True,
    )
    @click.option(
        "--enable-channelz",
        is_flag=True,
        default=VtsServingContainer.grpc.channelz.enabled.get(),
        type=click.BOOL,
        help="Enable Channelz. See https://github.com/grpc/proposal/blob/master/A14-channelz.md.",
    )
    @click.option(
        "--max-concurrent-streams",
        default=VtsServingContainer.grpc.max_concurrent_streams.get(),
        type=click.INT,
        help="Maximum number of concurrent incoming streams to allow on a http2 connection.",
        show_default=True,
    )
    @click.option(
        "--ssl-certfile",
        type=str,
        default=None,
        help="SSL certificate file",
        show_default=True,
    )
    @click.option(
        "--ssl-keyfile",
        type=str,
        default=None,
        help="SSL key file",
        show_default=True,
    )
    @click.option(
        "--ssl-ca-certs",
        type=str,
        default=None,
        help="CA certificates file",
        show_default=True,
    )
    @add_experimental_docstring
    def serve_grpc(  # type: ignore (unused warning)
        vts: str,
        production: bool,
        port: int,
        host: str,
        api_workers: int | None,
        backlog: int,
        reload: bool,
        working_dir: str,
        ssl_certfile: str | None,
        ssl_keyfile: str | None,
        ssl_ca_certs: str | None,
        enable_reflection: bool,
        enable_channelz: bool,
        max_concurrent_streams: int | None,
    ):
        """Start a gRPC BentoServer from a given 🍱

        \b
        VTS is the serving target, it can be the import as:
        - the import path of a 'vtsserving.Service' instance
        - a tag to a Bento in local Bento store
        - a folder containing a valid 'vtsfile.yaml' build file with a 'service' field, which provides the import path of a 'vtsserving.Service' instance
        - a path to a built Bento (for internal & debug use only)

        e.g.:

        \b
        Serve from a vtsserving.Service instance source code (for development use only):
            'vtsserving serve-grpc fraud_detector.py:svc'

        \b
        Serve from a Bento built in local store:
            'vtsserving serve-grpc fraud_detector:4tht2icroji6zput3suqi5nl2'
            'vtsserving serve-grpc fraud_detector:latest'

        \b
        Serve from a Bento directory:
            'vtsserving serve-grpc ./fraud_detector_vts'

        If '--reload' is provided, VtsServing will detect code and model store changes during development, and restarts the service automatically.

        \b
        The '--reload' flag will:
        - be default, all file changes under '--working-dir' (default to current directory) will trigger a restart
        - when specified, respect 'include' and 'exclude' under 'vtsfile.yaml' as well as the '.vtsignore' file in '--working-dir', for code and file changes
        - all model store changes will also trigger a restart (new model saved or existing model removed)
        """
        configure_server_logging()
        if working_dir is None:
            if os.path.isdir(os.path.expanduser(vts)):
                working_dir = os.path.expanduser(vts)
            else:
                working_dir = "."
        if production:
            if reload:
                logger.warning(
                    "'--reload' is not supported with '--production'; ignoring"
                )

            from vtsserving.serve import serve_grpc_production

            serve_grpc_production(
                vts,
                working_dir=working_dir,
                port=port,
                host=host,
                backlog=backlog,
                api_workers=api_workers,
                ssl_keyfile=ssl_keyfile,
                ssl_certfile=ssl_certfile,
                ssl_ca_certs=ssl_ca_certs,
                max_concurrent_streams=max_concurrent_streams,
                reflection=enable_reflection,
                channelz=enable_channelz,
            )
        else:
            from vtsserving.serve import serve_grpc_development

            serve_grpc_development(
                vts,
                working_dir=working_dir,
                port=port,
                backlog=backlog,
                reload=reload,
                host=DEFAULT_DEV_SERVER_HOST if not host else host,
                ssl_keyfile=ssl_keyfile,
                ssl_certfile=ssl_certfile,
                ssl_ca_certs=ssl_ca_certs,
                max_concurrent_streams=max_concurrent_streams,
                reflection=enable_reflection,
                channelz=enable_channelz,
            )
