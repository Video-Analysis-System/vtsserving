from __future__ import annotations

import socket

import click


@click.command()
@click.argument("vts_identifier", type=click.STRING, required=False, default=".")
@click.option("--fd", type=click.INT, required=True)
@click.option("--working-dir", required=False, type=click.Path(), default=None)
@click.option("--backlog", type=click.INT, default=2048)
@click.option(
    "--prometheus-dir",
    type=click.Path(exists=True),
    help="Required by prometheus to pass the metrics in multi-process mode",
)
@click.option(
    "--ssl-certfile",
    type=str,
    default=None,
    help="SSL certificate file",
)
@click.option(
    "--ssl-keyfile",
    type=str,
    default=None,
    help="SSL key file",
)
@click.option(
    "--ssl-keyfile-password",
    type=str,
    default=None,
    help="SSL keyfile password",
)
@click.option(
    "--ssl-version",
    type=int,
    default=None,
    help="SSL version to use (see stdlib 'ssl' module)",
)
@click.option(
    "--ssl-cert-reqs",
    type=int,
    default=None,
    help="Whether client certificate is required (see stdlib 'ssl' module)",
)
@click.option(
    "--ssl-ca-certs",
    type=str,
    default=None,
    help="CA certificates file",
)
@click.option(
    "--ssl-ciphers",
    type=str,
    default=None,
    help="Ciphers to use (see stdlib 'ssl' module)",
)
def main(
    vts_identifier: str,
    fd: int,
    working_dir: str | None,
    backlog: int,
    prometheus_dir: str | None,
    ssl_certfile: str | None,
    ssl_keyfile: str | None,
    ssl_keyfile_password: str | None,
    ssl_version: int | None,
    ssl_cert_reqs: int | None,
    ssl_ca_certs: str | None,
    ssl_ciphers: str | None,
):
    """
    Start a development server for the VtsServing service.
    """

    import psutil
    import uvicorn

    from vtsserving import load
    from vtsserving._internal.log import configure_server_logging
    from vtsserving._internal.context import component_context
    from vtsserving._internal.configuration.containers import VtsServingContainer

    component_context.component_type = "dev_api_server"
    configure_server_logging()

    if prometheus_dir is not None:
        VtsServingContainer.prometheus_multiproc_dir.set(prometheus_dir)

    svc = load(vts_identifier, working_dir=working_dir, standalone_load=True)

    # setup context
    component_context.component_name = svc.name
    if svc.tag is None:
        component_context.vts_name = svc.name
        component_context.vts_version = "not available"
    else:
        component_context.vts_name = svc.tag.name
        component_context.vts_version = svc.tag.version or "not available"

    sock = socket.socket(fileno=fd)

    uvicorn_options = {
        "backlog": backlog,
        "log_config": None,
        "workers": 1,
        "lifespan": "on",
    }

    if ssl_certfile:
        import ssl

        uvicorn_options["ssl_certfile"] = ssl_certfile
        if ssl_keyfile:
            uvicorn_options["ssl_keyfile"] = ssl_keyfile
        if ssl_keyfile_password:
            uvicorn_options["ssl_keyfile_password"] = ssl_keyfile_password
        if ssl_ca_certs:
            uvicorn_options["ssl_ca_certs"] = ssl_ca_certs

        if not ssl_version:
            ssl_version = ssl.PROTOCOL_TLS_SERVER
            uvicorn_options["ssl_version"] = ssl_version
        if not ssl_cert_reqs:
            ssl_cert_reqs = ssl.CERT_NONE
            uvicorn_options["ssl_cert_reqs"] = ssl_cert_reqs
        if not ssl_ciphers:
            ssl_ciphers = "TLSv1"
            uvicorn_options["ssl_ciphers"] = ssl_ciphers

    if psutil.WINDOWS:
        uvicorn_options["loop"] = "asyncio"
        import asyncio

        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore

    config = uvicorn.Config(svc.asgi_app, **uvicorn_options)
    uvicorn.Server(config).run(sockets=[sock])


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
