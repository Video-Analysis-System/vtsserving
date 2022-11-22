from __future__ import annotations

import click
import psutil

from vtsserving_cli.env import add_env_command
from vtsserving_cli.serve import add_serve_command
from vtsserving_cli.start import add_start_command
from vtsserving_cli.utils import VtsServingCommandGroup
from vtsserving_cli.yatai import add_login_command
from vtsserving_cli.vtss import add_vts_management_commands
from vtsserving_cli.models import add_model_management_commands
from vtsserving_cli.containerize import add_containerize_command


def create_vtsserving_cli() -> click.Group:

    from vtsserving import __version__ as VTSSERVING_VERSION
    from vtsserving._internal.context import component_context

    component_context.component_type = "cli"

    CONTEXT_SETTINGS = {"help_option_names": ("-h", "--help")}

    @click.group(cls=VtsServingCommandGroup, context_settings=CONTEXT_SETTINGS)
    @click.version_option(VTSSERVING_VERSION, "-v", "--version")
    def vtsserving_cli():
        """
        \b
        ██████╗░███████╗███╗░░██╗████████╗░█████╗░███╗░░░███╗██╗░░░░░
        ██╔══██╗██╔════╝████╗░██║╚══██╔══╝██╔══██╗████╗░████║██║░░░░░
        ██████╦╝█████╗░░██╔██╗██║░░░██║░░░██║░░██║██╔████╔██║██║░░░░░
        ██╔══██╗██╔══╝░░██║╚████║░░░██║░░░██║░░██║██║╚██╔╝██║██║░░░░░
        ██████╦╝███████╗██║░╚███║░░░██║░░░╚█████╔╝██║░╚═╝░██║███████╗
        ╚═════╝░╚══════╝╚═╝░░╚══╝░░░╚═╝░░░░╚════╝░╚═╝░░░░░╚═╝╚══════╝
        """

    # Add top-level CLI commands
    add_env_command(vtsserving_cli)
    add_login_command(vtsserving_cli)
    add_vts_management_commands(vtsserving_cli)
    add_model_management_commands(vtsserving_cli)
    add_start_command(vtsserving_cli)
    add_serve_command(vtsserving_cli)
    add_containerize_command(vtsserving_cli)

    if psutil.WINDOWS:
        import sys

        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore

    return vtsserving_cli


cli = create_vtsserving_cli()


if __name__ == "__main__":
    cli()
