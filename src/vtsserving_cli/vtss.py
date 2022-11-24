from __future__ import annotations

import sys
import json
import typing as t
import logging
from typing import TYPE_CHECKING

import yaml
import click
from rich.table import Table
from rich.syntax import Syntax

from vtsserving_cli.utils import is_valid_vts_tag
from vtsserving_cli.utils import is_valid_vts_name

if TYPE_CHECKING:
    from click import Group
    from click import Context
    from click import Parameter

logger = logging.getLogger("vtsserving")


def parse_delete_targets_argument_callback(
    ctx: Context, params: Parameter, value: t.Any  # pylint: disable=unused-argument
) -> list[str]:
    if value is None:
        return value
    delete_targets = value.split(",")
    delete_targets = list(map(str.strip, delete_targets))
    for delete_target in delete_targets:
        if not (
            is_valid_vts_tag(delete_target) or is_valid_vts_name(delete_target)
        ):
            raise click.BadParameter(
                f'Bad formatting: "{delete_target}". Please present a valid vts bundle name or "name:version" tag. For list of vts bundles, separate delete targets by ",", for example: "my_service:v1,my_service:v2,classifier"'
            )
    return delete_targets


def add_vts_management_commands(cli: Group):
    from vtsserving import Tag
    from vtsserving.vtss import import_vts
    from vtsserving.vtss import build_vtsfile
    from vtsserving._internal.utils import rich_console as console
    from vtsserving._internal.utils import calc_dir_size
    from vtsserving._internal.utils import human_readable_size
    from vtsserving._internal.utils import display_path_under_home
    from vtsserving._internal.vts.vts import DEFAULT_VTS_BUILD_FILE
    from vtsserving._internal.yatai_client import yatai_client
    from vtsserving._internal.configuration.containers import VtsServingContainer

    vts_store = VtsServingContainer.vts_store.get()

    @cli.command()
    @click.argument("vts_tag", type=click.STRING)
    @click.option(
        "-o",
        "--output",
        type=click.Choice(["json", "yaml", "path"]),
        default="yaml",
    )
    def get(vts_tag: str, output: str) -> None:  # type: ignore (not accessed)
        """Print Vts details by providing the vts_tag.

        \b
        vtsserving get iris_classifier:qojf5xauugwqtgxi
        vtsserving get iris_classifier:qojf5xauugwqtgxi --output=json
        """
        vts = vts_store.get(vts_tag)

        if output == "path":
            console.print(vts.path)
        elif output == "json":
            info = json.dumps(vts.info.to_dict(), indent=2, default=str)
            console.print_json(info)
        else:
            info = yaml.dump(vts.info, indent=2, sort_keys=False)
            console.print(Syntax(info, "yaml"))

    @cli.command(name="list")
    @click.argument("vts_name", type=click.STRING, required=False)
    @click.option(
        "-o",
        "--output",
        type=click.Choice(["json", "yaml", "table"]),
        default="table",
    )
    def list_vtss(vts_name: str, output: str) -> None:  # type: ignore (not accessed)
        """List Bentos in local store

        \b
        # show all vtss saved
        $ vtsserving list

        \b
        # show all verions of vts with the name FraudDetector
        $ vtsserving list FraudDetector
        """
        vtss = vts_store.list(vts_name)
        res = [
            {
                "tag": str(vts.tag),
                "path": display_path_under_home(vts.path),
                "size": human_readable_size(calc_dir_size(vts.path)),
                "creation_time": vts.info.creation_time.astimezone().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            }
            for vts in sorted(
                vtss, key=lambda x: x.info.creation_time, reverse=True
            )
        ]

        if output == "json":
            info = json.dumps(res, indent=2)
            console.print(info)
        elif output == "yaml":
            info = yaml.safe_dump(res, indent=2)
            console.print(Syntax(info, "yaml"))
        else:
            table = Table(box=None)
            table.add_column("Tag")
            table.add_column("Size")
            table.add_column("Creation Time")
            table.add_column("Path")
            for vts in res:
                table.add_row(
                    vts["tag"],
                    vts["size"],
                    vts["creation_time"],
                    vts["path"],
                )
            console.print(table)

    @cli.command()
    @click.argument(
        "delete_targets",
        type=click.STRING,
        callback=parse_delete_targets_argument_callback,
        required=True,
    )
    @click.option(
        "-y",
        "--yes",
        "--assume-yes",
        is_flag=True,
        help="Skip confirmation when deleting a specific vts bundle",
    )
    def delete(delete_targets: list[str], yes: bool) -> None:  # type: ignore (not accessed)
        """Delete Vts in local vts store.

        \b
        Examples:
            * Delete single vts bundle by "name:version", e.g: `vtsserving delete IrisClassifier:v1`
            * Bulk delete all vts bundles with a specific name, e.g.: `vtsserving delete IrisClassifier`
            * Bulk delete multiple vts bundles by name and version, separated by ",", e.g.: `benotml delete Irisclassifier:v1,MyPredictService:v2`
            * Bulk delete without confirmation, e.g.: `vtsserving delete IrisClassifier --yes`
        """

        def delete_target(target: str) -> None:
            tag = Tag.from_str(target)

            if tag.version is None:
                to_delete_vtss = vts_store.list(target)
            else:
                to_delete_vtss = [vts_store.get(tag)]

            for vts in to_delete_vtss:
                if yes:
                    delete_confirmed = True
                else:
                    delete_confirmed = click.confirm(f"delete vts {vts.tag}?")

                if delete_confirmed:
                    vts_store.delete(vts.tag)
                    logger.info("%s deleted.", vts)

        for target in delete_targets:
            delete_target(target)

    @cli.command()
    @click.argument("vts_tag", type=click.STRING)
    @click.argument(
        "out_path",
        type=click.STRING,
        default="",
        required=False,
    )
    def export(vts_tag: str, out_path: str) -> None:  # type: ignore (not accessed)
        """Export a Vts to an external file archive

        \b
        Arguments:
            VTS_TAG: vts identifier
            OUT_PATH: output path of exported vts.

        If out_path argument is not provided, vts is exported to name-version.vts in the current directory.
        Beside the native .vts format, we also support ('tar'), tar.gz ('gz'), tar.xz ('xz'), tar.bz2 ('bz2'), and zip.

        \b
        Examples:
            vtsserving export FraudDetector:20210709_DE14C9
            vtsserving export FraudDetector:20210709_DE14C9 ./my_vts.vts
            vtsserving export FraudDetector:latest ./my_vts.vts
            vtsserving export FraudDetector:latest s3://mybucket/vtss/my_vts.vts
        """
        vts = vts_store.get(vts_tag)
        out_path = vts.export(out_path)
        logger.info("%s exported to %s.", vts, out_path)

    @cli.command(name="import")
    @click.argument("vts_path", type=click.STRING)
    def import_vts_(vts_path: str) -> None:  # type: ignore (not accessed)
        """Import a previously exported Vts archive file

        \b
        Arguments:
            VTS_PATH: path of Vts archive file

        \b
        Examples:
            vtsserving import ./my_vts.vts
            vtsserving import s3://mybucket/vtss/my_vts.vts
        """
        vts = import_vts(vts_path)
        logger.info("%s imported.", vts)

    @cli.command()
    @click.argument("vts_tag", type=click.STRING)
    @click.option(
        "-f",
        "--force",
        is_flag=True,
        default=False,
        help="Force pull from yatai to local and overwrite even if it already exists in local",
    )
    def pull(vts_tag: str, force: bool) -> None:  # type: ignore (not accessed)
        """Pull Vts from a yatai server."""
        yatai_client.pull_vts(vts_tag, force=force)

    @cli.command()
    @click.argument("vts_tag", type=click.STRING)
    @click.option(
        "-f",
        "--force",
        is_flag=True,
        default=False,
        help="Forced push to yatai even if it exists in yatai",
    )
    @click.option(
        "-t",
        "--threads",
        default=10,
        help="Number of threads to use for upload",
    )
    def push(vts_tag: str, force: bool, threads: int) -> None:  # type: ignore (not accessed)
        """Push Vts to a yatai server."""
        vts_obj = vts_store.get(vts_tag)
        if not vts_obj:
            raise click.ClickException(f"Vts {vts_tag} not found in local store")
        yatai_client.push_vts(vts_obj, force=force, threads=threads)

    @cli.command()
    @click.argument("build_ctx", type=click.Path(), default=".")
    @click.option(
        "-f", "--vtsfile", type=click.STRING, default=DEFAULT_VTS_BUILD_FILE
    )
    @click.option("--version", type=click.STRING, default=None)
    def build(build_ctx: str, vtsfile: str, version: str) -> None:  # type: ignore (not accessed)
        """Build a new Vts from current directory."""
        if sys.path[0] != build_ctx:
            sys.path.insert(0, build_ctx)

        build_vtsfile(vtsfile, build_ctx=build_ctx, version=version)
