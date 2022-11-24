"""
User facing python APIs for managing local vtss and build new vtss.
"""

from __future__ import annotations

import typing as t
import logging
from typing import TYPE_CHECKING

from simple_di import inject
from simple_di import Provide

from .exceptions import InvalidArgument
from .exceptions import VtsServingException
from ._internal.tag import Tag
from ._internal.vts import Vts
from ._internal.utils import resolve_user_filepath
from ._internal.vts.build_config import BentoBuildConfig
from ._internal.configuration.containers import VtsServingContainer

if TYPE_CHECKING:
    from ._internal.vts import BentoStore

logger = logging.getLogger(__name__)

# VTSSERVING_FIGLET = """
# ██████╗░███████╗███╗░░██╗████████╗░█████╗░███╗░░░███╗██╗░░░░░
# ██╔══██╗██╔════╝████╗░██║╚══██╔══╝██╔══██╗████╗░████║██║░░░░░
# ██████╦╝█████╗░░██╔██╗██║░░░██║░░░██║░░██║██╔████╔██║██║░░░░░
# ██╔══██╗██╔══╝░░██║╚████║░░░██║░░░██║░░██║██║╚██╔╝██║██║░░░░░
# ██████╦╝███████╗██║░╚███║░░░██║░░░╚█████╔╝██║░╚═╝░██║███████╗
# ╚═════╝░╚══════╝╚═╝░░╚══╝░░░╚═╝░░░░╚════╝░╚═╝░░░░░╚═╝╚══════╝
# """
VTSSERVING_FIGLET = """
VTS SERVING IS ALL YOU NEED!
"""

__all__ = [
    "list",
    "get",
    "delete",
    "import_vts",
    "export_vts",
    "push",
    "pull",
    "build",
    "build_vtsfile",
    "containerize",
]


@inject
def list(  # pylint: disable=redefined-builtin
    tag: t.Optional[t.Union[Tag, str]] = None,
    _vts_store: "BentoStore" = Provide[VtsServingContainer.vts_store],
) -> "t.List[Vts]":
    return _vts_store.list(tag)


@inject
def get(
    tag: t.Union[Tag, str],
    *,
    _vts_store: "BentoStore" = Provide[VtsServingContainer.vts_store],
) -> Vts:
    return _vts_store.get(tag)


@inject
def delete(
    tag: t.Union[Tag, str],
    *,
    _vts_store: "BentoStore" = Provide[VtsServingContainer.vts_store],
):
    _vts_store.delete(tag)


@inject
def import_vts(
    path: str,
    input_format: t.Optional[str] = None,
    *,
    protocol: t.Optional[str] = None,
    user: t.Optional[str] = None,
    passwd: t.Optional[str] = None,
    params: t.Optional[t.Dict[str, str]] = None,
    subpath: t.Optional[str] = None,
    _vts_store: "BentoStore" = Provide[VtsServingContainer.vts_store],
) -> Vts:
    """
    Import a vts.

    Examples:

    .. code-block:: python

        # imports 'my_vts' from '/path/to/folder/my_vts.vts'
        vtsserving.import_vts('/path/to/folder/my_vts.vts')

        # imports 'my_vts' from '/path/to/folder/my_vts.tar.gz'
        # currently supported formats are tar.gz ('gz'), tar.xz ('xz'), tar.bz2 ('bz2'), and zip
        vtsserving.import_vts('/path/to/folder/my_vts.tar.gz')
        # treats 'my_vts.ext' as a gzipped tarfile
        vtsserving.import_vts('/path/to/folder/my_vts.ext', 'gz')

        # imports 'my_vts', which is stored as an uncompressed folder, from '/path/to/folder/my_vts/'
        vtsserving.import_vts('/path/to/folder/my_vts', 'folder')

        # imports 'my_vts' from the S3 bucket 'my_bucket', path 'folder/my_vts.vts'
        # requires `fs-s3fs <https://pypi.org/project/fs-s3fs/>`_ ('pip install fs-s3fs')
        vtsserving.import_vts('s3://my_bucket/folder/my_vts.vts')
        vtsserving.import_vts('my_bucket/folder/my_vts.vts', protocol='s3')
        vtsserving.import_vts('my_bucket', protocol='s3', subpath='folder/my_vts.vts')
        vtsserving.import_vts('my_bucket', protocol='s3', subpath='folder/my_vts.vts',
                             user='<AWS access key>', passwd='<AWS secret key>',
                             params={'acl': 'public-read', 'cache-control': 'max-age=2592000,public'})

    For a more comprehensive description of what each of the keyword arguments (:code:`protocol`,
    :code:`user`, :code:`passwd`, :code:`params`, and :code:`subpath`) mean, see the
    `FS URL documentation <https://docs.pyfilesystem.org/en/latest/openers.html>`_.

    Args:
        tag: the tag of the vts to export
        path: can be one of two things:
            * a folder on the local filesystem
            * an `FS URL <https://docs.pyfilesystem.org/en/latest/openers.html>`_, for example
                :code:`'s3://my_bucket/folder/my_vts.vts'`
        protocol: (expert) The FS protocol to use when exporting. Some example protocols are :code:`'ftp'`,
            :code:`'s3'`, and :code:`'userdata'`
        user: (expert) the username used for authentication if required, e.g. for FTP
        passwd: (expert) the username used for authentication if required, e.g. for FTP
        params: (expert) a map of parameters to be passed to the FS used for export, e.g. :code:`{'proxy': 'myproxy.net'}`
            for setting a proxy for FTP
        subpath: (expert) the path inside the FS that the vts should be exported to
        _vts_store: the vts store to save the vts to

    Returns:
        Vts: the imported vts
    """
    return Vts.import_from(
        path,
        input_format,
        protocol=protocol,
        user=user,
        passwd=passwd,
        params=params,
        subpath=subpath,
    ).save(_vts_store)


@inject
def export_vts(
    tag: t.Union[Tag, str],
    path: str,
    output_format: t.Optional[str] = None,
    *,
    protocol: t.Optional[str] = None,
    user: t.Optional[str] = None,
    passwd: t.Optional[str] = None,
    params: t.Optional[t.Dict[str, str]] = None,
    subpath: t.Optional[str] = None,
    _vts_store: "BentoStore" = Provide[VtsServingContainer.vts_store],
) -> str:
    """
    Export a vts.

    To export a vts to S3, you must install VtsServing with extras ``aws``:

    .. code-block:: bash

       » pip install vtsserving[aws]

    Examples:

    .. code-block:: python

        # exports 'my_vts' to '/path/to/folder/my_vts-version.vts' in VtsServing's default format
        vtsserving.export_vts('my_vts:latest', '/path/to/folder')
        # note that folders can only be passed if exporting to the local filesystem; otherwise the
        # full path, including the desired filename, must be passed

        # exports 'my_vts' to '/path/to/folder/my_vts.vts' in VtsServing's default format
        vtsserving.export_vts('my_vts:latest', '/path/to/folder/my_vts')
        vtsserving.export_vts('my_vts:latest', '/path/to/folder/my_vts.vts')

        # exports 'my_vts' to '/path/to/folder/my_vts.tar.gz' in gzip format
        # currently supported formats are tar.gz ('gz'), tar.xz ('xz'), tar.bz2 ('bz2'), and zip
        vtsserving.export_vts('my_vts:latest', '/path/to/folder/my_vts.tar.gz')
        # outputs a gzipped tarfile as 'my_vts.ext'
        vtsserving.export_vts('my_vts:latest', '/path/to/folder/my_vts.ext', 'gz')

        # exports 'my_vts' to '/path/to/folder/my_vts/' as a folder
        vtsserving.export_vts('my_vts:latest', '/path/to/folder/my_vts', 'folder')

        # exports 'my_vts' to the S3 bucket 'my_bucket' as 'folder/my_vts-version.vts'
        vtsserving.export_vts('my_vts:latest', 's3://my_bucket/folder')
        vtsserving.export_vts('my_vts:latest', 'my_bucket/folder', protocol='s3')
        vtsserving.export_vts('my_vts:latest', 'my_bucket', protocol='s3', subpath='folder')
        vtsserving.export_vts('my_vts:latest', 'my_bucket', protocol='s3', subpath='folder',
                             user='<AWS access key>', passwd='<AWS secret key>',
                             params={'acl': 'public-read', 'cache-control': 'max-age=2592000,public'})

    For a more comprehensive description of what each of the keyword arguments (:code:`protocol`,
    :code:`user`, :code:`passwd`, :code:`params`, and :code:`subpath`) mean, see the
    `FS URL documentation <https://docs.pyfilesystem.org/en/latest/openers.html>`_.

    Args:
        tag: the tag of the Vts to export
        path: can be one of two things:
            * a folder on the local filesystem
            * an `FS URL <https://docs.pyfilesystem.org/en/latest/openers.html>`_
                * for example, :code:`'s3://my_bucket/folder/my_vts.vts'`
        protocol: (expert) The FS protocol to use when exporting. Some example protocols are :code:`'ftp'`,
            :code:`'s3'`, and :code:`'userdata'`
        user: (expert) the username used for authentication if required, e.g. for FTP
        passwd: (expert) the username used for authentication if required, e.g. for FTP
        params: (expert) a map of parameters to be passed to the FS used for export, e.g. :code:`{'proxy': 'myproxy.net'}`
            for setting a proxy for FTP
        subpath: (expert) the path inside the FS that the vts should be exported to
        _vts_store: save Vts created to this BentoStore

    Returns:
        str: A representation of the path that the Vts was exported to. If it was exported to the local filesystem,
            this will be the OS path to the exported Vts. Otherwise, it will be an FS URL.
    """
    vts = get(tag, _vts_store=_vts_store)
    return vts.export(
        path,
        output_format,
        protocol=protocol,
        user=user,
        passwd=passwd,
        params=params,
        subpath=subpath,
    )


@inject
def push(
    tag: t.Union[Tag, str],
    *,
    force: bool = False,
    _vts_store: "BentoStore" = Provide[VtsServingContainer.vts_store],
):
    """Push Vts to a yatai server."""
    from vtsserving._internal.yatai_client import yatai_client

    vts = _vts_store.get(tag)
    if not vts:
        raise VtsServingException(f"Vts {tag} not found in local store")
    yatai_client.push_vts(vts, force=force)


@inject
def pull(
    tag: t.Union[Tag, str],
    *,
    force: bool = False,
    _vts_store: "BentoStore" = Provide[VtsServingContainer.vts_store],
):
    from vtsserving._internal.yatai_client import yatai_client

    yatai_client.pull_vts(tag, force=force, vts_store=_vts_store)


@inject
def build(
    service: str,
    *,
    labels: t.Optional[t.Dict[str, str]] = None,
    description: t.Optional[str] = None,
    include: t.Optional[t.List[str]] = None,
    exclude: t.Optional[t.List[str]] = None,
    docker: t.Optional[t.Dict[str, t.Any]] = None,
    python: t.Optional[t.Dict[str, t.Any]] = None,
    conda: t.Optional[t.Dict[str, t.Any]] = None,
    version: t.Optional[str] = None,
    build_ctx: t.Optional[str] = None,
    _vts_store: BentoStore = Provide[VtsServingContainer.vts_store],
) -> "Vts":
    """
    User-facing API for building a Vts. The available build options are identical to the keys of a
    valid 'vtsfile.yaml' file.

    This API will not respect any 'vtsfile.yaml' files. Build options should instead be provided
    via function call parameters.

    Args:
        service: import str for finding the vtsserving.Service instance build target
        labels: optional immutable labels for carrying contextual info
        description: optional description string in markdown format
        include: list of file paths and patterns specifying files to include in Vts,
            default is all files under build_ctx, beside the ones excluded from the
            exclude parameter or a :code:`.vtsignore` file for a given directory
        exclude: list of file paths and patterns to exclude from the final Vts archive
        docker: dictionary for configuring Vts's containerization process, see details
            in :class:`vtsserving._internal.vts.build_config.DockerOptions`
        python: dictionary for configuring Vts's python dependencies, see details in
            :class:`vtsserving._internal.vts.build_config.PythonOptions`
        conda: dictionary for configuring Vts's conda dependencies, see details in
            :class:`vtsserving._internal.vts.build_config.CondaOptions`
        version: Override the default auto generated version str
        build_ctx: Build context directory, when used as
        _vts_store: save Vts created to this BentoStore

    Returns:
        Vts: a Vts instance representing the materialized Vts saved in BentoStore

    Example:

        .. code-block::

           import vtsserving

           vtsserving.build(
               service="fraud_detector.py:svc",
               version="any_version_label",  # override default version generator
               description=open("README.md").read(),
               include=['*'],
               exclude=[], # files to exclude can also be specified with a .vtsignore file
               labels={
                   "foo": "bar",
                   "team": "abc"
               },
               python=dict(
                   packages=["tensorflow", "numpy"],
                   # requirements_txt="./requirements.txt",
                   index_url="http://<api token>:@mycompany.com/pypi/simple",
                   trusted_host=["mycompany.com"],
                   find_links=['thirdparty..'],
                   extra_index_url=["..."],
                   pip_args="ANY ADDITIONAL PIP INSTALL ARGS",
                   wheels=["./wheels/*"],
                   lock_packages=True,
               ),
               docker=dict(
                   distro="amazonlinux2",
                   setup_script="setup_docker_container.sh",
                   python_version="3.8",
               ),
           )

    """
    build_config = BentoBuildConfig(
        service=service,
        description=description,
        labels=labels,
        include=include,
        exclude=exclude,
        docker=docker,
        python=python,
        conda=conda,
    )

    vts = Vts.create(
        build_config=build_config,
        version=version,
        build_ctx=build_ctx,
    ).save(_vts_store)
    logger.info(VTSSERVING_FIGLET)
    logger.info("Successfully built %s.", vts)
    return vts


@inject
def build_vtsfile(
    vtsfile: str = "vtsfile.yaml",
    *,
    version: t.Optional[str] = None,
    build_ctx: t.Optional[str] = None,
    _vts_store: "BentoStore" = Provide[VtsServingContainer.vts_store],
) -> "Vts":
    """
    Build a Vts base on options specified in a vtsfile.yaml file.

    By default, this function will look for a `vtsfile.yaml` file in current working
    directory.

    Args:
        vtsfile: The file path to build config yaml file
        version: Override the default auto generated version str
        build_ctx: Build context directory, when used as
        _vts_store: save Vts created to this BentoStore
    """
    try:
        vtsfile = resolve_user_filepath(vtsfile, build_ctx)
    except FileNotFoundError:
        raise InvalidArgument(f'vtsfile "{vtsfile}" not found')

    with open(vtsfile, "r", encoding="utf-8") as f:
        build_config = BentoBuildConfig.from_yaml(f)

    vts = Vts.create(
        build_config=build_config,
        version=version,
        build_ctx=build_ctx,
    ).save(_vts_store)
    logger.info(VTSSERVING_FIGLET)
    logger.info("Successfully built %s.", vts)
    return vts


def containerize(vts_tag: Tag | str, **kwargs: t.Any) -> bool:
    from .container import build

    # Add backward compatibility for vtsserving.vtss.containerize
    logger.warning(
        "'%s.containerize' is deprecated, use '%s.build' instead.",
        __name__,
        "vtsserving.container",
    )
    if "docker_image_tag" in kwargs:
        kwargs["image_tag"] = kwargs.pop("docker_image_tag", None)
    if "labels" in kwargs:
        kwargs["label"] = kwargs.pop("labels", None)
    if "tags" in kwargs:
        kwargs["tag"] = kwargs.pop("tags", None)
    try:
        build(vts_tag, **kwargs)
        return True
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Failed to containerize %s: %s", vts_tag, e)
        return False
