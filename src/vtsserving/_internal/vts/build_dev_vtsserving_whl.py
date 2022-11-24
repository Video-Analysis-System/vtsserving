from __future__ import annotations

import os
import logging
from pathlib import Path

from ..utils.pkg import source_locations
from ...exceptions import VtsServingException
from ...exceptions import MissingDependencyException
from ..configuration import is_pypi_installed_vtsserving

logger = logging.getLogger(__name__)

VTSSERVING_DEV_BUILD = "VTSSERVING_BUNDLE_LOCAL_BUILD"


def build_vtsserving_editable_wheel(
    target_path: str, *, _internal_stubs_version: str = "v1"
) -> None:
    """
    This is for VtsServing developers to create Bentos that contains the local vtsserving
    build based on their development branch. To enable this behavior, one must
    set envar :code:`VTSSERVING_BUNDLE_LOCAL_BUILD=True` before building a Vts.
    """
    if str(os.environ.get(VTSSERVING_DEV_BUILD, False)).lower() != "true":
        return

    if is_pypi_installed_vtsserving():
        # skip this entirely if VtsServing is installed from PyPI
        return

    try:
        # NOTE: build.env is a standalone library,
        # different from build. However, isort sometimes
        # incorrectly re-order the imports order.
        # isort: off
        from build.env import IsolatedEnvBuilder

        from build import ProjectBuilder

        # isort: on
    except ModuleNotFoundError as e:
        raise MissingDependencyException(
            f"Environment variable '{VTSSERVING_DEV_BUILD}=True', which requires the 'pypa/build' package ({e}). Install development dependencies with 'pip install -r requirements/dev-requirements.txt' and try again."
        ) from None

    # Find vtsserving module path
    # This will be $GIT_ROOT/src/vtsserving
    module_location = source_locations("vtsserving")
    if not module_location:
        raise VtsServingException("Could not find vtsserving module location.")
    vtsserving_path = Path(module_location)

    if not Path(
        module_location, "grpc", _internal_stubs_version, "service_pb2.py"
    ).exists():
        raise ModuleNotFoundError(
            f"Generated stubs for version {_internal_stubs_version} are missing. Make sure to run '{vtsserving_path.as_posix()}/scripts/generate_grpc_stubs.sh {_internal_stubs_version}' beforehand to generate gRPC stubs."
        ) from None

    # location to pyproject.toml
    pyproject = vtsserving_path.parent.parent / "pyproject.toml"

    # this is for VtsServing developer to create Service containing custom development
    # branches of VtsServing library, it is True only when VtsServing module is installed
    # in development mode via "pip install --editable ."
    if os.path.isfile(pyproject):
        logger.info(
            "VtsServing is installed in `editable` mode; building VtsServing distribution with the local VtsServing code base. The built wheel file will be included in the target vts."
        )
        with IsolatedEnvBuilder() as env:
            builder = ProjectBuilder(pyproject.parent)
            builder.python_executable = env.executable
            builder.scripts_dir = env.scripts_dir
            env.install(builder.build_system_requires)
            builder.build(
                "wheel", target_path, config_settings={"--global-option": "--quiet"}
            )
    else:
        logger.info(
            "Custom VtsServing build is detected. For a Vts to use the same build at serving time, add your custom VtsServing build to the pip packages list, e.g. `packages=['git+https://github.com/vtsserving/vtsserving.git@13dfb36']`"
        )
