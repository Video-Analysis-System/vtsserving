from __future__ import annotations

import os
import sys
import typing as t
import logging
import importlib
from typing import TYPE_CHECKING

import fs
from simple_di import inject
from simple_di import Provide

from ..vts import Vts
from ..models import ModelStore
from .service import on_import_svc
from .service import on_load_vts
from ...exceptions import NotFound
from ...exceptions import VtsServingException
from ...exceptions import ImportServiceError
from ..vts.vts import VTS_YAML_FILENAME
from ..vts.vts import VTS_PROJECT_DIR_NAME
from ..vts.vts import DEFAULT_VTS_BUILD_FILE
from ..configuration import VTSSERVING_VERSION
from ..vts.build_config import VtsBuildConfig
from ..configuration.containers import VtsServingContainer

if TYPE_CHECKING:
    from ..vts import VtsStore
    from .service import Service

logger = logging.getLogger(__name__)


@inject
def import_service(
    svc_import_path: str,
    *,
    working_dir: t.Optional[str] = None,
    standalone_load: bool = False,
    model_store: ModelStore = Provide[VtsServingContainer.model_store],
) -> Service:
    """Import a Service instance from source code, by providing the svc_import_path
    which represents the module where the Service instance is created and optionally
    what attribute can be used to access this Service instance in that module

    Example usage:
        # When multiple service defined in the same module
        import_service("fraud_detector:svc_a")
        import_service("fraud_detector:svc_b")

        # Find svc by Python module name or file path
        import_service("fraud_detector:svc")
        import_service("fraud_detector.py:svc")
        import_service("foo.bar.fraud_detector:svc")
        import_service("./def/abc/fraud_detector.py:svc")

        # When there's only one Service instance in the target module, the attributes
        # part in the svc_import_path can be omitted
        import_service("fraud_detector.py")
        import_service("fraud_detector")
    """
    from vtsserving import Service

    prev_cwd = None
    sys_path_modified = False
    prev_cwd = os.getcwd()
    global_model_store = VtsServingContainer.model_store.get()

    def recover_standalone_env_change():
        # Reset to previous cwd
        os.chdir(prev_cwd)
        VtsServingContainer.model_store.set(global_model_store)

    try:
        if working_dir is not None:
            working_dir = os.path.realpath(os.path.expanduser(working_dir))
            # Set cwd(current working directory) to the Vts's project directory,
            # which allows user code to read files using relative path
            os.chdir(working_dir)
        else:
            working_dir = os.getcwd()

        if working_dir not in sys.path:
            sys.path.insert(0, working_dir)
            sys_path_modified = True

        if model_store is not global_model_store:
            VtsServingContainer.model_store.set(model_store)

        logger.debug(
            'Importing service "%s" from working dir: "%s"',
            svc_import_path,
            working_dir,
        )

        import_path, _, attrs_str = svc_import_path.partition(":")
        if not import_path:
            raise ImportServiceError(
                f'Invalid import target "{svc_import_path}", must format as '
                '"<module>:<attribute>" or "<module>'
            )

        if os.path.exists(import_path):
            import_path = os.path.realpath(import_path)
            # Importing from a module file path:
            if not import_path.startswith(working_dir):
                raise ImportServiceError(
                    f'Module "{import_path}" not found in working directory "{working_dir}"'
                )

            file_name, ext = os.path.splitext(import_path)
            if ext != ".py":
                raise ImportServiceError(
                    f'Invalid module extension "{ext}" in target "{svc_import_path}",'
                    ' the only extension acceptable here is ".py"'
                )

            # move up until no longer in a python package or in the working dir
            module_name_parts: t.List[str] = []
            path = file_name
            while True:
                path, name = os.path.split(path)
                module_name_parts.append(name)
                if (
                    not os.path.exists(os.path.join(path, "__init__.py"))
                    or path == working_dir
                ):
                    break
            module_name = ".".join(module_name_parts[::-1])
        else:
            # Importing by module name:
            module_name = import_path

        # Import the service using the Vts's own model store
        try:
            module = importlib.import_module(module_name, package=working_dir)
        except ImportError as e:
            raise ImportServiceError(f'Failed to import module "{module_name}": {e}')
        if not standalone_load:
            recover_standalone_env_change()

        if attrs_str:
            instance = module
            try:
                for attr_str in attrs_str.split("."):
                    instance = getattr(instance, attr_str)
            except AttributeError:
                raise ImportServiceError(
                    f'Attribute "{attrs_str}" not found in module "{module_name}".'
                )
        else:
            instances = [
                (k, v) for k, v in module.__dict__.items() if isinstance(v, Service)
            ]

            if len(instances) == 1:
                attrs_str = instances[0][0]
                instance = instances[0][1]
            else:
                raise ImportServiceError(
                    f'Multiple Service instances found in module "{module_name}", use'
                    '"<module>:<svc_variable_name>" to specify the service instance or'
                    "define only service instance per python module/file"
                )

        assert isinstance(
            instance, Service
        ), f'import target "{module_name}:{attrs_str}" is not a vtsserving.Service instance'

        on_import_svc(
            svc=instance,
            working_dir=working_dir,
            import_str=f"{module_name}:{attrs_str}",
        )
        return instance
    except ImportServiceError:
        if sys_path_modified and working_dir:
            # Undo changes to sys.path
            sys.path.remove(working_dir)

        recover_standalone_env_change()
        raise


@inject
def load_vts(
    vts_tag: str,
    vts_store: "VtsStore" = Provide[VtsServingContainer.vts_store],
    standalone_load: bool = False,
) -> "Service":
    """Load a Service instance from a vts found in local vts store:

    Example usage:
        load_vts("FraudDetector:latest")
        load_vts("FraudDetector:20210709_DE14C9")
    """
    vts = vts_store.get(vts_tag)
    logger.debug(
        'Loading vts "%s" found in local store: %s',
        vts.tag,
        vts._fs.getsyspath("/"),
    )

    # not in validate as it's only really necessary when getting vtss from disk
    if vts.info.vtsserving_version != VTSSERVING_VERSION:
        info_vtsserving_version = vts.info.vtsserving_version
        if tuple(info_vtsserving_version.split(".")) > tuple(VTSSERVING_VERSION.split(".")):
            logger.warning(
                "%s was built with newer version of VtsServing, which does not match with current running VtsServing version %s",
                vts,
                VTSSERVING_VERSION,
            )
        else:
            logger.debug(
                "%s was built with VtsServing version %s, which does not match the current VtsServing version %s",
                vts,
                info_vtsserving_version,
                VTSSERVING_VERSION,
            )
    return _load_vts(vts, standalone_load)


def load_vts_dir(path: str, standalone_load: bool = False) -> "Service":
    """Load a Service instance from a vts directory

    Example usage:
        load_vts_dir("~/vtsserving/vtss/iris_classifier/4tht2icroji6zput3suqi5nl2")
    """
    vts_fs = fs.open_fs(path)
    vts = Vts.from_fs(vts_fs)
    logger.debug(
        'Loading vts "%s" from directory: %s',
        vts.tag,
        path,
    )
    return _load_vts(vts, standalone_load)


def _load_vts(vts: Vts, standalone_load: bool) -> "Service":
    # Use Vts's user project path as working directory when importing the service
    working_dir = vts._fs.getsyspath(VTS_PROJECT_DIR_NAME)

    # Use Vts's local "{base_dir}/models/" directory as its model store
    model_store = ModelStore(vts._fs.getsyspath("models"))

    svc = import_service(
        vts.info.service,
        working_dir=working_dir,
        standalone_load=standalone_load,
        model_store=model_store,
    )
    on_load_vts(svc, vts)
    return svc


def load(
    vts_identifier: str,
    working_dir: t.Optional[str] = None,
    standalone_load: bool = False,
) -> "Service":
    """Load a Service instance by the vts_identifier

    Args:
        vts_identifier: target Service to import or Vts to load
        working_dir: when importing from service, set the working_dir
        standalone_load: treat target Service as standalone. This will change global
            current working directory and global model store.


    The argument vts_identifier can be one of the following forms:

    * Tag pointing to a Vts in local Vts store under `VTSSERVING_HOME/vtss`
    * File path to a Vts directory
    * "import_str" for loading a service instance from the `working_dir`

    Example load from Vts usage:

    .. code-block:: python

        # load from local vts store
        load("FraudDetector:latest")
        load("FraudDetector:4tht2icroji6zput")

        # load from vts directory
        load("~/vtsserving/vtss/iris_classifier/4tht2icroji6zput")


    Example load from working directory by "import_str" usage:

    .. code-block:: python

        # When multiple service defined in the same module
        load("fraud_detector:svc_a")
        load("fraud_detector:svc_b")

        # Find svc by Python module name or file path
        load("fraud_detector:svc")
        load("fraud_detector.py:svc")
        load("foo.bar.fraud_detector:svc")
        load("./def/abc/fraud_detector.py:svc")

        # When there's only one Service instance in the target module, the attributes
        # part in the svc_import_path can be omitted
        load("fraud_detector.py")
        load("fraud_detector")

    Limitations when `standalone_load=False`:
    * Models used in the Service being imported, if not accessed during module import,
        must be presented in the global model store
    * Files required for the Service to run, if not accessed during module import, must
        be presented in the current working directory
    """
    if os.path.isdir(os.path.expanduser(vts_identifier)):
        vts_path = os.path.abspath(os.path.expanduser(vts_identifier))

        if os.path.isfile(
            os.path.expanduser(os.path.join(vts_path, VTS_YAML_FILENAME))
        ):
            # Loading from path to a built Vts
            try:
                svc = load_vts_dir(vts_path, standalone_load=standalone_load)
            except ImportServiceError as e:
                raise VtsServingException(
                    f"Failed loading Vts from directory {vts_path}: {e}"
                )
            logger.info("Service loaded from Vts directory: %s", svc)
        elif os.path.isfile(
            os.path.expanduser(os.path.join(vts_path, DEFAULT_VTS_BUILD_FILE))
        ):
            # Loading from path to a project directory containing vtsfile.yaml
            try:
                with open(
                    os.path.join(vts_path, DEFAULT_VTS_BUILD_FILE),
                    "r",
                    encoding="utf-8",
                ) as f:
                    build_config = VtsBuildConfig.from_yaml(f)
                assert (
                    build_config.service
                ), '"service" field in "vtsfile.yaml" is required for loading the service, e.g. "service: my_service.py:svc"'
                svc = import_service(
                    build_config.service,
                    working_dir=working_dir,
                    standalone_load=standalone_load,
                )
            except ImportServiceError as e:
                raise VtsServingException(
                    f"Failed loading Vts from directory {vts_path}: {e}"
                )
            logger.debug("'%s' loaded from '%s': %s", svc.name, vts_path, svc)
        else:
            raise VtsServingException(
                f"Failed loading service from path {vts_path}. When loading from a path, it must be either a Vts containing vts.yaml or a project directory containing vtsfile.yaml"
            )
    else:
        try:
            # Loading from service definition file, e.g. "my_service.py:svc"
            svc = import_service(
                vts_identifier,
                working_dir=working_dir,
                standalone_load=standalone_load,
            )
            logger.debug("'%s' imported from source: %s", svc.name, svc)
        except ImportServiceError as e1:
            try:
                # Loading from local vts store by tag, e.g. "iris_classifier:latest"
                svc = load_vts(vts_identifier, standalone_load=standalone_load)
                logger.debug("'%s' loaded from Vts store: %s", svc.name, svc)
            except (NotFound, ImportServiceError) as e2:
                raise VtsServingException(
                    f"Failed to load vts or import service '{vts_identifier}'.\n"
                    f"If you are attempting to import vts in local store: '{e1}'.\n"
                    f"If you are importing by python module path: '{e2}'."
                )
    return svc
