# pylint: disable=unused-argument
from __future__ import annotations

import typing as t
import logging
from typing import TYPE_CHECKING

import yaml
import pytest
import cloudpickle

import vtsserving
from vtsserving.testing.pytest import TEST_MODEL_CONTEXT

if TYPE_CHECKING:
    from pathlib import Path

    from _pytest.fixtures import FixtureRequest


@pytest.fixture(scope="function")
def reload_directory(
    request: FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> t.Generator[Path, None, None]:
    """
    This fixture will create an example vtsserving working file directory
    and yield the results directory
    ./
    ├── models/  # mock default vtsserving home models directory
    ├── [fdir, fdir_one, fdir_two]/
    │   ├── README.md
        ├── subdir/
        │   ├── README.md
    │   │   └── app.py
    │   ├── somerust.rs
    │   └── app.py
    ├── README.md
    ├── .vtsignore
    ├── vtsfile.yaml
    ├── fname.ipynb
    ├── requirements.txt
    ├── service.py
    └── train.py
    """
    from vtsserving._internal.utils import vtsserving_cattr
    from vtsserving._internal.vts.build_config import VtsBuildConfig

    root = tmp_path_factory.mktemp("reload_directory")
    # create a models directory
    root.joinpath("models").mkdir()

    # enable this fixture to use with unittest.TestCase
    if request.cls is not None:
        request.cls.reload_directory = root

    root_file = [
        "README.md",
        "requirements.txt",
        "service.py",
        "train.py",
        "fname.ipynb",
    ]

    for f in root_file:
        p = root.joinpath(f)
        p.touch()
    build_config = VtsBuildConfig(
        service="service.py:svc",
        description="A mock service",
        exclude=["*.rs"],
    ).with_defaults()
    vtsfile = root / "vtsfile.yaml"
    vtsfile.touch()
    with vtsfile.open("w", encoding="utf-8") as f:
        yaml.safe_dump(vtsserving_cattr.unstructure(build_config), f)

    custom_library = ["fdir", "fdir_one", "fdir_two"]
    for app in custom_library:
        ap = root.joinpath(app)
        ap.mkdir()
        dir_files: list[tuple[str, list[t.Any]]] = [
            ("README.md", []),
            ("subdir", ["README.md", "app.py"]),
            ("lib.rs", []),
            ("app.py", []),
        ]
        for name, maybe_files in dir_files:
            if maybe_files:
                dpath = ap.joinpath(name)
                dpath.mkdir()
                for f in maybe_files:
                    p = dpath.joinpath(f)
                    p.touch()
            else:
                p = ap.joinpath(name)
                p.touch()

    yield root


@pytest.fixture(scope="session")
def simple_service() -> vtsserving.Service:
    """
    This fixture create a simple service implementation that implements a noop runnable with two APIs:

    - noop_sync: sync API that returns the input.
    - invalid: an invalid API that can be used to test error handling.
    """
    from vtsserving.io import Text

    class NoopModel:
        def predict(self, data: t.Any) -> t.Any:
            return data

    with vtsserving.models.create(
        "python_function",
        context=TEST_MODEL_CONTEXT,
        module=__name__,
        signatures={"predict": {"batchable": True}},
    ) as model:
        with open(model.path_of("test.pkl"), "wb") as f:
            cloudpickle.dump(NoopModel(), f)

    model_ref = vtsserving.models.get("python_function")

    class NoopRunnable(vtsserving.Runnable):
        SUPPORTED_RESOURCES = ("cpu",)
        SUPPORTS_CPU_MULTI_THREADING = True

        def __init__(self):
            self._model: NoopModel = vtsserving.picklable_model.load_model(model_ref)

        @vtsserving.Runnable.method(batchable=True)
        def predict(self, data: t.Any) -> t.Any:
            return self._model.predict(data)

    svc = vtsserving.Service(
        name="simple_service",
        runners=[vtsserving.Runner(NoopRunnable, models=[model_ref])],
    )

    @svc.api(input=Text(), output=Text())
    def noop_sync(data: str) -> str:  # type: ignore
        return data

    @svc.api(input=Text(), output=Text())
    def invalid(data: str) -> str:  # type: ignore
        raise NotImplementedError

    return svc


@pytest.fixture(scope="function", name="propagate_logs")
def fixture_propagate_logs() -> t.Generator[None, None, None]:
    """VtsServing sets propagate to False by default, hence this fixture enable log propagation."""
    logger = logging.getLogger("vtsserving")
    logger.propagate = True
    yield
    # restore propagate to False after tests
    logger.propagate = False
