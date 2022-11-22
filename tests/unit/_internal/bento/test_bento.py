# pylint: disable=unused-argument
from __future__ import annotations

import os
from sys import version_info
from typing import TYPE_CHECKING
from datetime import datetime
from datetime import timezone

import fs
import pytest

from vtsserving import Tag
from vtsserving._internal.vts import Bento
from vtsserving._internal.models import ModelStore
from vtsserving._internal.vts.vts import BentoInfo
from vtsserving._internal.vts.vts import BentoApiInfo
from vtsserving._internal.vts.vts import BentoModelInfo
from vtsserving._internal.vts.vts import BentoRunnerInfo
from vtsserving._internal.configuration import VTSSERVING_VERSION
from vtsserving._internal.vts.build_config import BentoBuildConfig

if TYPE_CHECKING:
    from pathlib import Path


def test_vts_info(tmpdir: Path):
    start = datetime.now(timezone.utc)
    vtsinfo_a = BentoInfo(tag=Tag("tag"), service="service")
    end = datetime.now(timezone.utc)

    assert vtsinfo_a.vtsserving_version == VTSSERVING_VERSION
    assert start <= vtsinfo_a.creation_time <= end
    # validate should fail

    tag = Tag("test", "version")
    service = "testservice"
    labels = {"label": "stringvalue"}
    model_creation_time = datetime.now(timezone.utc)
    model_a = BentoModelInfo(
        tag=Tag("model_a", "v1"),
        module="model_a_module",
        creation_time=model_creation_time,
    )
    model_b = BentoModelInfo(
        tag=Tag("model_b", "v3"),
        module="model_b_module",
        creation_time=model_creation_time,
    )
    models = [model_a, model_b]
    runner_a = BentoRunnerInfo(
        name="runner_a",
        runnable_type="test_runnable_a",
        models=["runner_a_model"],
        resource_config={"cpu": 2},
    )
    runners = [runner_a]
    api_predict = BentoApiInfo(
        name="predict",
        input_type="NumpyNdarray",
        output_type="NumpyNdarray",
    )
    apis = [api_predict]

    vtsinfo_b = BentoInfo(
        tag=tag,
        service=service,
        labels=labels,
        runners=runners,
        models=models,
        apis=apis,
    )

    vts_yaml_b_filename = os.path.join(tmpdir, "b_dump.yml")
    with open(vts_yaml_b_filename, "w", encoding="utf-8") as vts_yaml_b:
        vtsinfo_b.dump(vts_yaml_b)

    expected_yaml = """\
service: testservice
name: test
version: version
vtsserving_version: {vtsserving_version}
creation_time: '{creation_time}'
labels:
  label: stringvalue
models:
- tag: model_a:v1
  module: model_a_module
  creation_time: '{model_creation_time}'
- tag: model_b:v3
  module: model_b_module
  creation_time: '{model_creation_time}'
runners:
- name: runner_a
  runnable_type: test_runnable_a
  models:
  - runner_a_model
  resource_config:
    cpu: 2
apis:
- name: predict
  input_type: NumpyNdarray
  output_type: NumpyNdarray
docker:
  distro: debian
  python_version: '{python_version}'
  cuda_version: null
  env: null
  system_packages: null
  setup_script: null
  base_image: null
  dockerfile_template: null
python:
  requirements_txt: null
  packages: null
  lock_packages: true
  index_url: null
  no_index: null
  trusted_host: null
  find_links: null
  extra_index_url: null
  pip_args: null
  wheels: null
conda:
  environment_yml: null
  channels: null
  dependencies: null
  pip: null
"""

    with open(vts_yaml_b_filename, encoding="utf-8") as vts_yaml_b:
        assert vts_yaml_b.read() == expected_yaml.format(
            vtsserving_version=VTSSERVING_VERSION,
            creation_time=vtsinfo_b.creation_time.isoformat(),
            model_creation_time=model_creation_time.isoformat(),
            python_version=f"{version_info.major}.{version_info.minor}",
        )

    with open(vts_yaml_b_filename, encoding="utf-8") as vts_yaml_b:
        vtsinfo_b_from_yaml = BentoInfo.from_yaml_file(vts_yaml_b)

        assert vtsinfo_b_from_yaml == vtsinfo_b


def build_test_vts() -> Bento:
    vts_cfg = BentoBuildConfig(
        "simplevts.py:svc",
        include=["*.py", "config.json", "somefile", "*dir*", ".vtsignore"],
        exclude=["*.storage", "/somefile", "/subdir2"],
        conda={
            "environment_yml": "./environment.yaml",
        },
        docker={
            "setup_script": "./setup_docker_container.sh",
        },
        labels={
            "team": "foo",
            "dataset_version": "abc",
            "framework": "pytorch",
        },
    )

    return Bento.create(vts_cfg, version="1.0", build_ctx="./simplevts")


def fs_identical(fs1: fs.base.FS, fs2: fs.base.FS):
    for path in fs1.walk.dirs():
        assert fs2.isdir(path)

    for path in fs1.walk.files():
        assert fs2.isfile(path)
        assert fs1.readbytes(path) == fs2.readbytes(path)


@pytest.mark.usefixtures("change_test_dir")
def test_vts_export(tmpdir: "Path", model_store: "ModelStore"):
    working_dir = os.getcwd()

    testvts = build_test_vts()
    # Bento build will change working dir to the build_context, this will reset it
    os.chdir(working_dir)

    cfg = BentoBuildConfig("vtsa.py:svc")
    vtsa = Bento.create(cfg, build_ctx="./vtsa")
    # Bento build will change working dir to the build_context, this will reset it
    os.chdir(working_dir)

    vtsa1 = Bento.create(cfg, build_ctx="./vtsa1")
    # Bento build will change working dir to the build_context, this will reset it
    os.chdir(working_dir)

    cfg = BentoBuildConfig("vtsb.py:svc")
    vtsb = Bento.create(cfg, build_ctx="./vtsb")

    vts = testvts
    path = os.path.join(tmpdir, "testvts")
    export_path = vts.export(path)
    assert export_path == path + ".vts"
    assert os.path.isfile(export_path)
    imported_vts = Bento.import_from(export_path)
    assert imported_vts.tag == vts.tag
    assert imported_vts.info == vts.info
    del imported_vts

    vts = vtsa
    path = os.path.join(tmpdir, "vtsa")
    export_path = vts.export(path)
    assert export_path == path + ".vts"
    assert os.path.isfile(export_path)
    imported_vts = Bento.import_from(export_path)
    assert imported_vts.tag == vts.tag
    assert imported_vts.info == vts.info
    del imported_vts

    vts = vtsa1
    path = os.path.join(tmpdir, "vtsa1")
    export_path = vts.export(path)
    assert export_path == path + ".vts"
    assert os.path.isfile(export_path)
    imported_vts = Bento.import_from(export_path)
    assert imported_vts.tag == vts.tag
    assert imported_vts.info == vts.info
    del imported_vts

    vts = vtsb
    path = os.path.join(tmpdir, "vtsb")
    export_path = vts.export(path)
    assert export_path == path + ".vts"
    assert os.path.isfile(export_path)
    imported_vts = Bento.import_from(export_path)
    assert imported_vts.tag == vts.tag
    assert imported_vts.info == vts.info
    del imported_vts

    vts = testvts
    path = os.path.join(tmpdir, "testvts.vts")
    export_path = vts.export(path)
    assert export_path == path
    assert os.path.isfile(export_path)
    imported_vts = Bento.import_from(path)
    assert imported_vts.tag == vts.tag
    assert imported_vts.info == vts.info
    del imported_vts

    path = os.path.join(tmpdir, "testvts-parent")
    os.mkdir(path)
    export_path = vts.export(path)
    assert export_path == os.path.join(path, vts._export_name + ".vts")
    assert os.path.isfile(export_path)
    imported_vts = Bento.import_from(export_path)
    assert imported_vts.tag == vts.tag
    assert imported_vts.info == vts.info
    del imported_vts

    path = os.path.join(tmpdir, "testvts-parent-2/")
    with pytest.raises(ValueError):
        export_path = vts.export(path)

    path = os.path.join(tmpdir, "vts-dir")
    os.mkdir(path)
    export_path = vts.export(path)
    assert export_path == os.path.join(path, vts._export_name + ".vts")
    assert os.path.isfile(export_path)
    imported_vts = Bento.import_from(export_path)
    assert imported_vts.tag == vts.tag
    assert imported_vts.info == vts.info
    del imported_vts

    path = "temp://pytest-some-temp"
    export_path = vts.export(path)
    assert export_path.endswith(
        os.path.join("pytest-some-temp", vts._export_name + ".vts")
    )
    # because this is a tempdir, it's cleaned up immediately after creation...

    path = "osfs://" + fs.path.join(str(tmpdir), "testvts-by-url")
    export_path = vts.export(path)
    assert export_path == os.path.join(tmpdir, "testvts-by-url.vts")
    assert os.path.isfile(export_path)
    imported_vts = Bento.import_from(export_path)
    assert imported_vts.tag == vts.tag
    assert imported_vts.info == vts.info
    del imported_vts
    imported_vts = Bento.import_from(path + ".vts")
    assert imported_vts.tag == vts.tag
    assert imported_vts.info == vts.info
    del imported_vts

    path = "osfs://" + fs.path.join(str(tmpdir), "testvts-by-url")
    with pytest.raises(ValueError):
        vts.export(path, subpath="/badsubpath")

    path = "zip://" + fs.path.join(str(tmpdir), "testvts.zip")
    export_path = vts.export(path)
    assert export_path == path
    imported_vts = Bento.import_from(export_path)
    assert imported_vts.tag == vts.tag
    assert imported_vts.info == vts.info
    del imported_vts

    path = os.path.join(tmpdir, "testvts-gz")
    os.mkdir(path)
    export_path = vts.export(path, output_format="gz")
    assert export_path == os.path.join(path, vts._export_name + ".gz")
    assert os.path.isfile(export_path)
    imported_vts = Bento.import_from(export_path)
    assert imported_vts.tag == vts.tag
    assert imported_vts.info == vts.info
    del imported_vts

    path = os.path.join(tmpdir, "testvts-gz-1/")
    with pytest.raises(ValueError):
        vts.export(path, output_format="gz")


@pytest.mark.usefixtures("change_test_dir")
def test_vts(model_store: ModelStore):
    start = datetime.now(timezone.utc)
    vts = build_test_vts()
    end = datetime.now(timezone.utc)

    assert vts.info.vtsserving_version == VTSSERVING_VERSION
    assert start <= vts.creation_time <= end
    # validate should fail

    with vts._fs as vts_fs:  # type: ignore
        assert set(vts_fs.listdir("/")) == {
            "vts.yaml",
            "apis",
            "models",
            "README.md",
            "src",
            "env",
        }
        assert set(vts_fs.listdir("src")) == {
            "simplevts.py",
            "subdir",
            ".vtsignore",
        }
        assert set(vts_fs.listdir("src/subdir")) == {"somefile"}
