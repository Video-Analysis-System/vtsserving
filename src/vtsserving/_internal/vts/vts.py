from __future__ import annotations

import os
import typing as t
import logging
from typing import TYPE_CHECKING
from datetime import datetime
from datetime import timezone

import fs
import attr
import yaml
import fs.osfs
import fs.errors
import fs.mirror
from fs.copy import copy_file
from cattr.gen import override
from cattr.gen import make_dict_structure_fn
from cattr.gen import make_dict_unstructure_fn
from simple_di import inject
from simple_di import Provide

from ..tag import Tag
from ..store import Store
from ..store import StoreItem
from ..types import PathType
from ..utils import vtsserving_cattr
from ..utils import copy_file_to_fs_folder
from ..models import ModelStore
from ..runner import Runner
from ...exceptions import InvalidArgument
from ...exceptions import VtsServingException
from .build_config import CondaOptions
from .build_config import BentoPathSpec
from .build_config import DockerOptions
from .build_config import PythonOptions
from .build_config import BentoBuildConfig
from ..configuration import VTSSERVING_VERSION
from ..configuration.containers import VtsServingContainer

if TYPE_CHECKING:
    from fs.base import FS

    from ..models import Model
    from ..service import Service
    from ..service.inference_api import InferenceAPI

logger = logging.getLogger(__name__)

VTS_YAML_FILENAME = "vts.yaml"
VTS_PROJECT_DIR_NAME = "src"
VTS_README_FILENAME = "README.md"
DEFAULT_VTS_BUILD_FILE = "vtsfile.yaml"

API_INFO_MD = "| POST [`/{api}`](#{link}) | {input} | {output} |"

INFERENCE_TABLE_MD = """\
| InferenceAPI | Input | Output |
| ------------ | ----- | ------ |
{content}
"""


def create_inference_api_table(svc: Service) -> str:
    from ..service.openapi import APP_TAG

    contents = [
        API_INFO_MD.format(
            api=api.name,
            link=f"operations-{APP_TAG.name.replace(' ','_')}-{svc.name}__{api.name}",  # follows operationId from OpenAPI
            input=api.input.__class__.__name__,
            output=api.output.__class__.__name__,
        )
        for api in svc.apis.values()
    ]
    return INFERENCE_TABLE_MD.format(content="\n".join(contents))


def get_default_svc_readme(svc: Service, svc_version: str | None = None) -> str:
    if svc.vts:
        vtsserving_version = svc.vts.info.vtsserving_version
    else:
        vtsserving_version = VTSSERVING_VERSION

    if not svc_version:
        if svc.tag and svc.tag.version:
            svc_version = svc.tag.version
        else:
            svc_version = "None"

    doc = f"""\
# {svc.name}:{svc_version}

[![pypi_status](https://img.shields.io/badge/VtsServing-{vtsserving_version}-informational)](https://pypi.org/project/VtsServing)
[![documentation_status](https://readthedocs.org/projects/vtsserving/badge/?version=latest)](https://docs.vtsserving.org/)
[![join_slack](https://badgen.net/badge/Join/VtsServing%20Slack/cyan?icon=slack)](https://l.vtsserving.com/join-slack-swagger)
[![VtsServing GitHub Repo](https://img.shields.io/github/stars/vtsserving/vtsserving?style=social)](https://github.com/vtsserving/VtsServing)
[![Twitter Follow](https://img.shields.io/twitter/follow/vtsservingai?label=Follow%20VtsServing&style=social)](https://twitter.com/vtsservingai)

This is a Machine Learning Service created with VtsServing."""

    if svc.apis:
        doc += f"\n{create_inference_api_table(svc)}\n\n"

    doc += """

## Help

* [📖 Documentation](https://docs.vtsserving.org/en/latest/): Learn how to use VtsServing.
* [💬 Community](https://l.vtsserving.com/join-slack-swagger): Join the VtsServing Slack community.
* [🐛 GitHub Issues](https://github.com/vtsserving/VtsServing/issues): Report bugs and feature requests.
* Tip: you can also [customize this README](https://docs.vtsserving.org/en/latest/concepts/vts.html#description).
"""
    # TODO: add links to documentation that may help with API client development
    return doc


@attr.define(repr=False, auto_attribs=False)
class Bento(StoreItem):
    _tag: Tag = attr.field()
    __fs: FS = attr.field()

    _info: BentoInfo

    _model_store: ModelStore
    _doc: t.Optional[str] = None

    @staticmethod
    def _export_ext() -> str:
        return "vts"

    @__fs.validator  # type:ignore # attrs validators not supported by pyright
    def check_fs(self, _attr: t.Any, new_fs: FS):
        try:
            new_fs.makedir("models", recreate=True)
        except fs.errors.ResourceReadOnly:
            # when we import a tarfile, it will be read-only, so just skip the step where we create
            # the models folder.
            pass
        self._model_store = ModelStore(new_fs.opendir("models"))

    def __init__(self, tag: Tag, vts_fs: "FS", info: "BentoInfo"):
        self._tag = tag
        self.__fs = vts_fs
        self.check_fs(None, vts_fs)
        self._info = info

    @property
    def tag(self) -> Tag:
        return self._tag

    @property
    def _fs(self) -> FS:
        return self.__fs

    @property
    def info(self) -> BentoInfo:
        return self._info

    @classmethod
    @inject
    def create(
        cls,
        build_config: BentoBuildConfig,
        version: t.Optional[str] = None,
        build_ctx: t.Optional[str] = None,
    ) -> Bento:
        from ..service.loader import import_service

        build_ctx = (
            os.getcwd()
            if build_ctx is None
            else os.path.realpath(os.path.expanduser(build_ctx))
        )
        if not os.path.isdir(build_ctx):
            raise InvalidArgument(
                f"Bento build context {build_ctx} does not exist or is not a directory."
            )

        # This also verifies that svc can be imported correctly
        svc = import_service(
            build_config.service, working_dir=build_ctx, standalone_load=True
        )

        tag = Tag(svc.name, version)
        if version is None:
            tag = tag.make_new_version()

        logger.info(
            'Building VtsServing service "%s" from build context "%s".', tag, build_ctx
        )

        vts_fs = fs.open_fs(f"temp://vtsserving_vts_{svc.name}")
        ctx_fs = fs.open_fs(build_ctx)

        models: t.Set[Model] = set()
        # Add all models required by the service
        for model in svc.models:
            models.add(model)
        # Add all models required by service runners
        for runner in svc.runners:
            for model in runner.models:
                models.add(model)

        vts_fs.makedir("models", recreate=True)
        vts_model_store = ModelStore(vts_fs.opendir("models"))
        for model in models:
            logger.info('Packing model "%s"', model.tag)
            model._save(vts_model_store)

        # Apply default build options
        build_config = build_config.with_defaults()
        # create ignore specs
        specs = BentoPathSpec(build_config.include, build_config.exclude)  # type: ignore (unfinished attrs converter type)

        # Copy all files base on include and exclude, into `src` directory
        relpaths = [s for s in build_config.include if s.startswith("../")]
        if len(relpaths) != 0:
            raise InvalidArgument(
                "Paths outside of the build context directory cannot be included; use a symlink or copy those files into the working directory manually."
            )
        vts_fs.makedir(VTS_PROJECT_DIR_NAME)
        target_fs = vts_fs.opendir(VTS_PROJECT_DIR_NAME)

        for dir_path, _, files in ctx_fs.walk():
            for f in files:
                path = fs.path.combine(dir_path, f.name).lstrip("/")
                if specs.includes(
                    path,
                    recurse_exclude_spec=filter(
                        lambda s: fs.path.isparent(s[0], dir_path),
                        specs.from_path(build_ctx),
                    ),
                ):
                    target_fs.makedirs(dir_path, recreate=True)
                    copy_file(ctx_fs, path, target_fs, path)

        # NOTE: we need to generate both Python and Conda
        # first to make sure we can generate the Dockerfile correctly.
        build_config.python.write_to_vts(vts_fs, build_ctx)
        build_config.conda.write_to_vts(vts_fs, build_ctx)
        build_config.docker.write_to_vts(vts_fs, build_ctx, build_config.conda)

        # Create `readme.md` file
        if build_config.description is None:
            with vts_fs.open(VTS_README_FILENAME, "w", encoding="utf-8") as f:
                f.write(get_default_svc_readme(svc, svc_version=tag.version))
        else:
            if build_config.description.startswith("file:"):
                file_name = build_config.description[5:].strip()
                copy_file_to_fs_folder(
                    file_name, vts_fs, dst_filename=VTS_README_FILENAME
                )
            else:
                with vts_fs.open(VTS_README_FILENAME, "w") as f:
                    f.write(build_config.description)

        # Create 'apis/openapi.yaml' file
        vts_fs.makedir("apis")
        with vts_fs.open(fs.path.combine("apis", "openapi.yaml"), "w") as f:
            yaml.dump(svc.openapi_spec, f)

        res = Bento(
            tag,
            vts_fs,
            BentoInfo(
                tag=tag,
                service=svc,  # type: ignore # attrs converters do not typecheck
                labels=build_config.labels,
                models=[BentoModelInfo.from_vts_model(m) for m in models],
                runners=[BentoRunnerInfo.from_runner(r) for r in svc.runners],
                apis=[
                    BentoApiInfo.from_inference_api(api) for api in svc.apis.values()
                ],
                docker=build_config.docker,
                python=build_config.python,
                conda=build_config.conda,
            ),
        )
        # Create vts.yaml
        res.flush_info()
        try:
            res.validate()
        except VtsServingException as e:
            raise VtsServingException(f"Failed to create {res!s}: {e}") from None

        return res

    @classmethod
    def from_fs(cls, item_fs: FS) -> Bento:
        try:
            with item_fs.open(VTS_YAML_FILENAME, "r", encoding="utf-8") as vts_yaml:
                info = BentoInfo.from_yaml_file(vts_yaml)
        except fs.errors.ResourceNotFound:
            raise VtsServingException(
                f"Failed to load vts because it does not contain a '{VTS_YAML_FILENAME}'"
            )

        res = cls(info.tag, item_fs, info)
        try:
            res.validate()
        except VtsServingException as e:
            raise VtsServingException(f"Failed to load vts: {e}") from None

        return res

    @property
    def path(self) -> str:
        return self.path_of("/")

    def path_of(self, item: str) -> str:
        return self._fs.getsyspath(item)

    def flush_info(self):
        with self._fs.open(VTS_YAML_FILENAME, "w") as vts_yaml:
            self.info.dump(vts_yaml)

    @property
    def doc(self) -> str:
        if self._doc is not None:
            return self._doc

        with self._fs.open(VTS_README_FILENAME, "r") as readme_md:
            self._doc = str(readme_md.read())
            return self._doc

    @property
    def creation_time(self) -> datetime:
        return self.info.creation_time

    @inject
    def save(
        self,
        vts_store: "BentoStore" = Provide[VtsServingContainer.vts_store],
    ) -> "Bento":
        try:
            self.validate()
        except VtsServingException as e:
            raise VtsServingException(f"Failed to save {self!s}: {e}") from None

        with vts_store.register(self.tag) as vts_path:
            out_fs = fs.open_fs(vts_path, create=True, writeable=True)
            fs.mirror.mirror(self._fs, out_fs, copy_if_newer=False)
            self._fs.close()
            self.__fs = out_fs

        return self

    def validate(self):
        if not self._fs.isfile(VTS_YAML_FILENAME):
            raise VtsServingException(
                f"{self!s} does not contain a {VTS_YAML_FILENAME}."
            )

    def __str__(self):
        return f'Bento(tag="{self.tag}")'


class BentoStore(Store[Bento]):
    def __init__(self, base_path: t.Union[PathType, "FS"]):
        super().__init__(base_path, Bento)


@attr.frozen
class BentoRunnerInfo:
    name: str
    runnable_type: str
    models: t.List[str] = attr.field(factory=list)
    resource_config: t.Optional[t.Dict[str, t.Any]] = attr.field(default=None)

    @classmethod
    def from_runner(cls, r: Runner) -> BentoRunnerInfo:
        return cls(
            name=r.name,
            runnable_type=r.runnable_class.__name__,
            models=[str(model.tag) for model in r.models],
            resource_config=r.resource_config,
        )


@attr.frozen
class BentoApiInfo:
    name: str
    input_type: str
    output_type: str

    @classmethod
    def from_inference_api(cls, api: InferenceAPI) -> BentoApiInfo:
        return cls(
            name=api.name,
            input_type=api.input.__class__.__name__,
            output_type=api.output.__class__.__name__,
        )


@attr.frozen
class BentoModelInfo:
    tag: Tag = attr.field(converter=Tag.from_taglike)
    module: str
    creation_time: datetime

    @classmethod
    def from_vts_model(cls, vts_model: Model) -> BentoModelInfo:
        return cls(
            tag=vts_model.tag,
            module=vts_model.info.module,
            creation_time=vts_model.info.creation_time,
        )


@attr.frozen(repr=False)
class BentoInfo:

    # for backward compatibility in case new fields are added to BentoInfo.
    __forbid_extra_keys__ = False
    # omit field in yaml file if it is not provided by the user.
    __omit_if_default__ = True

    tag: Tag
    service: str = attr.field(
        converter=lambda svc: svc if isinstance(svc, str) else svc._import_str
    )
    name: str = attr.field(init=False)
    version: str = attr.field(init=False)
    # using factory explicitly instead of default because omit_if_default is enabled for BentoInfo
    vtsserving_version: str = attr.field(factory=lambda: VTSSERVING_VERSION)
    creation_time: datetime = attr.field(factory=lambda: datetime.now(timezone.utc))

    labels: t.Dict[str, t.Any] = attr.field(factory=dict)
    models: t.List[BentoModelInfo] = attr.field(factory=list)
    runners: t.List[BentoRunnerInfo] = attr.field(factory=list)
    apis: t.List[BentoApiInfo] = attr.field(factory=list)
    docker: DockerOptions = attr.field(factory=lambda: DockerOptions().with_defaults())
    python: PythonOptions = attr.field(factory=lambda: PythonOptions().with_defaults())
    conda: CondaOptions = attr.field(factory=lambda: CondaOptions().with_defaults())

    def __attrs_post_init__(self):
        # Direct set is not available when frozen=True
        object.__setattr__(self, "name", self.tag.name)
        object.__setattr__(self, "version", self.tag.version)

        try:
            self.validate()
        except VtsServingException as e:
            raise VtsServingException(f"Failed to initialize {self!s}: {e}") from None

    def to_dict(self) -> t.Dict[str, t.Any]:
        return vtsserving_cattr.unstructure(self)

    def dump(self, stream: t.IO[t.Any]):
        return yaml.dump(self, stream, sort_keys=False)

    @classmethod
    def from_yaml_file(cls, stream: t.IO[t.Any]) -> BentoInfo:
        try:
            yaml_content = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.error("Error while parsing YAML file: %s", exc)
            raise

        assert yaml_content is not None

        yaml_content["tag"] = Tag(yaml_content["name"], yaml_content["version"])
        del yaml_content["name"]
        del yaml_content["version"]

        # For backwards compatibility for vtss created prior to version 1.0.0rc1
        if "runners" in yaml_content:
            runners = yaml_content["runners"]
            for r in runners:
                if "runner_type" in r:  # BentoRunnerInfo prior to 1.0.0rc1 release
                    r["runnable_type"] = r["runner_type"]
                    del r["runner_type"]
                    if "model_runner_module" in r:
                        del r["model_runner_module"]

        if "models" in yaml_content:
            # For backwards compatibility for vtss created prior to version 1.0.0a7
            models = yaml_content["models"]
            if models and len(models) > 0 and isinstance(models[0], str):
                yaml_content["models"] = list(
                    map(
                        lambda model_tag: {
                            "tag": model_tag,
                            "module": "unknown",
                            "creation_time": datetime.fromordinal(1),
                        },
                        models,
                    )
                )
        try:
            return vtsserving_cattr.structure(yaml_content, cls)
        except KeyError as e:
            raise VtsServingException(f"Missing field {e} in {VTS_YAML_FILENAME}")

    def validate(self):
        # Validate vts.yml file schema, content, vtsserving version, etc
        ...


vtsserving_cattr.register_structure_hook_func(
    lambda cls: issubclass(cls, BentoInfo),
    make_dict_structure_fn(
        BentoInfo,
        vtsserving_cattr,
        name=override(omit=True),
        version=override(omit=True),
    ),
)
vtsserving_cattr.register_unstructure_hook(
    BentoInfo,
    # Ignore tag, tag is saved via the name and version field
    make_dict_unstructure_fn(BentoInfo, vtsserving_cattr, tag=override(omit=True)),
)


def _BentoInfo_dumper(dumper: yaml.Dumper, info: BentoInfo) -> yaml.Node:
    return dumper.represent_dict(info.to_dict())


yaml.add_representer(BentoInfo, _BentoInfo_dumper)  # type: ignore (incomplete yaml types)
