from __future__ import annotations

import typing as t
import logging
from types import ModuleType
from typing import TYPE_CHECKING

import cloudpickle

import vtsserving
from vtsserving import Tag
from vtsserving.models import Model
from vtsserving.models import ModelContext
from vtsserving.exceptions import NotFound

from ..models import PKL_EXT
from ..models import SAVE_NAMESPACE
from ..models.model import ModelSignature

if TYPE_CHECKING:
    from .. import external_typing as ext

    ModelType = t.Any

MODULE_NAME = "vtsserving.picklable_model"
API_VERSION = "v1"


logger = logging.getLogger(__name__)


def get(tag_like: str | Tag) -> Model:
    model = vtsserving.models.get(tag_like)
    if model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {model.tag} was saved with module {model.info.module}, not loading with {MODULE_NAME}."
        )
    return model


def load_model(vts_model: str | Tag | Model) -> ModelType:
    """
    Load the picklable model with the given tag from the local VtsServing model store.

    Args:
        vts_model (``str`` ``|`` :obj:`~vtsserving.Tag` ``|`` :obj:`~vtsserving.Model`):
            Either the tag of the model to get from the store, or a VtsServing `~vtsserving.Model`
            instance to load the model from.
        ...
    Returns:
        ``object``
            The picklable model loaded from the model store or VtsServing :obj:`~vtsserving.Model`.
    Example:
    .. code-block:: python
        import vtsserving

        picklable_model = vtsserving.picklable_model.load_model('my_model:latest')
    """  # noqa
    if not isinstance(vts_model, Model):
        vts_model = get(vts_model)

    if vts_model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {vts_model.tag} was saved with module {vts_model.info.module}, not loading with {MODULE_NAME}."
        )

    model_file = vts_model.path_of(f"{SAVE_NAMESPACE}{PKL_EXT}")

    with open(model_file, "rb") as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        return cloudpickle.load(f)


def save_model(
    name: str,
    model: ModelType,
    *,
    signatures: dict[str, ModelSignature] | None = None,
    labels: t.Dict[str, str] | None = None,
    custom_objects: t.Dict[str, t.Any] | None = None,
    external_modules: t.List[ModuleType] | None = None,
    metadata: t.Dict[str, t.Any] | None = None,
) -> vtsserving.Model:
    """
    Save a model instance to VtsServing modelstore.

    Args:
        name (:code:`str`):
            Name for given model instance. This should pass Python identifier check.
        model (:code:`Union[BaseEstimator, Pipeline]`):
            Instance of model to be saved.
        signatures (:code: `Dict[str, ModelSignatureDict]`)
            Methods to expose for running inference on the target model. Signatures are
             used for creating Runner instances when serving model with vtsserving.Service
        labels (:code:`Dict[str, str]`, `optional`, default to :code:`None`):
            user-defined labels for managing models, e.g. team=nlp, stage=dev
        custom_objects (:code:`Dict[str, Any]]`, `optional`, default to :code:`None`):
            user-defined additional python objects to be saved alongside the model,
             e.g. a tokenizer instance, preprocessor function, model configuration json
        external_modules (:code:`List[ModuleType]`, `optional`, default to :code:`None`):
            user-defined additional python modules to be saved alongside the model or custom objects,
            e.g. a tokenizer module, preprocessor module, model configuration module
        metadata (:code:`Dict[str, Any]`, `optional`,  default to :code:`None`):
            Custom metadata for given model.

    Returns:
        :obj:`~vtsserving.Tag`: A :obj:`tag` with a format `name:version` where `name` is
        the user-defined model's name, and a generated `version`.

    Examples:

    .. code-block:: python

        import vtsserving

        vts_model = vtsserving.picklable.save_model('picklable_pyobj', model)
    """  # noqa
    context = ModelContext(
        framework_name="cloudpickle",
        framework_versions={"cloudpickle": cloudpickle.__version__},
    )

    if signatures is None:
        signatures = {"__call__": ModelSignature(batchable=False)}
        logger.info(
            'Using the default model signature for pickable model (%s) for model "%s".',
            signatures,
            name,
        )

    with vtsserving.models.create(
        name,
        module=MODULE_NAME,
        api_version=API_VERSION,
        labels=labels,
        custom_objects=custom_objects,
        external_modules=external_modules,
        metadata=metadata,
        context=context,
        signatures=signatures,
    ) as vts_model:

        with open(vts_model.path_of(f"{SAVE_NAMESPACE}{PKL_EXT}"), "wb") as f:
            cloudpickle.dump(model, f)

        return vts_model


def get_runnable(vts_model: Model):
    """
    Private API: use :obj:`~vtsserving.Model.to_runnable` instead.
    """

    class PicklableRunnable(vtsserving.Runnable):
        SUPPORTED_RESOURCES = ("cpu",)
        SUPPORTS_CPU_MULTI_THREADING = False

        def __init__(self):
            super().__init__()
            self.model = load_model(vts_model)

    def _get_run(method_name: str):
        def _run(
            self: PicklableRunnable,
            *args: ext.NpNDArray | ext.PdDataFrame,
            **kwargs: ext.NpNDArray | ext.PdDataFrame,
        ) -> ext.NpNDArray:
            assert isinstance(method_name, str), repr(method_name)
            return getattr(self.model, method_name)(*args, **kwargs)

        return _run

    for method_name, options in vts_model.info.signatures.items():
        assert isinstance(method_name, str), repr(method_name)
        PicklableRunnable.add_method(
            _get_run(method_name),
            name=method_name,
            batchable=options.batchable,
            batch_dim=options.batch_dim,
            input_spec=options.input_spec,
            output_spec=options.output_spec,
        )

    return PicklableRunnable
