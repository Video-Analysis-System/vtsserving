from __future__ import annotations

import typing as t
import logging
from types import ModuleType
from typing import TYPE_CHECKING

import vtsserving
from vtsserving import Tag
from vtsserving.models import Model
from vtsserving.models import ModelContext
from vtsserving.exceptions import NotFound
from vtsserving.exceptions import MissingDependencyException

from ..types import LazyType
from ..utils.pkg import get_pkg_version

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator
    from sklearn.pipeline import Pipeline

    from vtsserving.types import ModelSignature

    from .. import external_typing as ext
    from ..models.model import ModelSignaturesType

    SklearnModel: t.TypeAlias = BaseEstimator | Pipeline


try:
    import joblib
    from joblib import parallel_backend
except ImportError:  # pragma: no cover
    try:
        from sklearn.utils._joblib import joblib
        from sklearn.utils._joblib import parallel_backend
    except ImportError:
        raise MissingDependencyException(
            "scikit-learn is required in order to use the module 'vtsserving.sklearn', install sklearn with 'pip install sklearn'. For more information, refer to https://scikit-learn.org/stable/install.html"
        )

MODULE_NAME = "vtsserving.sklearn"
MODEL_FILENAME = "saved_model.pkl"
API_VERSION = "v1"

logger = logging.getLogger(__name__)


def get(tag_like: str | Tag) -> Model:
    model = vtsserving.models.get(tag_like)
    if model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {model.tag} was saved with module {model.info.module}, not loading with {MODULE_NAME}."
        )
    return model


def load_model(
    vts_model: str | Tag | Model,
) -> SklearnModel:
    """
    Load the scikit-learn model with the given tag from the local VtsServing model store.

    Args:
        vts_model (``str`` ``|`` :obj:`~vtsserving.Tag` ``|`` :obj:`~vtsserving.Model`):
            Either the tag of the model to get from the store, or a VtsServing `~vtsserving.Model`
            instance to load the model from.
        ...
    Returns:
        ``BaseEstimator`` ``|`` ``Pipeline``:
            The scikit-learn model loaded from the model store or VtsServing :obj:`~vtsserving.Model`.
    Example:
    .. code-block:: python
        import vtsserving
        sklearn = vtsserving.sklearn.load_model('my_model:latest')
    """  # noqa
    if not isinstance(vts_model, Model):
        vts_model = get(vts_model)

    if vts_model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {vts_model.tag} was saved with module {vts_model.info.module}, not loading with {MODULE_NAME}."
        )
    model_file = vts_model.path_of(MODEL_FILENAME)

    return joblib.load(model_file)


def save_model(
    name: str,
    model: SklearnModel,
    *,
    signatures: ModelSignaturesType | None = None,
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

        from sklearn.datasets import load_iris
        from sklearn.neighbors import KNeighborsClassifier

        model = KNeighborsClassifier()
        iris = load_iris()
        X = iris.data[:, :4]
        Y = iris.target
        model.fit(X, Y)

        vts_model = vtsserving.sklearn.save_model('kneighbors', model)
    """  # noqa
    if not (
        LazyType("sklearn.base.BaseEstimator").isinstance(model)
        or LazyType("sklearn.pipeline.Pipeline").isinstance(model)
    ):
        raise TypeError(
            f"Given model ({model}) is not a sklearn.base.BaseEstimator or sklearn.pipeline.Pipeline."
        )

    context = ModelContext(
        framework_name="sklearn",
        framework_versions={"scikit-learn": get_pkg_version("scikit-learn")},
    )

    if signatures is None:
        signatures = {"predict": {"batchable": False}}
        logger.info(
            'Using the default model signature for scikit-learn (%s) for model "%s".',
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

        joblib.dump(model, vts_model.path_of(MODEL_FILENAME))

        return vts_model


def get_runnable(vts_model: Model):
    """
    Private API: use :obj:`~vtsserving.Model.to_runnable` instead.
    """

    class SklearnRunnable(vtsserving.Runnable):
        SUPPORTED_RESOURCES = ("cpu",)
        SUPPORTS_CPU_MULTI_THREADING = True

        def __init__(self):
            super().__init__()
            self.model = load_model(vts_model)

    def add_runnable_method(method_name: str, options: ModelSignature):
        def _run(
            self: SklearnRunnable, input_data: ext.NpNDArray | ext.PdDataFrame
        ) -> ext.NpNDArray:
            # TODO: set inner_max_num_threads and n_jobs param here base on strategy env vars
            with parallel_backend(backend="loky"):
                return getattr(self.model, method_name)(input_data)

        SklearnRunnable.add_method(
            _run,
            name=method_name,
            batchable=options.batchable,
            batch_dim=options.batch_dim,
            input_spec=options.input_spec,
            output_spec=options.output_spec,
        )

    for method_name, options in vts_model.info.signatures.items():
        add_runnable_method(method_name, options)

    return SklearnRunnable
