from __future__ import annotations

import typing as t
import logging
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np

import vtsserving
from vtsserving import Tag
from vtsserving.exceptions import NotFound
from vtsserving.exceptions import InvalidArgument
from vtsserving.exceptions import MissingDependencyException

from ..utils.pkg import get_pkg_version
from ..models.model import ModelContext

if TYPE_CHECKING:
    from vtsserving.types import ModelSignature
    from vtsserving.types import ModelSignatureDict

    from .. import external_typing as ext

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        "lightgbm is required in order to use module 'vtsserving.lightgbm', install lightgbm with 'pip install lightgbm'. For more information, refer to https://github.com/microsoft/LightGBM/tree/master/python-package"
    )

MODULE_NAME = "vtsserving.lightgbm"
MODEL_FILENAME = "saved_model.ubj"
API_VERSION = "v1"

logger = logging.getLogger(__name__)


def get(tag_like: str | Tag) -> vtsserving.Model:
    """
    Get the VtsServing model with the given tag.

    Args:
        tag_like (``str`` ``|`` :obj:`~vtsserving.Tag`):
            The tag of the model to retrieve from the model store.
    Returns:
        :obj:`~vtsserving.Model`: A VtsServing :obj:`~vtsserving.Model` with the matching tag.
    Example:

    .. code-block:: python

        import vtsserving
        # target model must be from the VtsServing model store
        model = vtsserving.lightgbm.get("my_lightgbm_model:latest")
    """
    model = vtsserving.models.get(tag_like)
    if model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {model.tag} was saved with module {model.info.module}, not loading with {MODULE_NAME}."
        )
    return model


def load_model(vts_model: str | Tag | vtsserving.Model) -> lgb.basic.Booster:  # type: ignore (incomplete ligthgbm type stubs)
    """
    Load the LightGBM model with the given tag from the local VtsServing model store.

    Args:
        vts_model (``str`` ``|`` :obj:`~vtsserving.Tag` ``|`` :obj:`~vtsserving.Model`):
            Either the tag of the model to get from the store, or a VtsServing `~vtsserving.Model`
            instance to load the model from.
    Returns:
        :obj:`~lightgbm.basic.Booster`: The LightGBM model loaded from the model store or VtsServing :obj:`~vtsserving.Model`.

    Example:

    .. code-block:: python

        import vtsserving
        gbm = vtsserving.lightgbm.load("my_lightgbm_model:latest")
    """  # noqa
    if not isinstance(vts_model, vtsserving.Model):
        vts_model = get(vts_model)
        assert isinstance(vts_model, vtsserving.Model)

    if vts_model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {vts_model.tag} was saved with module {vts_model.info.module}, not loading with {MODULE_NAME}."
        )

    model_file = vts_model.path_of(MODEL_FILENAME)
    booster = lgb.basic.Booster(model_file=model_file)  # type: ignore (incomplete ligthgbm type stubs)
    return booster  # type: ignore


def save_model(
    name: str,
    model: lgb.basic.Booster,  # type: ignore (incomplete ligthgbm type stubs)
    *,
    signatures: dict[str, ModelSignatureDict] | None = None,
    labels: dict[str, str] | None = None,
    custom_objects: dict[str, t.Any] | None = None,
    external_modules: t.List[ModuleType] | None = None,
    metadata: dict[str, t.Any] | None = None,
) -> vtsserving.Model:
    """
    Save a LightGBM model instance to the VtsServing model store.

    Args:
        name (``str``):
            The name to give to the model in the VtsServing store. This must be a valid
            :obj:`~vtsserving.Tag` name.
        model (:obj:`~lgb.basic.Booster`):
            The LightGBM model (booster) to be saved.
        signatures (``dict[str, ModelSignatureDict]``, optional):
            Signatures of predict methods to be used. If not provided, the signatures default to
            ``{"predict": {"batchable": False}}``. See :obj:`~vtsserving.types.ModelSignature` for more
            details.
        labels (``dict[str, str]``, optional):
            A default set of management labels to be associated with the model. An example is
            ``{"training-set": "data-1"}``.
        custom_objects (``dict[str, Any]``, optional):
            Custom objects to be saved with the model. An example is
            ``{"my-normalizer": normalizer}``.

            Custom objects are currently serialized with cloudpickle, but this implementation is
            subject to change.
        external_modules (:code:`List[ModuleType]`, `optional`, default to :code:`None`):
            user-defined additional python modules to be saved alongside the model or custom objects,
            e.g. a tokenizer module, preprocessor module, model configuration module
        metadata (``dict[str, Any]``, optional):
            Metadata to be associated with the model. An example is ``{"max_depth": 2}``.

            Metadata is intended for display in model management UI and therefore must be a default
            Python type, such as ``str`` or ``int``.
    Returns:
        :obj:`~vtsserving.Tag`: A :obj:`tag` with a format `name:version` where `name` is the
        user-defined model's name, and a generated `version` by VtsServing.

    Example:

    .. code-block:: python

        import vtsserving

        import lightgbm as lgb
        import pandas as pd

        # load a dataset
        df_train = pd.read_csv("regression.train", header=None, sep="\t")
        df_test = pd.read_csv("regression.test", header=None, sep="\t")

        y_train = df_train[0]
        y_test = df_test[0]
        X_train = df_train.drop(0, axis=1)
        X_test = df_test.drop(0, axis=1)

        # create dataset for lightgbm
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

        # specify your configurations as a dict
        params = {
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": {"l2", "l1"},
            "num_leaves": 31,
            "learning_rate": 0.05,
        }

        # train
        gbm = lgb.train(
            params, lgb_train, num_boost_round=20, valid_sets=lgb_eval
        )

        # save the booster to VtsServing modelstore:
        vts_model = vtsserving.lightgbm.save_model("my_lightgbm_model", gbm, booster_params=params)
    """

    # Ensure that `model` is actually the Booster object, and not for example one of the scikit-learn wrapper objects.
    if not isinstance(model, lgb.basic.Booster):  # type: ignore (incomplete ligthgbm type stubs)
        try:
            # Work around a LightGBM issue (https://github.com/microsoft/LightGBM/issues/3014)
            # 'model.booster_' chjecks that the model has been fitted and will error otherwise.
            if not hasattr(model, "fitted_"):  # type: ignore (incomplete ligthgbm type stubs)
                model.fitted_ = True

            model = model.booster_  # type: ignore (incomplete ligthgbm type stubs)
        except AttributeError as e:
            logger.error('Unable to obtain a "lightgbm.basic.Booster" from %s.', model)
            raise e

    if not isinstance(model, lgb.basic.Booster):  # type: ignore (incomplete ligthgbm type stubs)
        raise TypeError(f"Given model ({model}) is not a lightgbm.basic.Booster.")

    context: ModelContext = ModelContext(
        framework_name="lightgbm",
        framework_versions={"lightgbm": get_pkg_version("lightgbm")},
    )

    if signatures is None:
        signatures = {
            "predict": {"batchable": False},
        }
        logger.info(
            'Using the default model signature for LightGBM (%s) for model "%s".',
            signatures,
            name,
        )

    with vtsserving.models.create(
        name,
        module=MODULE_NAME,
        api_version=API_VERSION,
        signatures=signatures,
        labels=labels,
        custom_objects=custom_objects,
        external_modules=external_modules,
        metadata=metadata,
        context=context,
    ) as vts_model:
        model.save_model(vts_model.path_of(MODEL_FILENAME))

        return vts_model


def get_runnable(vts_model: vtsserving.Model) -> t.Type[vtsserving.Runnable]:
    """
    Private API: use :obj:`~vtsserving.Model.to_runnable` instead.
    """

    class LightGBMRunnable(vtsserving.Runnable):
        # LightGBM only supports GPU during training, not for inference.
        SUPPORTED_RESOURCES = ("cpu",)
        SUPPORTS_CPU_MULTI_THREADING = True

        def __init__(self):
            super().__init__()
            self.model = load_model(vts_model)

            self.predict_fns: dict[str, t.Callable[..., t.Any]] = {}
            for method_name in vts_model.info.signatures:
                try:
                    self.predict_fns[method_name] = getattr(self.model, method_name)  # type: ignore (incomplete ligthgbm type stubs)
                except AttributeError:
                    raise InvalidArgument(
                        f"No method with name {method_name} found for LightGBM model of type {self.model.__class__}"
                    )

    def add_runnable_method(method_name: str, options: ModelSignature):
        def _run(
            self: LightGBMRunnable,
            input_data: ext.NpNDArray | ext.PdDataFrame,
        ) -> ext.NpNDArray:
            res = self.predict_fns[method_name](input_data)
            return np.asarray(res)  # type: ignore (unknown ndarray types)

        LightGBMRunnable.add_method(
            _run,
            name=method_name,
            batchable=options.batchable,
            batch_dim=options.batch_dim,
            input_spec=options.input_spec,
            output_spec=options.output_spec,
        )

    for method_name, options in vts_model.info.signatures.items():
        add_runnable_method(method_name, options)

    return LightGBMRunnable
