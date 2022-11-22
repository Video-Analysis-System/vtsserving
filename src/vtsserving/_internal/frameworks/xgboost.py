from __future__ import annotations

import os
import typing as t
import logging
from types import ModuleType
from typing import TYPE_CHECKING

import attr
import numpy as np

import vtsserving
from vtsserving import Tag
from vtsserving.models import ModelOptions
from vtsserving.exceptions import NotFound
from vtsserving.exceptions import InvalidArgument
from vtsserving.exceptions import VtsServingException
from vtsserving.exceptions import MissingDependencyException

from ..utils.pkg import get_pkg_version
from ..models.model import ModelContext

if TYPE_CHECKING:
    from vtsserving.types import ModelSignature
    from vtsserving.types import ModelSignatureDict

    from .. import external_typing as ext

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        "'xgboost' is required in order to use module 'vtsserving.xgboost', install xgboost with 'pip install xgboost'. For more information, refer to https://xgboost.readthedocs.io/en/latest/install.html"
    )

try:
    from xgboost import XGBModel
except ImportError:  # pragma: no cover
    # if sklearn is not installed, XGBoost will not expose XGBModel, so make
    # a dummy class ourselves
    class XGBModel:
        pass


MODULE_NAME = "vtsserving.xgboost"
MODEL_FILENAME = "saved_model.ubj"
API_VERSION = "v2"

logger = logging.getLogger(__name__)


@attr.define
class XGBoostOptions(ModelOptions):
    model_class: str | None = None


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
        model = vtsserving.xgboost.get("my_xgboost_model")
    """
    model = vtsserving.models.get(tag_like)
    if model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {model.tag} was saved with module {model.info.module}, not loading with {MODULE_NAME}."
        )
    return model


def load_model(
    vts_model: str | Tag | vtsserving.Model,
) -> xgb.Booster | xgb.XGBModel:
    """
    Load the XGBoost model with the given tag from the local VtsServing model store.

    Args:
        vts_model (``str`` ``|`` :obj:`~vtsserving.Tag` ``|`` :obj:`~vtsserving.Model`):
            Either the tag of the model to get from the store, or a VtsServing `~vtsserving.Model`
            instance to load the model from.
    Returns:
        The XGBoost model loaded from the model store or VtsServing :obj:`~vtsserving.Model`.
    Example:

    .. code-block:: python

        import vtsserving
        # target model must be from the VtsServing model store
        booster = vtsserving.xgboost.load_model("my_xgboost_model")
    """  # noqa: LN001
    if not isinstance(vts_model, vtsserving.Model):
        vts_model = get(vts_model)
        assert isinstance(vts_model, vtsserving.Model)

    if vts_model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {vts_model.tag} was saved with module {vts_model.info.module}, not loading with {MODULE_NAME}."
        )

    model_file = vts_model.path_of(MODEL_FILENAME)
    model_api_version = vts_model.info.api_version
    if model_api_version == "v1":
        model = xgb.Booster(model_file=model_file)
    else:
        if model_api_version != "v2":
            logger.warning(
                "Got an XGBoost model with an unsupported version '%s', unexpected errors may occur.",
                model_api_version,
            )
        model_class = t.cast(XGBoostOptions, vts_model.info.options).model_class
        if model_class is None:
            raise VtsServingException(
                f"Model '{vts_model.tag}' is missing the required 'model_class' option. This should not be possible; please file an issue if you encounter this error."
            )
        try:
            xgb_class: type[xgb.XGBModel] | type[xgb.Booster] = getattr(
                xgb, model_class
            )
        except AttributeError:
            if model_class != "Booster":
                raise VtsServingException(
                    f"Model '{vts_model.tag}' is an XGBoost Scikit-Learn model, but sklearn is not installed."
                ) from None
            else:
                raise VtsServingException(
                    "xgboost.Booster could not be found, your XGBoost installation may be corrupted. Ensure there is no file named 'xgboost.py' that may be being loaded instead of the XGBoost library."
                ) from None
        model: xgb.Booster | xgb.XGBModel = xgb_class()
        model.load_model(model_file)
    return model


def save_model(
    name: str,
    model: xgb.Booster | xgb.XGBModel,
    *,
    signatures: dict[str, ModelSignatureDict] | None = None,
    labels: dict[str, str] | None = None,
    custom_objects: dict[str, t.Any] | None = None,
    external_modules: t.List[ModuleType] | None = None,
    metadata: dict[str, t.Any] | None = None,
) -> vtsserving.Model:
    """
    Save an XGBoost model instance to the VtsServing model store.

    Args:
        name:
            The name to give to the model in the VtsServing store. This must be a valid
            :obj:`~vtsserving.Tag` name.
        model:
            The XGBoost model to be saved.
        signatures:
            Signatures of predict methods to be used. If not provided, the signatures default to
            ``{"predict": {"batchable": False}}``. See :obj:`~vtsserving.types.ModelSignature` for more
            details.
        labels:
            A default set of management labels to be associated with the model. An example is
            ``{"training-set": "data-1"}``.
        custom_objects:
            Custom objects to be saved with the model. An example is
            ``{"my-normalizer": normalizer}``.

            Custom objects are currently serialized with cloudpickle, but this implementation is
            subject to change.
        external_modules:
            user-defined additional python modules to be saved alongside the model or custom objects,
            e.g. a tokenizer module, preprocessor module, model configuration module
        metadata:
            Metadata to be associated with the model. An example is ``{"max_depth": 2}``.

            Metadata is intended for display in model management UI and therefore must be a default
            Python type, such as ``str`` or ``int``.
    Returns:
        A VtsServing tag with the user-defined name and a generated version.

    Example:

    .. code-block:: python

        import xgboost as xgb
        import vtsserving

        # read in data
        dtrain = xgb.DMatrix('demo/data/agaricus.txt.train')
        dtest = xgb.DMatrix('demo/data/agaricus.txt.test')
        # specify parameters via map
        param = dict(max_depth=2, eta=1, objective='binary:logistic')
        num_round = 2
        bst = xgb.train(param, dtrain, num_round)
        ...

        # `save` the booster to VtsServing modelstore:
        vts_model = vtsserving.xgboost.save_model("my_xgboost_model", bst, booster_params=param)
    """  # noqa: LN001
    if isinstance(model, xgb.Booster):
        model_class = "Booster"
    elif isinstance(model, XGBModel):
        model_class = model.__class__.__name__
    else:
        raise TypeError(f"Given model ({model}) is not a xgboost.Booster.")

    context: ModelContext = ModelContext(
        framework_name="xgboost",
        framework_versions={"xgboost": get_pkg_version("xgboost")},
    )

    if signatures is None:
        signatures = {
            "predict": {"batchable": False},
        }
        logger.info(
            'Using the default model signature for xgboost (%s) for model "%s".',
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
        options=XGBoostOptions(model_class=model_class),
    ) as vts_model:
        model.save_model(vts_model.path_of(MODEL_FILENAME))  # type: ignore (incomplete XGBoost types)

        return vts_model


def get_runnable(vts_model: vtsserving.Model) -> t.Type[vtsserving.Runnable]:
    """
    Private API: use :obj:`~vtsserving.Model.to_runnable` instead.
    """

    class XGBoostRunnable(vtsserving.Runnable):
        SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
        SUPPORTS_CPU_MULTI_THREADING = True

        def __init__(self):
            super().__init__()
            self.model = load_model(vts_model)

            self.booster = (
                self.model
                if isinstance(self.model, xgb.Booster)
                else self.model.get_booster()
            )

            # check for resources
            if os.getenv("CUDA_VISIBLE_DEVICES") not in (None, "", "-1"):
                self.booster.set_param({"predictor": "gpu_predictor", "gpu_id": 0})  # type: ignore (incomplete XGBoost types)
            else:
                nthreads = os.getenv("OMP_NUM_THREADS")
                if nthreads is not None and nthreads != "":
                    nthreads = max(int(nthreads), 1)
                else:
                    nthreads = 1
                self.booster.set_param({"predictor": "cpu_predictor", "nthread": nthreads})  # type: ignore (incomplete XGBoost types)

            self.predict_fns: dict[str, t.Callable[..., t.Any]] = {}
            for method_name in vts_model.info.signatures:
                try:
                    self.predict_fns[method_name] = getattr(self.model, method_name)
                except AttributeError:
                    raise InvalidArgument(
                        f"No method with name {method_name} found for XGBoost model of type {self.model.__class__}"
                    )

    def add_runnable_method(method_name: str, options: ModelSignature):
        def _run(
            self: XGBoostRunnable,
            input_data: ext.NpNDArray
            | ext.PdDataFrame,  # TODO: add support for DMatrix
        ) -> ext.NpNDArray:
            if isinstance(self.model, xgb.Booster):
                inp = xgb.DMatrix(input_data)
            else:
                inp = input_data

            res = self.predict_fns[method_name](inp)
            return np.asarray(res)  # type: ignore (incomplete np types)

        XGBoostRunnable.add_method(
            _run,
            name=method_name,
            batchable=options.batchable,
            batch_dim=options.batch_dim,
            input_spec=options.input_spec,
            output_spec=options.output_spec,
        )

    for method_name, options in vts_model.info.signatures.items():
        add_runnable_method(method_name, options)

    return XGBoostRunnable
