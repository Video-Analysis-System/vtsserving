from __future__ import annotations

import os
import typing as t
import logging
from types import ModuleType
from typing import TYPE_CHECKING

import attr

import vtsserving
from vtsserving import Tag
from vtsserving.models import ModelContext
from vtsserving.models import ModelOptions
from vtsserving.exceptions import NotFound
from vtsserving.exceptions import VtsServingException
from vtsserving.exceptions import MissingDependencyException

from ..utils.pkg import get_pkg_version
from ..utils.pkg import PackageNotFoundError
from .utils.onnx import gen_input_casting_func

if TYPE_CHECKING:

    from vtsserving.types import ModelSignature
    from vtsserving.types import ModelSignatureDict

    from .utils.onnx import ONNXArgType
    from .utils.onnx import ONNXArgCastedType

    ProvidersType = list[str | tuple[str, dict[str, t.Any]]]


try:
    import onnx
    from google.protobuf.json_format import MessageToDict

except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        "onnx is required in order to use module 'vtsserving.onnx', install onnx with 'pip install onnx'. For more information, refer to https://onnx.ai/get-started.html"
    )

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        "'onnxruntime' or 'onnxruntime-gpu' is required by 'vtsserving.onnx', install onnxruntime or onnxruntime-gpu with 'pip install onnxruntime' or 'pip install onnxruntime-gpu'. For more information, refer to https://onnxruntime.ai/"
    )

MODULE_NAME = "vtsserving.onnx"
MODEL_FILENAME = "saved_model.onnx"
API_VERSION = "v2"

logger = logging.getLogger(__name__)


# helper methods
def _yield_providers(
    iterable: t.Sequence[t.Any],
) -> t.Generator[str, None, None]:  # pragma: no cover
    if isinstance(iterable, tuple):
        yield iterable[0]
    elif isinstance(iterable, str):
        yield iterable
    else:
        yield from iterable


def flatten_list(lst: t.List[t.Any]) -> t.List[str]:  # pragma: no cover
    return [k for i in lst for k in _yield_providers(i)]


@attr.define
class ONNXOptions(ModelOptions):
    """Options for the ONNX model"""

    input_specs: dict[str, list[dict[str, t.Any]]] = attr.field(factory=dict)
    output_specs: dict[str, list[dict[str, t.Any]]] = attr.field(factory=dict)
    providers: t.Optional[list[str]] = attr.field(default=None)
    session_options: t.Optional["ort.SessionOptions"] = attr.field(default=None)


def get(tag_like: str | Tag) -> vtsserving.Model:
    """
    Get the VtsServing model with the given tag.

    Args:
        tag_like: The tag of the model to retrieve from the model store.

    Returns:
        :obj:`~vtsserving.Model`: A VtsServing :obj:`~vtsserving.Model` with the matching tag.

    Example:

    .. code-block:: python

       import vtsserving
       # target model must be from the VtsServing model store
       model = vtsserving.onnx.get("onnx_resnet50")
    """
    model = vtsserving.models.get(tag_like)
    if model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {model.tag} was saved with module {model.info.module}, not loading with {MODULE_NAME}."
        )
    return model


def _load_raw_model(vts_model: str | Tag | vtsserving.Model) -> onnx.ModelProto:

    if not isinstance(vts_model, vtsserving.Model):
        vts_model = get(vts_model)

    model_path = vts_model.path_of(MODEL_FILENAME)
    raw_model = onnx.load(model_path)
    return raw_model


def load_model(
    vts_model: str | Tag | vtsserving.Model,
    *,
    providers: ProvidersType | None = None,
    session_options: ort.SessionOptions | None = None,
) -> ort.InferenceSession:
    """
    Load the onnx model with the given tag from the local VtsServing model store.

    Args:
        vts_model (``str`` ``|`` :obj:`~vtsserving.Tag` ``|`` :obj:`~vtsserving.Model`):
            Either the tag of the model to get from the store, or a VtsServing `~vtsserving.Model`
            instance to load the model from.
        providers (`List[Union[str, Tuple[str, Dict[str, Any]]`, `optional`, default to :code:`None`):
            Different providers provided by users. By default VtsServing will use ``["CPUExecutionProvider"]``
            when loading a model.
        session_options (`onnxruntime.SessionOptions`, `optional`, default to :code:`None`):
            SessionOptions per use case. If not specified, then default to :code:`None`.

    Returns:
        :obj:`onnxruntime.InferenceSession`:
            An instance of ONNX Runtime inference session created using ONNX model loaded from the model store.

    Example:

    .. code-block:: python

        import vtsserving
        sess = vtsserving.onnx.load_model("my_onnx_model")
    """  # noqa

    if not isinstance(vts_model, vtsserving.Model):
        vts_model = get(vts_model)

    if vts_model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {vts_model.tag} was saved with module {vts_model.info.module}, not loading with {MODULE_NAME}."
        )

    if providers:
        if not all(i in ort.get_all_providers() for i in flatten_list(providers)):
            raise VtsServingException(f"'{providers}' cannot be parsed by `onnxruntime`")
    else:
        providers = ["CPUExecutionProvider"]

    return ort.InferenceSession(
        vts_model.path_of(MODEL_FILENAME),
        sess_options=session_options,
        providers=providers,
    )


def save_model(
    name: str,
    model: onnx.ModelProto,
    *,
    signatures: dict[str, ModelSignatureDict] | dict[str, ModelSignature] | None = None,
    labels: dict[str, str] | None = None,
    custom_objects: dict[str, t.Any] | None = None,
    external_modules: t.List[ModuleType] | None = None,
    metadata: dict[str, t.Any] | None = None,
) -> vtsserving.Model:
    """Save a onnx model instance to the VtsServing model store.

    Args:
        name (``str``):
            The name to give to the model in the VtsServing store. This must be a valid
            :obj:`~vtsserving.Tag` name.
        model (:obj:`~onnx.ModelProto`):
            The ONNX model to be saved.
        signatures (``dict[str, ModelSignatureDict]``, optional):
            Signatures of predict methods to be used. If not provided, the signatures default to
            ``{"run": {"batchable": False}}``. Because we are using :obj:``onnxruntime.InferenceSession``,
            the only allowed method name is "run"
            See :obj:`~vtsserving.types.ModelSignature` for more details.
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
            Metadata to be associated with the model. An example is ``{"bias": 4}``.

            Metadata is intended for display in a model management UI and therefore must be a
            default Python type, such as ``str`` or ``int``.

    Returns:

        :obj:`~vtsserving.Model`: A VtsServing model containing the saved
        ONNX model instance.  store.

    Example:

    .. code-block:: python

        import vtsserving

        import torch
        import torch.nn as nn

        class ExtendedModel(nn.Module):
            def __init__(self, D_in, H, D_out):
                # In the constructor we instantiate two nn.Linear modules and assign them as
                #  member variables.
                super(ExtendedModel, self).__init__()
                self.linear1 = nn.Linear(D_in, H)
                self.linear2 = nn.Linear(H, D_out)

            def forward(self, x, bias):
                # In the forward function we accept a Tensor of input data and an optional bias
                h_relu = self.linear1(x).clamp(min=0)
                y_pred = self.linear2(h_relu)
                return y_pred + bias


        N, D_in, H, D_out = 64, 1000, 100, 1
        x = torch.randn(N, D_in)
        model = ExtendedModel(D_in, H, D_out)

        input_names = ["x", "bias"]
        output_names = ["output1"]

        tmpdir = "/tmp/model"
        model_path = os.path.join(tmpdir, "test_torch.onnx")
        torch.onnx.export(
            model,
            (x, torch.Tensor([1.0])),
            model_path,
            input_names=input_names,
            output_names=output_names,
        )

        vts_model = vtsserving.onnx.save_model("onnx_model", model_path, signatures={"run": "batchable": True})

    """

    # prefer "onnxruntime-gpu" over "onnxruntime" for framework versioning
    _onnxruntime_version = None
    _onnxruntime_pkg = None
    _PACKAGE = ["onnxruntime-gpu", "onnxruntime", "onnxruntime-silicon"]
    for p in _PACKAGE:
        try:
            _onnxruntime_version = get_pkg_version(p)
            _onnxruntime_pkg = p
            break
        except PackageNotFoundError:
            pass
    assert (
        _onnxruntime_pkg is not None and _onnxruntime_version is not None
    ), "Failed to find onnxruntime package version."

    assert _onnxruntime_version is not None, "onnxruntime is not installed"
    if not isinstance(model, onnx.ModelProto):
        raise TypeError(f"Given model ({model}) is not a onnx.ModelProto.")

    context = ModelContext(
        framework_name="onnx",
        framework_versions={
            "onnx": get_pkg_version("onnx"),
            _onnxruntime_pkg: _onnxruntime_version,
        },
    )

    if signatures is None:
        signatures = {
            "run": {"batchable": False},
        }
        logger.info(
            'Using the default model signature for ONNX (%s) for model "%s".',
            signatures,
            name,
        )
    else:
        provided_methods = list(signatures.keys())
        if provided_methods != ["run"]:
            raise VtsServingException(
                f"Provided method names {[m for m  in provided_methods if m != 'run']} are invalid. 'vtsserving.onnx' will load ONNX model into an 'onnxruntime.InferenceSession' for inference, so the only supported method name is 'run'."
            )

    run_input_specs = [MessageToDict(inp) for inp in model.graph.input]
    run_output_specs = [MessageToDict(out) for out in model.graph.output]
    input_specs = {"run": run_input_specs}
    output_specs = {"run": run_output_specs}

    options = ONNXOptions(input_specs=input_specs, output_specs=output_specs)

    with vtsserving.models.create(
        name,
        module=MODULE_NAME,
        api_version=API_VERSION,
        signatures=signatures,
        labels=labels,
        options=options,
        custom_objects=custom_objects,
        external_modules=external_modules,
        metadata=metadata,
        context=context,
    ) as vts_model:
        onnx.save(model, vts_model.path_of(MODEL_FILENAME))

        return vts_model


def get_runnable(vts_model: vtsserving.Model) -> t.Type[vtsserving.Runnable]:
    """
    Private API: use :obj:`~vtsserving.Model.to_runnable` instead.
    """

    # backward compatibility for v1, load raw model to infer
    # input_specs/output_specs for onnx model
    if vts_model.info.api_version == "v1":

        raw_model: onnx.ModelProto | None = None
        options = t.cast(ONNXOptions, vts_model.info.options)

        if not options.input_specs:
            raw_model = _load_raw_model(vts_model)
            run_input_specs = [MessageToDict(inp) for inp in raw_model.graph.input]
            input_specs = {"run": run_input_specs}
            vts_model = vts_model.with_options(input_specs=input_specs)

        if not options.output_specs:
            raw_model = raw_model or _load_raw_model(vts_model)
            run_output_specs = [MessageToDict(out) for out in raw_model.graph.output]
            output_specs = {"run": run_output_specs}
            vts_model = vts_model.with_options(output_specs=output_specs)

    class ONNXRunnable(vtsserving.Runnable):
        SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
        SUPPORTS_CPU_MULTI_THREADING = True

        def __init__(self):
            super().__init__()

            session_options = (
                vts_model.info.options.session_options or ort.SessionOptions()
            )

            # check for resources
            available_gpus = os.getenv("CUDA_VISIBLE_DEVICES")
            if available_gpus is not None and available_gpus not in ("", "-1"):
                # assign GPU resources
                providers = vts_model.info.options.providers or [
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ]

            else:
                # assign CPU resources

                # If onnxruntime-gpu is installed,
                # CUDAExecutionProvider etc. will be available even no
                # GPU is presented in system, which may result some
                # error when initializing ort.InferenceSession
                providers = vts_model.info.options.providers or [
                    "CPUExecutionProvider"
                ]

                # set CPUExecutionProvider parallelization options
                # TODO @larme: follow onnxruntime issue 11668 and
                # 10330 to decide best cpu parallelization strategy
                thread_count = int(os.getenv("VTSSERVING_NUM_THREAD", 1))
                session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
                if session_options.intra_op_num_threads != 0:
                    logger.warning(
                        "Overriding specified 'session_options.intra_op_num_threads'."
                    )
                session_options.intra_op_num_threads = thread_count
                if session_options.inter_op_num_threads != 0:
                    logger.warning(
                        "Overriding specified 'session_options.inter_op_num_threads'."
                    )
                session_options.inter_op_num_threads = thread_count

            self.model = load_model(
                vts_model, session_options=session_options, providers=providers
            )

            self.predict_fns: dict[str, t.Callable[..., t.Any]] = {}
            for method_name in vts_model.info.signatures:
                self.predict_fns[method_name] = getattr(self.model, method_name)

    def add_runnable_method(
        method_name: str,
        signatures: ModelSignature,
        input_specs: list[dict[str, t.Any]],
        output_specs: list[dict[str, t.Any]],
    ):

        casting_funcs = [gen_input_casting_func(spec) for spec in input_specs]

        if len(output_specs) > 1:

            def _process_output(outs):
                return tuple(outs)

        else:

            def _process_output(outs):
                return outs[0]

        def _run(self: ONNXRunnable, *args: ONNXArgType) -> t.Any:
            casted_args = [
                casting_funcs[idx](args[idx]) for idx in range(len(casting_funcs))
            ]

            input_names: dict[str, ONNXArgCastedType] = {
                i.name: val for i, val in zip(self.model.get_inputs(), casted_args)
            }
            output_names: list[str] = [o.name for o in self.model.get_outputs()]
            raw_outs = self.predict_fns[method_name](output_names, input_names)
            return _process_output(raw_outs)

        ONNXRunnable.add_method(
            _run,
            name=method_name,
            batchable=signatures.batchable,
            batch_dim=signatures.batch_dim,
            input_spec=signatures.input_spec,
            output_spec=signatures.output_spec,
        )

    for method_name, signatures in vts_model.info.signatures.items():
        options = t.cast(ONNXOptions, vts_model.info.options)
        input_specs = options.input_specs[method_name]
        output_specs = options.output_specs[method_name]
        add_runnable_method(method_name, signatures, input_specs, output_specs)

    return ONNXRunnable
