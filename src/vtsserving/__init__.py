# pylint: skip-file
"""
VtsServing
=======

VtsServing is the unified ML Model Serving framework. Data Scientists and ML Engineers use
VtsServing to:

* Accelerate and standardize the process of taking ML models to production across teams
* Build reliable, scalable, and high performance model serving systems
* Provide a flexible MLOps platform that grows with your Data Science needs

To learn more, visit VtsServing documentation at: http://docs.vtsserving.org
To get involved with the development, find us on GitHub: https://github.com/vtsserving
And join us in the VtsServing slack community: https://l.linklyhq.com/l/ktOh
"""

from typing import TYPE_CHECKING

from ._internal.configuration import VTSSERVING_VERSION as __version__
from ._internal.configuration import load_global_config

# Inject dependencies and configurations
load_global_config()

# Bento management APIs
from .vtss import get
from .vtss import list  # pylint: disable=W0622
from .vtss import pull
from .vtss import push
from .vtss import delete
from .vtss import export_vts
from .vtss import import_vts

# VtsServing built-in types
from ._internal.tag import Tag
from ._internal.vts import Bento
from ._internal.models import Model
from ._internal.runner import Runner
from ._internal.runner import Runnable
from ._internal.context import InferenceApiContext as Context
from ._internal.service import Service
from ._internal.utils.http import Cookie
from ._internal.yatai_client import YataiClient
from ._internal.monitoring.api import monitor
from ._internal.service.loader import load

# Framework specific modules, model management and IO APIs are lazily loaded upon import.
if TYPE_CHECKING:
    from . import h2o
    from . import flax
    from . import onnx
    from . import gluon
    from . import keras
    from . import spacy
    from . import fastai
    from . import mlflow
    from . import paddle
    from . import easyocr
    from . import pycaret
    from . import pytorch
    from . import sklearn
    from . import xgboost
    from . import catboost
    from . import lightgbm
    from . import onnxmlir
    from . import detectron
    from . import tensorflow
    from . import statsmodels
    from . import torchscript
    from . import transformers
    from . import tensorflow_v1
    from . import picklable_model
    from . import pytorch_lightning

    # isort: off
    from . import io
    from . import models
    from . import metrics  # Prometheus metrics client
    from . import container  # Container API

    # isort: on
else:
    from ._internal.utils import LazyLoader as _LazyLoader

    catboost = _LazyLoader("vtsserving.catboost", globals(), "vtsserving.catboost")
    detectron = _LazyLoader("vtsserving.detectron", globals(), "vtsserving.detectron")
    easyocr = _LazyLoader("vtsserving.easyocr", globals(), "vtsserving.easyocr")
    flax = _LazyLoader("vtsserving.flax", globals(), "vtsserving.flax")
    fastai = _LazyLoader("vtsserving.fastai", globals(), "vtsserving.fastai")
    gluon = _LazyLoader("vtsserving.gluon", globals(), "vtsserving.gluon")
    h2o = _LazyLoader("vtsserving.h2o", globals(), "vtsserving.h2o")
    lightgbm = _LazyLoader("vtsserving.lightgbm", globals(), "vtsserving.lightgbm")
    mlflow = _LazyLoader("vtsserving.mlflow", globals(), "vtsserving.mlflow")
    onnx = _LazyLoader("vtsserving.onnx", globals(), "vtsserving.onnx")
    onnxmlir = _LazyLoader("vtsserving.onnxmlir", globals(), "vtsserving.onnxmlir")
    keras = _LazyLoader("vtsserving.keras", globals(), "vtsserving.keras")
    paddle = _LazyLoader("vtsserving.paddle", globals(), "vtsserving.paddle")
    pycaret = _LazyLoader("vtsserving.pycaret", globals(), "vtsserving.pycaret")
    pytorch = _LazyLoader("vtsserving.pytorch", globals(), "vtsserving.pytorch")
    pytorch_lightning = _LazyLoader(
        "vtsserving.pytorch_lightning", globals(), "vtsserving.pytorch_lightning"
    )
    sklearn = _LazyLoader("vtsserving.sklearn", globals(), "vtsserving.sklearn")
    picklable_model = _LazyLoader(
        "vtsserving.picklable_model", globals(), "vtsserving.picklable_model"
    )
    spacy = _LazyLoader("vtsserving.spacy", globals(), "vtsserving.spacy")
    statsmodels = _LazyLoader("vtsserving.statsmodels", globals(), "vtsserving.statsmodels")
    tensorflow = _LazyLoader("vtsserving.tensorflow", globals(), "vtsserving.tensorflow")
    tensorflow_v1 = _LazyLoader(
        "vtsserving.tensorflow_v1", globals(), "vtsserving.tensorflow_v1"
    )
    torchscript = _LazyLoader("vtsserving.torchscript", globals(), "vtsserving.torchscript")
    transformers = _LazyLoader(
        "vtsserving.transformers", globals(), "vtsserving.transformers"
    )
    xgboost = _LazyLoader("vtsserving.xgboost", globals(), "vtsserving.xgboost")

    io = _LazyLoader("vtsserving.io", globals(), "vtsserving.io")
    models = _LazyLoader("vtsserving.models", globals(), "vtsserving.models")
    metrics = _LazyLoader("vtsserving.metrics", globals(), "vtsserving.metrics")
    container = _LazyLoader("vtsserving.container", globals(), "vtsserving.container")

    del _LazyLoader

__all__ = [
    "__version__",
    "Context",
    "Cookie",
    "Service",
    "models",
    "metrics",
    "container",
    "io",
    "Tag",
    "Model",
    "Runner",
    "Runnable",
    "YataiClient",  # Yatai REST API Client
    # vts APIs
    "list",
    "get",
    "delete",
    "import_vts",
    "export_vts",
    "load",
    "push",
    "pull",
    "Bento",
    # Framework specific modules
    "catboost",
    "detectron",
    "easyocr",
    "flax",
    "fastai",
    "gluon",
    "h2o",
    "lightgbm",
    "mlflow",
    "onnx",
    "onnxmlir",
    "paddle",
    "picklable_model",
    "pycaret",
    "pytorch",
    "pytorch_lightning",
    "keras",
    "sklearn",
    "spacy",
    "statsmodels",
    "tensorflow",
    "tensorflow_v1",
    "torchscript",
    "transformers",
    "xgboost",
    "monitor",
]
