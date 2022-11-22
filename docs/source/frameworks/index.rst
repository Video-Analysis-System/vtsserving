================
Framework Guides
================

Here is the list of supported ML libraries and formats in VtsServing. You can also find example
projects in the `vtsserving/examples <https://github.com/vtsserving/VtsServing/tree/main/examples>`_ directory.


.. grid:: 1 2 2 2
    :gutter: 3
    :margin: 0
    :padding: 3 4 0 0

    .. grid-item-card:: :doc:`/frameworks/catboost`
        :link: /frameworks/catboost
        :link-type: doc

    .. grid-item-card:: :doc:`/frameworks/fastai`
        :link: /frameworks/fastai
        :link-type: doc

    .. grid-item-card:: :doc:`/frameworks/keras`
        :link: /frameworks/keras
        :link-type: doc

    .. grid-item-card:: :doc:`/frameworks/lightgbm`
        :link: /frameworks/lightgbm
        :link-type: doc

    .. grid-item-card:: :doc:`/integrations/mlflow`
        :link: /integrations/mlflow
        :link-type: doc

    .. grid-item-card:: :doc:`/frameworks/onnx`
        :link: /frameworks/onnx
        :link-type: doc

    .. grid-item-card:: :doc:`/frameworks/pytorch`
        :link: /frameworks/pytorch
        :link-type: doc

    .. grid-item-card:: :doc:`/frameworks/pytorch_lightning`
        :link: /frameworks/pytorch_lightning
        :link-type: doc

    .. grid-item-card:: :doc:`/frameworks/sklearn`
        :link: /frameworks/sklearn
        :link-type: doc

    .. grid-item-card:: :doc:`/frameworks/tensorflow`
        :link: /frameworks/tensorflow
        :link-type: doc

    .. grid-item-card:: :doc:`/frameworks/transformers`
        :link: /frameworks/transformers
        :link-type: doc

    .. grid-item-card:: :doc:`/frameworks/xgboost`
        :link: /frameworks/xgboost
        :link-type: doc


Custom Models
-------------

.. grid:: 1 2 2 2
    :gutter: 3
    :margin: 0
    :padding: 3 4 0 0

    .. grid-item-card:: :doc:`/frameworks/picklable`
        :link: /frameworks/picklable
        :link-type: doc

    .. grid-item-card:: :ref:`concepts/runner:Custom Runner`
        :link: custom-runner
        :link-type: ref


Roadmap
-------

The following frameworks are supported in pre-1.0 VtsServing versions and are being migrated to the new 1.0 API. In the meantime, users may use :ref:`Custom Models <frameworks/index:Custom Models>` as a workaround.

- Detectron
- EasyOCR
- EvalML
- FastText
- Flax
- Gluon
- H2O
- Jax
- Neuropod
- ONNX-MLIR
- PaddlePaddle
- PyCaret
- PyTorch ignite
- SnapML
- Spacy
- Spark MLlib
- Statsmodels


.. admonition:: Help us improve the project!

    Found an issue or a TODO item? You're always welcome to make contributions to the
    project and its documentation. Check out the
    `VtsServing development guide <https://github.com/vtsserving/VtsServing/blob/main/DEVELOPMENT.md>`_
    and `documentation guide <https://github.com/vtsserving/VtsServing/blob/main/docs/README.md>`_
    to get started.


.. toctree::
    :hidden:

    catboost
    fastai
    keras
    lightgbm
    onnx
    picklable_model
    pytorch
    pytorch_lightning
    sklearn
    tensorflow
    transformers
    xgboost
