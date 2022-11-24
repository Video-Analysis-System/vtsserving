=======
fast.ai
=======

fastai is a popular deep learning library which provides high-level components for practioners to get state-of-the-art results in standard deep learning domains, as well as low-level components
for researchers to build new approaches. To learn more about fastai, visit their `documentation <docs.fast.ai>`_.

VtsServing provides native support for `fastai <https://github.com/fastai/fastai>`_, and this guide provides an overview of how to use VtsServing with fastai.

Compatibility 
-------------

VtsServing requires fastai **version 2** or higher to be installed. 

VtsServing does not support fastai version 1. If you are using fastai version 1, consider using :ref:`concepts/runner:Custom Runner`.

Saving a trained fastai learner
--------------------------------

This example is based on `Transfer Learning with text <https://docs.fast.ai/tutorial.text.html#The-ULMFiT-approach>`_ from fastai.

.. code-block:: python

   from fastai.basics import URLs
   from fastai.metrics import accuracy
   from fastai.text.data import DataBlock
   from fastai.text.data import TextBlock
   from fastai.text.data import untar_data
   from fastai.text.data import CategoryBlock
   from fastai.text.models import AWD_LSTM
   from fastai.text.learner import text_classifier_learner
   from fastai.data.transforms import parent_label
   from fastai.data.transforms import get_text_files
   from fastai.data.transforms import GrandparentSplitter

   # Download IMDB dataset
   path = untar_data(URLs.IMDB)

   # Create IMDB DataBlock
   imdb = DataBlock(
       blocks=(TextBlock.from_folder(path), CategoryBlock),
       get_items=get_text_files,
       get_y=parent_label,
       splitter=GrandparentSplitter(valid_name="test"),
   )
   dls = imdb.dataloaders(path)

   # define a Learner object
   learner = text_classifier_learner(
        dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy
    )

   # quickly fine tune the model
   learner.fine_tune(4, 1e-2)

   # output:
   # epoch     train_loss  valid_loss  accuracy  time
   # 0         0.453252    0.395130    0.822080  36:45

   learner.predict("I really liked that movie!")

   # output:
   # ('pos', TensorText(1), TensorText([0.1216, 0.8784]))


After training, use :obj:`~vtsserving.fastai.save_model` to save the `Learner <https://docs.fast.ai/learner.html#Learner>`_ instance to VtsServing model store.

.. code-block:: python

   vtsserving.fastai.save_model("fastai_sentiment", learner)



To verify that the saved learner can be loaded properly:

.. code-block:: python

   learner = vtsserving.fastai.load_model("fastai_sentiment:latest")

   learner.predict("I really liked that movie!")


Building a Service using fastai
--------------------------------

.. seealso::

   :ref:`Building a Service <concepts/service:Service and APIs>`: more information on creating a prediction service with VtsServing.

.. code-block:: python

   import vtsserving

   import numpy as np

   from vtsserving.io import Text
   from vtsserving.io import NumpyNdarray

   runner = vtsserving.fastai.get("fastai_sentiment:latest").to_runner()

   svc = vtsserving.Service("fast_sentiment", runners=[runner])


   @svc.api(input=Text(), output=NumpyNdarray())
   async def classify_text(text: str) -> np.ndarray:
      # returns sentiment score of a given text
      res = await runner.predict.async_run(text)
      return np.asarray(res[-1])


When constructing a :ref:`vtsfile.yaml <concepts/vts:Vts Build Options>`,
there are two ways to include fastai as a dependency, via ``python`` or
``conda``:

.. tab-set::

   .. tab-item:: python

      .. code-block:: yaml

         python:
	   packages:
	     - fastai

   .. tab-item:: conda

      .. code-block:: yaml

         conda:
           channels:
           - fastchan
           dependencies:
           - fastai


Using Runners
-------------

.. seealso::

   See :ref:`concepts/runner:Using Runners` doc for a general introduction to the Runner concept and its usage.


``runner.predict.run`` is generally a drop-in replacement for ``learner.predict`` regardless of the learner type 
for executing the prediction in the model runner. A fastai runner will receive the same inputs type as 
the given learner.


For example, Runner created from a `Tabular learner <https://docs.fast.ai/tabular.learner.html>`_ model will
accept a :obj:`pandas.DataFrame` as input, where as a Text learner based runner will accept a :obj:`str` as input.


Using PyTorch layer
-------------------

Since fastai is built on top of PyTorch, it is also possible to use PyTorch
models from within a fastai learner directly for inference. Note that by using
the PyTorch layer, you will not be able to use the fastai :obj:`Learner`'s
features such as :code:`.predict()`, :code:`.get_preds()`, etc.

To get the PyTorch model, access it via ``learner.model``:

.. code-block:: python

   import vtsserving

   vtsserving.pytorch.save_model(
      "my_pytorch_model", learner.model, signatures={"__call__": {"batchable": True}}
   )

Learn more about using PyTorch with VtsServing :ref:`here <frameworks/pytorch:PyTorch>`.

Using GPU
---------

Since fastai doesn't support using GPU for inference, VtsServing
can only support CPU inference with fastai models.

Additionally, if the model uses ``mixed_precision``, then the loaded model will also be converted to FP32.
See `mixed precision <https://docs.fast.ai/callback.fp16.html>`_ to learn more about mixed precision.

If you need to use GPU for inference, you can :ref:`use the PyTorch layer <frameworks/fastai:Using PyTorch layer>`.

Adaptive batching 
~~~~~~~~~~~~~~~~~

fastai's ``Learner#predict`` does not support taking batch input for inference, hence
the adaptive batching feature in VtsServing is not available for fastai models.

The default signature has :code:`batchable` set to :code:`False`.

If you need to use adaptive batching for inference, you can :ref:`use the PyTorch layer <frameworks/fastai:Using PyTorch layer>`.
