=============
Using Runners
=============

What is Runner?
---------------

In VtsServing, Runner represents a unit of computation that can be executed on a remote
Python worker and scales independently.

Runner allows :ref:`vtsserving.Service <reference/core:vtsserving.Service>` to parallelize
multiple instances of a :ref:`vtsserving.Runnable <reference/core:vtsserving.Runnable>` class,
each on its own Python worker. When a VtsServer is launched, a group of runner worker
processes will be created, and :code:`run` method calls made from the
:code:`vtsserving.Service` code will be scheduled among those runner workers.

Runner also supports :doc:`/guides/batching`. For a
:ref:`vtsserving.Runnable <reference/core:vtsserving.Runnable>` configured with batching,
multiple :code:`run` method invocations made from other processes can be dynamically
grouped into one batch execution in real-time. This is especially beneficial for compute
intensive workloads such as model inference, helps to bring better performance through
vectorization or multi-threading.


Pre-built Model Runners
-----------------------

VtsServing provides pre-built Runners implemented for each ML framework supported. These
pre-built runners are carefully configured to work well with each specific ML framework.
They handle working with GPU when GPU is available, set the number of threads and number
of workers automatically, and convert the model signatures to corresponding Runnable
methods.

.. code:: python

    trained_model = train()

    vtsserving.pytorch.save_model(
        "demo_mnist",  # model name in the local model store
        trained_model,  # model instance being saved
        signatures={   # model signatures for runner inference
            "predict": {
                "batchable": True,
                "batch_dim": 0,
            }
        }
    )

    runner = vtsserving.pytorch.get("demo_mnist:latest").to_runner()
    runner.init_local()
    runner.predict.run( MODEL_INPUT )


.. _custom-runner:

Custom Runner
-------------

Creating a Runnable
^^^^^^^^^^^^^^^^^^^

Runner can be created from a :ref:`vtsserving.Runnable <reference/core:vtsserving.Runnable>`
class. By implementing a :code:`Runnable` class, users can create Runner instances that
runs custom logic. Here's an example, creating an NLTK runner that does sentiment
analysis with a pre-trained model:

.. literalinclude:: ../../../examples/custom_runner/nltk_pretrained_model/service.py
   :language: python
   :caption: `service.py`

.. note::

    Full code example can be found :github:`here <tree/main/examples/custom_runner/nltk_pretrained_model>`_.


The constant attribute ``SUPPORTED_RESOURCES`` indicates which resources this Runnable class
implementation supports. The only currently pre-defined resources are ``"cpu"`` and
``"nvidia.com/gpu"``.

The constant attribute ``SUPPORTS_CPU_MULTI_THREADING`` indicates whether or not the runner supports
CPU multi-threading.

.. tip::

    Neither constant can be set inside of the runner's ``__init__`` or ``__new__`` methods, as they are class-level attributes. The reason being VtsServing’s scheduling policy is not invoked in runners’ initialization code, as instantiating runners can be quite expensive.

Since NLTK library doesn't support utilizing GPU or multiple CPU cores natively, supported resources
is specified as :code:`("cpu",)`, and ``SUPPORTS_CPU_MULTI_THREADING`` is set to False. This is the default configuration.
This information is then used by the VtsServer scheduler to determine the worker pool size for this runner.

The :code:`vtsserving.Runnable.method` decorator is used for creating
:code:`RunnableMethod` - the decorated method will be exposed as the runner interface
for accessing remotely. :code:`RunnableMethod` can be configured with a signature,
which is defined same as the :ref:`concepts/model:Model Signatures`.


Reusable Runnable
^^^^^^^^^^^^^^^^^

Runnable class can also take :code:`__init__` parameters to customize its behavior for
different scenarios. The same Runnable class can also be used to create multiple runners
and used in the same service. For example:

.. code-block:: python
   :caption: `service.py`

    import vtsserving
    import torch

    class MyModelRunnable(vtsserving.Runnable):
        SUPPORTED_RESOURCES = ("nvidia.com/gpu",)
        SUPPORTS_CPU_MULTI_THREADING = True

        def __init__(self, model_file):
            self.model = torch.load_model(model_file)

        @vtsserving.Runnable.method(batchable=True, batch_dim=0)
        def predict(self, input_tensor):
            return self.model(input_tensor)

    my_runner_1 = vtsserving.Runner(
        MyModelRunnable,
        name="my_runner_1",
        runnable_init_params={
            "model_file": "./saved_model_1.pt",
        }
    )
    my_runner_2 = vtsserving.Runner(
        MyModelRunnable,
        name="my_runner_2",
        runnable_init_params={
            "model_file": "./saved_model_2.pt",
        }
    )

    svc = vtsserving.Service(__name__, runners=[my_runner_1, my_runner_2])

.. epigraph::
    All runners presented in one ``vtsserving.Service`` object must have unique names.

.. note::

    The default Runner name is the Runnable class name. When using the same Runnable
    class to create multiple runners and use them in the same service, user must rename
    runners by specifying the ``name`` parameter when creating the runners. Runner
    name are a key to configuring individual runner at deploy time and to runner related
    logging and tracing features.


Custom Model Runner
^^^^^^^^^^^^^^^^^^^

Custom Runnable built with Model from VtsServing's model store:

.. code::

    from typing import Any

    import vtsserving
    from vtsserving.io import JSON
    from vtsserving.io import NumpyNdarray
    from numpy.typing import NDArray

    vts_model = vtsserving.pytorch.get("spam_detection:latest")

    class SpamDetectionRunnable(vtsserving.Runnable):
        SUPPORTED_RESOURCES = ("cpu",)
        SUPPORTS_CPU_MULTI_THREADING = True

        def __init__(self):
            # load the model instance
            self.classifier = vtsserving.sklearn.load_model(vts_model)

        @vtsserving.Runnable.method(batchable=False)
        def is_spam(self, input_data: NDArray[Any]) -> NDArray[Any]:
            return self.classifier.predict(input_data)

    spam_detection_runner = vtsserving.Runner(SpamDetectionRunnable, models=[vts_model])
    svc = vtsserving.Service("spam_detector", runners=[spam_detection_runner])

    @svc.api(input=NumpyNdarray(), output=JSON())
    def analysis(input_text: NDArray[Any]) -> dict[str, Any]:
        return {"res": spam_detection_runner.is_spam.run(input_text)}


Serving Multiple Models via Runner
----------------------------------

Serving multiple models in the same workflow is also a common pattern in VtsServing’s prediction framework. This pattern can be achieved by simply instantiating multiple runners up front and passing them to the service that’s being created. Each runner/model will be configured with its’ own resources and run autonomously. If no configuration is passed, VtsServing will then determine the optimal resources to allocate to each runner.


Sequential Runs
^^^^^^^^^^^^^^^

.. code:: python

    import asyncio
    import vtsserving
    import PIL.Image

    import vtsserving
    from vtsserving.io import Image, Text

    transformers_runner = vtsserving.transformers.get("sentiment_model:latest").to_runner()
    ocr_runner = vtsserving.easyocr.get("ocr_model:latest").to_runner()

    svc = vtsserving.Service("sentiment_analysis", runners=[transformers_runner, ocr_runner])

    @svc.api(input=Image(),output=Text())
    def classify(input: PIL.Image.Image) -> str:
        ocr_text = ocr_runner.run(input)
        return transformers_runner.run(ocr_text)

It’s as simple as creating two runners and invoking them synchronously in your prediction endpoint. Note that an async endpoint is often preferred in these use cases as the primary event loop is yielded while waiting for other IO-expensive tasks. 

For example, the same API above can be achieved as an ``async`` endpoint:


.. code:: python

    @svc.api(input=Image(),output=Text())
    async def classify_async(input: PIL.Image.Image) -> str:
        ocr_text = await ocr_runner.async_run(input)
        return await transformers_runner.async_run(ocr_text)


Concurrent Runs
^^^^^^^^^^^^^^^

In cases where certain steps can be executed concurrently, :code:`asyncio.gather` can be used to aggregate results from multiple concurrent runs. For instance, if you are running two models simultaneously, you could invoke ``asyncio.gather`` as follows:

.. code-block:: python

    import asyncio
    import PIL.Image

    import vtsserving
    from vtsserving.io import Image, Text

    preprocess_runner = vtsserving.Runner(MyPreprocessRunnable)
    model_a_runner = vtsserving.xgboost.get('model_a:latest').to_runner()
    model_b_runner = vtsserving.pytorch.get('model_b:latest').to_runner()

    svc = vtsserving.Service('inference_graph_demo', runners=[
        preprocess_runner,
        model_a_runner,
        model_b_runner
    ])

    @svc.api(input=Image(), output=Text())
    async def predict(input_image: PIL.Image.Image) -> str:
        model_input = await preprocess_runner.async_run(input_image)

        results = await asyncio.gather(
            model_a_runner.async_run(model_input),
            model_b_runner.async_run(model_input),
        )

        return post_process(
            results[0], # model a result
            results[1], # model b result
        )


Once each model completes, the results can be compared and logged as a post processing
step.


Runner Definition
-----------------

.. TODO::
    Document detailed list of Runner options

    .. code:: python

        my_runner = vtsserving.Runner(
            MyRunnable,
            runnable_init_params={"foo": foo, "bar": bar},
            name="custom_runner_name",
            strategy=None, # default strategy will be selected depending on the SUPPORTED_RESOURCES and SUPPORTS_CPU_MULTI_THREADING flag on runnable
            models=[..],

            # below are also configurable via config file:

            # default configs:
            max_batch_size=..  # default max batch size will be applied to all run methods, unless override in the runnable_method_configs
            max_latency_ms=.. # default max latency will be applied to all run methods, unless override in the runnable_method_configs

            runnable_method_configs=[
                {
                    method_name="predict",
                    max_batch_size=..,
                    max_latency_ms=..,
                }
            ],
        )

Runner Configuration
--------------------

Runner behaviors and resource allocation can be specified via VtsServing :ref:`configuration <guides/configuration:Configuration>`.
Runners can be both configured individually or in aggregate under the ``runners`` configuration key. To configure a specific runner, specify its name
under the ``runners`` configuration key. Otherwise, the configuration will be applied to all runners. The examples below demonstrate both
the configuration for all runners in aggregate and for an individual runner (``iris_clf``).

Adaptive Batching
^^^^^^^^^^^^^^^^^

If a model or custom runner supports batching, the :ref:`adaptive batching <guides/batching:Adaptive Batching>` mechanism is enabled by default.
To explicitly disable or control adaptive batching behaviors at runtime, configuration can be specified under the ``batching`` key.

.. tab-set::

    .. tab-item:: All Runners
        :sync: all_runners

        .. code-block:: yaml
	    :caption: ⚙️ `configuration.yml`

            runners:
                batching:
                    enabled: true
                    max_batch_size: 100
                    max_latency_ms: 500
    
    .. tab-item:: Individual Runner
        :sync: individual_runner
        
        .. code-block:: yaml
	    :caption: ⚙️ `configuration.yml`

            runners:
                iris_clf:
                    batching:
                        enabled: true
                        max_batch_size: 100
                        max_latency_ms: 500

Resource Allocation
^^^^^^^^^^^^^^^^^^^

By default, a runner will attempt to utilize all available resources in the container. Runner's resource allocation can also be customized
through configuration, with a `float` value for ``cpu`` and an `int` value for ``nvidia.com/gpu``. Fractional GPU is currently not supported.

.. tab-set::

    .. tab-item:: All Runners
        :sync: all_runners

        .. code-block:: yaml
	    :caption: ⚙️ `configuration.yml`

            runners:
                resources:
                    cpu: 0.5
                    nvidia.com/gpu: 1
    
    .. tab-item:: Individual Runner
        :sync: individual_runner
        
        .. code-block:: yaml
	    :caption: ⚙️ `configuration.yml`

            runners:
                iris_clf:
                    resources:
                        cpu: 0.5
                        nvidia.com/gpu: 1

Alternatively, a runner can be mapped to a specific set of GPUs. To specify GPU mapping, instead of defining an `integer` value, a list of device IDs
can be specified for the ``nvidia.com/gpu`` key. For example, the following configuration maps the configured runners to GPU device 2 and 4.

.. tab-set::

    .. tab-item:: All Runners
        :sync: all_runners

        .. code-block:: yaml
	    :caption: ⚙️ `configuration.yml`

            runners:
                resources:
                    nvidia.com/gpu: [2, 4]
    
    .. tab-item:: Individual Runner
        :sync: individual_runner
        
        .. code-block:: yaml
	    :caption: ⚙️ `configuration.yml`

            runners:
                iris_clf:
                    resources:
                        nvidia.com/gpu: [2, 4]

Timeout
^^^^^^^

Runner timeout defines the amount of time in seconds to wait before calls a runner is timed out on the API server.

.. tab-set::

    .. tab-item:: All Runners
        :sync: all_runners

        .. code-block:: yaml
	    :caption: ⚙️ `configuration.yml`

            runners:
                timeout: 60
    
    .. tab-item:: Individual Runner
        :sync: individual_runner
        
        .. code-block:: yaml
	    :caption: ⚙️ `configuration.yml`

            runners:
                iris_clf:
                    timeout: 60

Access Logging
^^^^^^^^^^^^^^

See :ref:`guides/logging:Logging Configuration` for access log customization.


Distributed Runner with Yatai
-----------------------------

`🦄️ Yatai <https://github.com/vtsserving/Yatai>`_ provides a more advanced Runner
architecture specifically designed for running large scale inference workloads on a
Kubernetes cluster.

While the standalone :code:`VtsServer` schedules Runner workers on their own Python
processes, the :code:`VtsDeployment` created by Yatai, scales Runner workers in their
own group of `Pods <https://kubernetes.io/docs/concepts/workloads/pods/>`_ and made it
possible to set a different resource requirement for each Runner, and auto-scaling each
Runner separately based on their workloads.


Sample :code:`VtsDeployment` definition file for deploying in Kubernetes:

.. code:: yaml

    apiVersion: yatai.vtsserving.org/v1beta1
    kind: VtsDeployment
    spec:
    vts_tag: 'fraud_detector:dpijemevl6nlhlg6'
    autoscaling:
        minReplicas: 3
        maxReplicas: 20
    resources:
        limits:
            cpu: 500m
        requests:
            cpu: 200m
    runners:
    - name: model_runner_a
        autoscaling:
            minReplicas: 1
            maxReplicas: 5
        resources:
            requests:
                nvidia.com/gpu: 1
                cpu: 2000m
            ...

.. TODO::
    add graph explaining Yatai Runner architecture
