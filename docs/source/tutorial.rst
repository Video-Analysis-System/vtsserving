==========================
Tutorial: Intro to VtsServing
==========================

*time expected: 10 minutes*

In this tutorial, we will focus on online model serving with VtsServing, using a
classification model trained with `scikit-learn <https://scikit-learn.org/stable/>`_ and the Iris dataset.
By the end of this tutorial, we will have a Vts that can be served easily using HTTP or gRPC for handling inference requests, and a docker
container image for deployment.


.. note::

    You might be tempted to skip this tutorial because you are not using scikit-learn,
    but give it a chance. The concepts you will learn in the tutorial are fundamental to
    model serving with any ML framework using VtsServing, and mastering it will give you a
    deep understanding of VtsServing.


Setup for the tutorial
----------------------

There are three ways to complete this tutorial:

#. Run with Google Colab in your browser

   👉 `Open Tutorial Notebook on Colab <https://colab.research.google.com/github/vtsserving/VtsServing/blob/main/examples/quickstart/iris_classifier.ipynb>`_
   side by side with this guide. As you go through this guide, you can simply run the
   sample code from the Colab Notebook.

   You will be able to try out most of the content in the tutorial on Colab besides
   the docker container part towards the end. This is because Google Colab currently
   does not support docker.

#. Run the tutorial notebook from Docker

   If you have Docker installed, you can run the tutorial notebook from a pre-configured
   docker image with:

   .. code-block:: bash

      » docker run -it --rm -p 8888:8888 -p 3000:3000 -p 3001:3001 vtsserving/quickstart:latest

#. Local Development Environment

   Download the source code of this tutorial from `vtsserving/examples <https://github.com/vtsserving/VtsServing/tree/main/examples>`_:

   .. code-block:: bash

      » git clone --depth=1 git@github.com:vtsserving/VtsServing.git
      » cd vtsserving/examples/quickstart/

   VtsServing supports Linux, Windows and MacOS. You will need Python 3.7 or above to run
   this tutorial. We recommend using `virtual environment <https://docs.python.org/3/library/venv.html>`_
   to create an isolated local environment. However this is not required.

   Install all dependencies required for this tutorial:

   .. code-block:: bash

      » pip install vtsserving scikit-learn pandas

.. note::

   VtsServing provides gRPC support, and we will provide gRPC examples alongside the HTTP
   ones in this tutorial. However, these examples are optional and you don't have to
   know about gRPC to get started with VtsServing.

   If you are interested in trying the gRPC examples in this tutorial, install
   the gRPC dependencies for VtsServing:

   .. code-block:: bash

      » pip install "vtsserving[grpc]"


Saving Models with VtsServing
--------------------------

To begin with VtsServing, you will need to save your trained models with VtsServing API in
its model store (a local directory managed by VtsServing). The model store is used for
managing all your trained models locally as well as accessing them for serving.

.. code-block:: python
   :emphasize-lines: 14,15

   import vtsserving

   from sklearn import svm
   from sklearn import datasets

   # Load training data set
   iris = datasets.load_iris()
   X, y = iris.data, iris.target

   # Train the model
   clf = svm.SVC(gamma='scale')
   clf.fit(X, y)

   # Save model to the VtsServing local model store
   saved_model = vtsserving.sklearn.save_model("iris_clf", clf)
   print(f"Model saved: {saved_model}")

   # Model saved: Model(tag="iris_clf:zy3dfgxzqkjrlgxi")


The model is now saved under the name ``iris_clf`` with an automatically generated
version. The name and version pair can then be used for retrieving the model. For
instance, the original model object can be loaded back into memory for testing via:

.. code-block:: python

   model = vtsserving.sklearn.load_model("iris_clf:2uo5fkgxj27exuqj")

   # Alternatively, use `latest` to find the newest version
   model = vtsserving.sklearn.load_model("iris_clf:latest")


The ``vtsserving.sklearn.save_model`` API is built specifically for the Scikit-Learn
framework and uses its native saved model format under the hood for best compatibility
and performance. This goes the same for other ML frameworks, e.g.
``vtsserving.pytorch.save_model``, see the :doc:`frameworks/index` to learn more.


.. seealso::

   It is possible to use pre-trained models directly with VtsServing or import existing
   trained model files to VtsServing. Learn more about it from :doc:`concepts/model`.


Saved models can be managed via the ``vtsserving models`` CLI command or Python API,
learn about it here: :ref:`concepts/model:Managing Models`.


Creating a Service
------------------

Services are the core components of VtsServing, where the serving logic is defined. Create
a file ``service.py`` with:

.. code-block:: python
   :caption: `service.py`

    import numpy as np
    import vtsserving
    from vtsserving.io import NumpyNdarray

    iris_clf_runner = vtsserving.sklearn.get("iris_clf:latest").to_runner()

    svc = vtsserving.Service("iris_classifier", runners=[iris_clf_runner])

    @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
    def classify(input_series: np.ndarray) -> np.ndarray:
        result = iris_clf_runner.predict.run(input_series)
        return result


We can now run the VtsServing server for our new service in development mode:

.. tab-set::

    .. tab-item:: HTTP
       :sync: http

       .. code-block:: bash

          » vtsserving serve service:svc --reload
          2022-09-18T21:11:22-0700 [INFO] [cli] Prometheus metrics for HTTP BentoServer from "service.py:svc" can be accessed at http://localhost:3000/metrics.
          2022-09-18T21:11:22-0700 [INFO] [cli] Starting development HTTP BentoServer from "service.py:svc" listening on 0.0.0.0:3000 (Press CTRL+C to quit)
          2022-09-18 21:11:23 circus[80177] [INFO] Loading the plugin...
          2022-09-18 21:11:23 circus[80177] [INFO] Endpoint: 'tcp://127.0.0.1:61825'
          2022-09-18 21:11:23 circus[80177] [INFO] Pub/sub: 'tcp://127.0.0.1:61826'
          2022-09-18T21:11:23-0700 [INFO] [observer] Watching directories: ['~/workspace/vtsserving/examples/quickstart', '~/vtsserving/models']

    .. tab-item:: gRPC
       :sync: grpc

       .. code-block:: bash

          » vtsserving serve-grpc service:svc --reload --enable-reflection
          2022-09-18T21:12:18-0700 [INFO] [cli] Prometheus metrics for gRPC BentoServer from "service.py:svc" can be accessed at http://localhost:3001.
          2022-09-18T21:12:18-0700 [INFO] [cli] Starting development gRPC BentoServer from "service.py:svc" listening on 0.0.0.0:3000 (Press CTRL+C to quit)
          2022-09-18 21:12:19 circus[81102] [INFO] Loading the plugin...
          2022-09-18 21:12:19 circus[81102] [INFO] Endpoint: 'tcp://127.0.0.1:61849'
          2022-09-18 21:12:19 circus[81102] [INFO] Pub/sub: 'tcp://127.0.0.1:61850'
          2022-09-18T21:12:19-0700 [INFO] [observer] Watching directories: ['~/workspace/vtsserving/examples/quickstart', '~/vtsserving/models']

Send prediction request to the service:

.. tab-set::

   .. tab-item:: HTTP
      :sync: http

      .. tab-set::

         .. tab-item:: Python
            :sync: python-client

            .. code-block:: python

               import requests

               requests.post(
                  "http://127.0.0.1:3000/classify",
                  headers={"content-type": "application/json"},
                  data="[[5.9, 3, 5.1, 1.8]]",
               ).text

         .. tab-item:: CURL
            :sync: curl-client

            .. code-block:: bash

               » curl -X POST \
                  -H "content-type: application/json" \
                  --data "[[5.9, 3, 5.1, 1.8]]" \
                  http://127.0.0.1:3000/classify

         .. tab-item:: Browser
            :sync: browser-client

            Open http://127.0.0.1:3000 in your browser and send test request from the web UI.

   .. tab-item:: gRPC
      :sync: grpc

      .. tab-set::

         .. tab-item:: Python
            :sync: python-client

            .. code-block:: python

               import grpc
               import numpy as np
               from vtsserving.grpc.utils import import_generated_stubs

               pb, services = import_generated_stubs()

               with grpc.insecure_channel("localhost:3000") as channel:
                  stub = services.BentoServiceStub(channel)

                  req: pb.Response = stub.Call(
                     request=pb.Request(
                           api_name="classify",
                           ndarray=pb.NDArray(
                              dtype=pb.NDArray.DTYPE_FLOAT,
                              shape=(1, 4),
                              float_values=[5.9, 3, 5.1, 1.8],
                           ),
                     )
                  )
                  print(req)

         .. tab-item:: grpcURL
            :sync: curl-client

            We will use `fullstorydev/grpcurl <https://github.com/fullstorydev/grpcurl>`_ to send a CURL-like request to the gRPC BentoServer.

            Note that we will use `docker <https://docs.docker.com/get-docker/>`_ to run the ``grpcurl`` command.

            .. tab-set::

               .. tab-item:: MacOS/Windows
                  :sync: macwin

                  .. code-block:: bash

                     » docker run -i --rm fullstorydev/grpcurl -d @ -plaintext host.docker.internal:3000 vtsserving.grpc.v1.BentoService/Call <<EOM
                     {
                        "apiName": "classify",
                        "ndarray": {
                           "shape": [1, 4],
                           "floatValues": [5.9, 3, 5.1, 1.8]
                        }
                     }
                     EOM

               .. tab-item:: Linux
                  :sync: Linux

                  .. code-block:: bash

                     » docker run -i --rm --network=host fullstorydev/grpcurl -d @ -plaintext 0.0.0.0:3000 vtsserving.grpc.v1.BentoService/Call <<EOM
                     {
                        "apiName": "classify",
                        "ndarray": {
                           "shape": [1, 4],
                           "floatValues": [5.9, 3, 5.1, 1.8]
                        }
                     }
                     EOM

         .. tab-item:: Browser
            :sync: browser-client

            We will use `fullstorydev/grpcui <https://github.com/fullstorydev/grpcui>`_ to send request from a web browser.

            Note that we will use `docker <https://docs.docker.com/get-docker/>`_ to run the ``grpcui`` command.

            .. tab-set::

               .. tab-item:: MacOS/Windows
                  :sync: macwin

                  .. code-block:: bash

                     » docker run --init --rm -p 8080:8080 fullstorydev/grpcui -plaintext host.docker.internal:3000

               .. tab-item:: Linux
                  :sync: Linux

                  .. code-block:: bash

                     » docker run --init --rm -p 8080:8080 --network=host fullstorydev/grpcui -plaintext 0.0.0.0:3000


            Proceed to http://127.0.0.1:8080 in your browser and send test request from the web UI.


Using Models in a Service
~~~~~~~~~~~~~~~~~~~~~~~~~

In this example, ``vtsserving.sklearn.get`` creates a reference to the saved model
in the model store, and ``to_runner`` creates a Runner instance from the model.
The Runner abstraction gives BentoServer more flexibility in terms of how to schedule
the inference computation, how to dynamically batch inference calls and better take
advantage of all hardware resource available.

You can test out the Runner interface this way:

.. code-block:: python

   import vtsserving

   iris_clf_runner = vtsserving.sklearn.get("iris_clf:latest").to_runner()
   iris_clf_runner.init_local()
   iris_clf_runner.predict.run([[5.9, 3., 5.1, 1.8]])

.. note::

   For custom Runners and advanced runner options, see :doc:`concepts/runner` and :doc:`guides/batching`.


Service API and IO Descriptor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``svc.api`` decorator adds a function to the ``vtsserving.Service`` object's
APIs list. The ``input`` and ``output`` parameter takes an
:doc:`IO Descriptor <reference/api_io_descriptors>` object, which specifies the API
function's expected input/output types, and is used for generating HTTP endpoints.

In this example, both ``input`` and ``output`` are defined with
:ref:`vtsserving.io.NumpyNdarray <reference/api_io_descriptors:NumPy \`\`ndarray\`\`>`, which means
the API function being decorated, takes a ``numpy.ndarray`` as input, and returns a
``numpy.ndarray`` as output.

.. note::

   More options, such as ``pandas.DataFrame``, ``JSON``, and ``PIL.Image``
   are also supported. An IO Descriptor object can also be configured with a schema or
   a shape for input/output validation. Learn more about them in
   :doc:`reference/api_io_descriptors`.

Inside the API function, users can define any business logic, feature fetching, and
feature transformation code. Model inference calls are made directly through runner
objects, that are passed into ``vtsserving.Service(name=.., runners=[..])`` call when
creating the service object.

.. tip::

   VtsServing supports both :ref:`sync and async endpoints <concepts/service:Sync vs Async APIs>`.
   For performance sensitive use cases, especially when working with IO-intense
   workloads (e.g. fetching features from multiple sources) or when
   :ref:`composing multiple models <concepts/runner:Serving Multiple Models via Runner>` , you may consider defining an
   ``async`` API instead.

   Here's an example of the same endpoint above defined with ``async``:

   .. code-block:: python

      @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
      async def classify(input_series: np.ndarray) -> np.ndarray:
         result = await iris_clf_runner.predict.async_run(input_series)
         return result


Building a Vts 🍱
-------------------

Once the service definition is finalized, we can build the model and service into a
``vts``. Vts is the distribution format for a service. It is a self-contained
archive that contains all the source code, model files and dependency specifications
required to run the service.

To build a Vts, first create a ``vtsfile.yaml`` file in your project directory:

.. tab-set::

    .. tab-item:: HTTP
       :sync: http

       .. code-block:: yaml

          service: "service:svc"  # Same as the argument passed to `vtsserving serve`
          labels:
             owner: vtsserving-team
             stage: dev
          include:
          - "*.py"  # A pattern for matching which files to include in the vts
          python:
             packages:  # Additional pip packages required by the service
             - scikit-learn
             - pandas

    .. tab-item:: gRPC
       :sync: grpc

       .. code-block:: yaml

          service: "service:svc"  # Same as the argument passed to `vtsserving serve`
          labels:
             owner: vtsserving-team
             stage: dev
          include:
          - "*.py"  # A pattern for matching which files to include in the vts
          python:
             packages:  # Additional pip packages required by the service
             - vtsserving[grpc]
             - scikit-learn
             - pandas

.. tip::

   VtsServing provides lots of build options in ``vtsfile.yaml`` for customizing the
   Python dependencies, cuda installation, docker image distro, etc. Read more about it
   on the :doc:`concepts/vts` page.


Next, run the ``vtsserving build`` CLI command from the same directory:

.. code-block:: bash

    » vtsserving build

    Building VtsServing service "iris_classifier:6otbsmxzq6lwbgxi" from build context "/home/user/gallery/quickstart"
    Packing model "iris_clf:zy3dfgxzqkjrlgxi"
    Locking PyPI package versions..

    Successfully built Vts(tag="iris_classifier:6otbsmxzq6lwbgxi")

🎉 You've just created your first Vts, and it is now ready for serving in production!
For starters, you can now serve it with the ``vtsserving serve`` CLI command:

.. tab-set::

    .. tab-item:: HTTP
       :sync: http

       .. code-block:: bash

          » vtsserving serve iris_classifier:latest --production

          2022-09-18T21:22:17-0700 [INFO] [cli] Environ for worker 0: set CPU thread count to 10
          2022-09-18T21:22:17-0700 [INFO] [cli] Prometheus metrics for HTTP BentoServer from "iris_classifier:latest" can be accessed at http://0.0.0.0:3000/metrics.
          2022-09-18T21:22:18-0700 [INFO] [cli] Starting production HTTP BentoServer from "iris_classifier:latest" running on http://0.0.0.0:3000 (Press CTRL+C to quit)

    .. tab-item:: gRPC
       :sync: grpc

       .. code-block:: bash

          » vtsserving serve-grpc iris_classifier:latest --production

          2022-09-18T21:23:11-0700 [INFO] [cli] Environ for worker 0: set CPU thread count to 10
          2022-09-18T21:23:11-0700 [INFO] [cli] Prometheus metrics for gRPC BentoServer from "iris_classifier:latest" can be accessed at http://0.0.0.0:3001.
          2022-09-18T21:23:11-0700 [INFO] [cli] Starting production gRPC BentoServer from "iris_classifier:latest" running on http://0.0.0.0:3000 (Press CTRL+C to quit)

.. note::

   The build process resolves ``iris_clf:latest`` and packages the latest version of the ``iris_clf`` model in the model store to ensure the same version of the model gets deployed every time.


Vts is the unit of deployment in VtsServing, one of the most important artifacts to keep
track of in your model deployment workflow. VtsServing provides CLI commands and APIs for
managing Bentos and moving them around, see the :ref:`concepts/vts:Managing Bentos`
section to learn more.


Generate Docker Image
---------------------

A docker image can be automatically generated from a Vts for production deployment,
via the ``vtsserving containerize`` CLI command:

.. tab-set::

    .. tab-item:: HTTP
       :sync: http

       .. code-block:: bash

          » vtsserving containerize iris_classifier:latest

          Building docker image for Vts(tag="iris_classifier:6otbsmxzq6lwbgxi")...
          Successfully built docker image for "iris_classifier:6otbsmxzq6lwbgxi" with tags "iris_classifier:6otbsmxzq6lwbgxi"
          To run your newly built Vts container, pass "iris_classifier:6otbsmxzq6lwbgxi" to "docker run". For example: "docker run -it --rm -p 3000:3000 iris_classifier:6otbsmxzq6lwbgxi serve --production".

    .. tab-item:: gRPC
       :sync: grpc

       .. code-block:: bash

          » vtsserving containerize iris_classifier:latest --enable-features grpc

          Building docker image for Vts(tag="iris_classifier:6otbsmxzq6lwbgxi")...
          Successfully built docker image for "iris_classifier:6otbsmxzq6lwbgxi" with tags "iris_classifier:6otbsmxzq6lwbgxi"
          To run your newly built Vts container, pass "iris_classifier:6otbsmxzq6lwbgxi" to "docker run". For example: "docker run -it --rm -p 3000:3000 iris_classifier:6otbsmxzq6lwbgxi serve --production".
          Additionally, to run your Vts container as a gRPC server, do: "docker run -it --rm -p 3000:3000 -p 3001:3001 iris_classifier:6otbsmxzq6lwbgxi serve-grpc --production"

.. note::

   You will need to `install Docker <https://docs.docker.com/get-docker/>`_ before
   running this command.

.. dropdown:: For Mac with Apple Silicon
   :icon: cpu

   Specify the ``--platform`` to avoid potential compatibility issues with some
   Python libraries.

   .. code-block:: bash

      » vtsserving containerize --platform=linux/amd64 iris_classifier:latest

This creates a docker image that includes the Vts, and has all its dependencies
installed. The docker image tag will be same as the Vts tag by default:

.. code-block:: bash

   » docker images

   REPOSITORY         TAG                 IMAGE ID        CREATED          SIZE
   iris_classifier    6otbsmxzq6lwbgxi    0b4f5ec01941    10 seconds ago   1.06GB


Run the docker image to start the BentoServer:

.. tab-set::

    .. tab-item:: HTTP
       :sync: http

       .. code-block:: bash

          » docker run -it --rm -p 3000:3000 iris_classifier:6otbsmxzq6lwbgxi serve --production

          2022-09-19T05:27:31+0000 [INFO] [cli] Service loaded from Vts directory: vtsserving.Service(tag="iris_classifier:6otbsmxzq6lwbgxi", path="/home/vtsserving/vts/")
          2022-09-19T05:27:31+0000 [WARNING] [cli] GPU not detected. Unable to initialize pynvml lib.
          2022-09-19T05:27:31+0000 [INFO] [cli] Environ for worker 0: set CPU thread count to 4
          2022-09-19T05:27:31+0000 [INFO] [cli] Prometheus metrics for HTTP BentoServer from "/home/vtsserving/vts" can be accessed at http://0.0.0.0:3000/metrics.
          2022-09-19T05:27:32+0000 [INFO] [cli] Starting production HTTP BentoServer from "/home/vtsserving/vts" running on http://0.0.0.0:3000 (Press CTRL+C to quit)
          2022-09-19T05:27:32+0000 [INFO] [api_server:2] Service loaded from Vts directory: vtsserving.Service(tag="iris_classifier:6otbsmxzq6lwbgxi", path="/home/vtsserving/vts/")
          2022-09-19T05:27:32+0000 [INFO] [api_server:1] Service loaded from Vts directory: vtsserving.Service(tag="iris_classifier:6otbsmxzq6lwbgxi", path="/home/vtsserving/vts/")
          2022-09-19T05:27:32+0000 [INFO] [runner:iris_clf:1] Service loaded from Vts directory: vtsserving.Service(tag="iris_classifier:6otbsmxzq6lwbgxi", path="/home/vtsserving/vts/")
          2022-09-19T05:27:32+0000 [INFO] [api_server:3] Service loaded from Vts directory: vtsserving.Service(tag="iris_classifier:6otbsmxzq6lwbgxi", path="/home/vtsserving/vts/")
          2022-09-19T05:27:32+0000 [INFO] [api_server:4] Service loaded from Vts directory: vtsserving.Service(tag="iris_classifier:6otbsmxzq6lwbgxi", path="/home/vtsserving/vts/")

    .. tab-item:: gRPC
       :sync: grpc

       .. code-block:: bash

          » docker run -it --rm -p 3000:3000 -p 3001:3001 iris_classifier:6otbsmxzq6lwbgxi serve-grpc --production

          2022-09-19T05:28:29+0000 [INFO] [cli] Service loaded from Vts directory: vtsserving.Service(tag="iris_classifier:6otbsmxzq6lwbgxi", path="/home/vtsserving/vts/")
          2022-09-19T05:28:29+0000 [WARNING] [cli] GPU not detected. Unable to initialize pynvml lib.
          2022-09-19T05:28:29+0000 [INFO] [cli] Environ for worker 0: set CPU thread count to 4
          2022-09-19T05:28:29+0000 [INFO] [cli] Prometheus metrics for gRPC BentoServer from "/home/vtsserving/vts" can be accessed at http://0.0.0.0:3001.
          2022-09-19T05:28:30+0000 [INFO] [cli] Starting production gRPC BentoServer from "/home/vtsserving/vts" running on http://0.0.0.0:3000 (Press CTRL+C to quit)
          2022-09-19T05:28:30+0000 [INFO] [grpc_api_server:2] Service loaded from Vts directory: vtsserving.Service(tag="iris_classifier:6otbsmxzq6lwbgxi", path="/home/vtsserving/vts/")
          2022-09-19T05:28:30+0000 [INFO] [grpc_api_server:4] Service loaded from Vts directory: vtsserving.Service(tag="iris_classifier:6otbsmxzq6lwbgxi", path="/home/vtsserving/vts/")
          2022-09-19T05:28:30+0000 [INFO] [grpc_api_server:3] Service loaded from Vts directory: vtsserving.Service(tag="iris_classifier:6otbsmxzq6lwbgxi", path="/home/vtsserving/vts/")
          2022-09-19T05:28:30+0000 [INFO] [grpc_api_server:1] Service loaded from Vts directory: vtsserving.Service(tag="iris_classifier:6otbsmxzq6lwbgxi", path="/home/vtsserving/vts/")
          2022-09-19T05:28:30+0000 [INFO] [runner:iris_clf:1] Service loaded from Vts directory: vtsserving.Service(tag="iris_classifier:6otbsmxzq6lwbgxi", path="/home/vtsserving/vts/")


Most of the deployment tools built on top of VtsServing use Docker under the hood. It is
recommended to test out serving from a containerized Vts docker image first, before
moving to a production deployment. This helps verify the correctness of all the docker
and dependency configs specified in the ``vtsfile.yaml``.


Deploying Bentos
----------------

VtsServing standardizes the saved model format, service API definition and the Vts build
process, which opens up many different deployment options for ML teams.

The Vts we built and the docker image created in the previous steps are designed to
be DevOps friendly and ready for deployment in a production environment. If your team
has existing infrastructure for running docker, it's likely that the Vts generated
docker images can be directly deployed to your infrastructure without any modification.

.. note::

   To streamline the deployment process, BentoServer follows most common best practices
   found in a backend service: it provides
   :doc:`health check and prometheus metrics <guides/monitoring>`
   endpoints for monitoring out-of-the-box; It provides configurable
   :doc:`distributed tracing <guides/tracing>` and :doc:`logging <guides/logging>` for
   performance analysis and debugging; and it can be easily
   :doc:`integrated with other tools <integrations/index>` that are commonly used by
   Data Engineers and DevOps engineers.


For teams looking for an end-to-end solution, with more powerful deployment features
specific for ML, the VtsServing team has also created Yatai and vtsctl:

.. grid::  1 2 2 2
    :gutter: 3
    :margin: 0
    :padding: 0

    .. grid-item-card:: `🦄️ Yatai <https://github.com/vtsserving/Yatai>`_
        :link: https://github.com/vtsserving/Yatai
        :link-type: url

        Model Deployment at scale on Kubernetes.

    .. grid-item-card:: `🚀 vtsctl <https://github.com/vtsserving/vtsctl>`_
        :link: https://github.com/vtsserving/vtsctl
        :link-type: url

        Fast model deployment on any cloud platform.

Learn more about different deployment options with VtsServing from the
:doc:`concepts/deploy` page.


----

.. button-ref:: concepts/index
   :ref-type: doc
   :color: secondary
   :expand:

   Continue Reading
