===================
1.0 Migration Guide
===================

VtsServing version 1.0.0 APIs are backward incompatible with version 0.13.1. However, most of the common
functionality can be achieved with the new version. We will guide and demonstrate the migration by
transforming the `quickstart <https://github.com/vtsserving/VtsServing/tree/main/examples/quickstart>`_ gallery project
from VtsServing version 0.13.1 to 1.0.0. Complete every migration action denoted like the section below.

.. admonition:: 💡 Migration Task

   Install VtsServing version 1.0.0 by running the following command.


.. code-block:: bash

    > pip install vtsserving


Train Models
------------

First, the quickstart project begins by training a classifier Scikit-Learn model from the iris datasets.
By running :code:`python train.py`, we obtain a trained classifier model.

.. code-block:: python
    :caption: train.py

    from sklearn import svm
    from sklearn import datasets

    # Load training data
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # Model Training
    clf = svm.SVC(gamma='scale')
    clf.fit(X, y)

VtsServing version 1.0.0 introduces the model store concept to help improve model management during development.
Once we are happy with the model trained, we can save the  model instance with the :code:`save_model()`
framework API to persist it in the model store. Optionally, you may attach custom labels, metadata, or custom
objects like tokenizers to be saved alongside the model. See
:ref:`Save A Trained Model <concepts/model:Save A Trained Model>` to learn more.

.. admonition:: 💡 Migration Task

   Append the model saving logic below to `train.py` and run `python train.py`.

.. code-block:: python

    vtsserving.sklearn.save_model("iris_clf", clf)
    print(f"Model saved: {saved_model}")

You can view and manage all saved models via the :code:`vtsserving models` CLI command.

.. code-block:: bash

    > vtsserving models list

    Tag                        Module           Size        Creation Time        Path
    iris_clf:zy3dfgxzqkjrlgxi  vtsserving.sklearn  5.81 KiB    2022-05-19 08:36:52  ~/vtsserving/models/iris_clf/zy3dfgxzqkjrlgxi

Define Services
---------------

Next, we will transform the service definition module and breakdown each section into details.

.. admonition:: 💡 Migration Task

   Update the service definition module `service.py` from the VtsServing 0.13.1 specification to 1.0.0 specification.

.. tab-set::

    .. tab-item:: 0.13.1

        .. code-block:: python
            :caption: service.py

            import pandas as pd

            from vtsserving import env, artifacts, api, VtsService
            from vtsserving.adapters import DataframeInput
            from vtsserving.frameworks.sklearn import SklearnModelArtifact

            @env(infer_pip_packages=True)
            @artifacts([SklearnModelArtifact('model')])
            class IrisClassifier(VtsService):
                @api(input=DataframeInput(), batch=True)
                def predict(self, df: pd.DataFrame):
                    return self.artifacts.model.predict(df)

    .. tab-item:: 1.0.0

        .. code-block:: python
            :caption: service.py

            import numpy as np
            import pandas as pd

            import vtsserving
            from vtsserving.io import NumpyNdarray, PandasDataFrame

            iris_clf_runner = vtsserving.sklearn.get("iris_clf:latest").to_runner()

            svc = vtsserving.Service("iris_classifier", runners=[iris_clf_runner])

            @svc.api(input=PandasDataFrame(), output=NumpyNdarray())
            def predict(input_series: pd.DataFrame) -> np.ndarray:
                result = iris_clf_runner.predict.run(input_series)
                return result

Environment
~~~~~~~~~~~

VtsServing version 0.13.1 relies on the :code:`@env`
`decorator API <https://docs.vtsserving.org/en/0.13-lts/concepts.html#defining-service-environment>`_ for defining the
environment settings and dependencies of the service. Typical arguments of the environment decorator includes Python
dependencies (e.g. :code:`pip_packages`, :code:`pip_index_url`), Conda dependencies (e.g. :code:`conda_channels`,
:code:`conda_dependencies`), and Docker options (e.g. :code:`setup_sh`, :code:`docker_base_image`).

.. code-block:: python

    @env(pip_packages=["scikit-learn", "pandas"])

VtsServing version 1.0.0 no longer relies on the environment decorator. Environment settings and service dependencies are
defined in the :code:`vtsfile.yaml` file in the project directory. The contents are used to specify the
:code:`vtsserving build` opations when :ref:`building vtss <concepts/vts:Vts Build Options>`.

.. admonition:: 💡 Migration Task

   Save the contents below to the `vtsfile.yaml` file in the same directory as `service.py`.

.. code-block:: yaml

    service: "service.py:svc"
    labels:
        owner: vtsserving-team
        project: gallery
    include:
        - "*.py"
    python:
        packages:
        - scikit-learn
        - pandas

Artifacts
~~~~~~~~~

VtsServing version 0.13.1 provides the :code:`@artifacts`
`decorator API <https://docs.vtsserving.org/en/0.13-lts/concepts.html#packaging-model-artifacts>`_ for users to specify
the trained models required by a VtsService. The specified artifacts are automatically serialized and deserialized
when saving and loading a VtsService.

.. code-block:: python

    @artifacts([SklearnModelArtifact('model')])

VtsServing 1.0.0 leverages a combination of :ref:`model store <concepts/model:Managing Models>` and
:ref:`runners <concepts/runner:What is Runner?>` APIs for specifying the required models at runtime. Methods on the
model can be invoked by calling the run function on the runner. Runner represents a unit of computation that can be
executed on a remote Python worker and scales independently.

.. code-block:: python

    iris_clf_runner = vtsserving.sklearn.get("iris_clf:latest").to_runner()

API
~~~

VtsServing version 0.13.1 defines the inference API through the :code:`@api`
`decorator <https://docs.vtsserving.org/en/0.13-lts/concepts.html#api-function-and-adapters>`_.
Input and output types can be specified through the adapters. The service will convert the inference request from
HTTP to the desired format specified by the input adaptor, in this case, a :code:`pandas.DataFrame` object.

.. code-block:: python

    @api(input=DataframeInput(), batch=True)
    def predict(self, df: pd.DataFrame):
        return self.artifacts.model.predict(df)

VtsServing version 1.0.0 also provides a similar :code:`@svc.api` :ref:`decorator <concepts/service:Service APIs>`.
The inference API is no longer defined within the service class. The association with the service is declared with the
:code:`@svc.api` decorator from the :code:`vtsserving.Service` class. Input and output specifications are defined by IO
descriptor arguments passed to the :code:`@src.api` decorator. Similar to the adaptors, they help describe the expected
data types, validate that the input and output conform to the expected format and schema, and convert them from and to
the specified native types. In addition, multiple input and output can be defined using the tuple syntax,
e.g. :code:`input=(image=Image(), metadata=JSON())`.

.. code-block:: python

    @svc.api(input=PandasDataFrame(), output=NumpyNdarray())
    def predict(input_series: pd.DataFrame) -> np.ndarray:
        result = iris_clf_runner.predict.run(input_series)
        return result

VtsServing version 1.0.0 supports defining inference API as an asynchronous coroutine. Asynchronous APIs are preferred if
the processing logic is IO-bound or invokes multiple runners simultaneously which is ideal for fetching features and
calling remote APIs.

Test Services
~~~~~~~~~~~~~

To improve development agility, VtsServing version 1.0.0 adds the capability to test the service in development before
saving. Executing the :code:`vtsserving serve` command will bring up an API server for rapid development iterations. The
:code:`--reload` option allows the development API server to reload upon every change of the service module.

.. code-block:: bash

    > vtsserving serve --reload

To bring up the API server and runners in a production like setting, use the :code:`--production` option. In production
mode, API servers and runners will run in separate processes to maximize server utility and parallelism.

.. code-block:: bash

    > vtsserving serve --production


Building Vtss
---------------

Next, we will build the service into a vts and save it to the vts store. Building a service to vts is to persist
the service for distribution. This operation is unique to VtsServing version 1.0.0. The comparable operation in version
0.13.1 is to save a service to disk by calling the :code:`save()` function on the service instance.

.. admonition:: 💡 Migration Task

   Run :code:`vtsserving build` command from the same directory as `service.py` and `vtsfile.yaml`.

.. tab-set::

    .. tab-item:: 0.13.1

        .. code-block:: python
            :caption: packer.py

            # import the IrisClassifier class defined above
            from vts_service import IrisClassifier

            # Create a iris classifier service instance
            iris_classifier_service = IrisClassifier()

            # Pack the newly trained model artifact
            from sklearn import svm
            from sklearn import datasets

            # Load training data
            iris = datasets.load_iris()
            X, y = iris.data, iris.target

            # Model Training
            clf = svm.SVC(gamma='scale')
            clf.fit(X, y)

            iris_classifier_service.pack('model', clf)

            # Save the prediction service to disk for model serving
            saved_path = iris_classifier_service.save()

    .. tab-item:: 1.0.0

        .. code-block:: bash

            > vtsserving build

            Building VtsServing service "iris_classifier:6otbsmxzq6lwbgxi" from build context "/home/user/gallery/quickstart"
            Packing model "iris_clf:zy3dfgxzqkjrlgxi"
            Locking PyPI package versions..

            ██████╗░███████╗███╗░░██╗████████╗░█████╗░███╗░░░███╗██╗░░░░░
            ██╔══██╗██╔════╝████╗░██║╚══██╔══╝██╔══██╗████╗░████║██║░░░░░
            ██████╦╝█████╗░░██╔██╗██║░░░██║░░░██║░░██║██╔████╔██║██║░░░░░
            ██╔══██╗██╔══╝░░██║╚████║░░░██║░░░██║░░██║██║╚██╔╝██║██║░░░░░
            ██████╦╝███████╗██║░╚███║░░░██║░░░╚█████╔╝██║░╚═╝░██║███████╗
            ╚═════╝░╚══════╝╚═╝░░╚══╝░░░╚═╝░░░░╚════╝░╚═╝░░░░░╚═╝╚══════╝

            Successfully built Vts(tag="iris_classifier:6otbsmxzq6lwbgxi")

You can view and manage all saved models via the :code:`vtsserving` CLI command.

.. code-block:: bash

    > vtsserving list

    Tag                               Size        Creation Time        Path
    iris_classifier:6otbsmxzq6lwbgxi  16.48 KiB   2022-07-01 16:03:44  ~/vtsserving/vtss/iris_classifier/6otbsmxzq6lwbgxi


Serve Vtss
~~~~~~~~~~~~

We can serve the saved vtss by running the :code:`vtsserving serve` command. We can add :code:`--production` to have
API servers and runners will run in separate processes to maximize server utility and parallelism.

.. code-block:: bash

    > vtsserving serve iris_classifier:latest --production

    2022-07-06T02:02:30-0700 [INFO] [] Starting production VtsServer from "." running on http://0.0.0.0:3000 (Press CTRL+C to quit)
    2022-07-06T02:02:31-0700 [INFO] [runner-iris_clf:1] Setting up worker: set CPU thread count to 10

Generate Docker Images
----------------------

Similar to version 0.13.1, we can generate docker images from vtss using the :code:`vtsserving containerize` command in VtsServing
version 1.0.0, see :ref:`Containerize Vtss <concepts/deploy:Containerize Vtss>` to learn more.

.. code-block:: bash

    > vtsserving containerize iris_classifier:latest

    Building docker image for Vts(tag="iris_classifier:6otbsmxzq6lwbgxi")...
    Successfully built docker image "iris_classifier:6otbsmxzq6lwbgxi"

You can run the docker image to start the service.

.. code-block:: bash

    > docker run -p 3000:3000 iris_classifier:6otbsmxzq6lwbgxi

    2022-07-01T21:57:47+0000 [INFO] [] Service loaded from Vts directory: vtsserving.Service(tag="iris_classifier:6otbsmxzq6lwbgxi", path="/home/vtsserving/vts/")
    2022-07-01T21:57:47+0000 [INFO] [] Starting production VtsServer from "/home/vtsserving/vts" running on http://0.0.0.0:3000 (Press CTRL+C to quit)
    2022-07-01T21:57:48+0000 [INFO] [api_server:1] Service loaded from Vts directory: vtsserving.Service(tag="iris_classifier:6otbsmxzq6lwbgxi", path="/home/vtsserving/vts/")
    2022-07-01T21:57:48+0000 [INFO] [runner-iris_clf:1] Service loaded from Vts directory: vtsserving.Service(tag="iris_classifier:6otbsmxzq6lwbgxi", path="/home/vtsserving/vts/")
    2022-07-01T21:57:48+0000 [INFO] [api_server:2] Service loaded from Vts directory: vtsserving.Service(tag="iris_classifier:6otbsmxzq6lwbgxi", path="/home/vtsserving/vts/")
    2022-07-01T21:57:48+0000 [INFO] [runner-iris_clf:1] Setting up worker: set CPU thread count to 4
    2022-07-01T21:57:48+0000 [INFO] [api_server:3] Service loaded from Vts directory: vtsserving.Service(tag="iris_classifier:6otbsmxzq6lwbgxi", path="/home/vtsserving/vts/")
    2022-07-01T21:57:48+0000 [INFO] [api_server:4] Service loaded from Vts directory: vtsserving.Service(tag="iris_classifier:6otbsmxzq6lwbgxi", path="/home/vtsserving/vts/")

Deploy Vtss
-------------

VtsServing version 0.13.1 supports deployment of Vtss to various cloud providers, including Google Cloud Platform, Amazon Web Services,
and Microsoft Azure. To better support the devops workflows, cloud deployment of Vtss has been moved to a separate project,
`🚀 vtsctl <https://github.com/vtsserving/vtsctl>`_, to better focus on the deployment tasks. :code:`vtsctl` is a CLI tool for
deploying your machine-learning models to any cloud platforms.

Manage Vtss
-------------

VtsServing version 0.13.1 relies on Yatai as a vts registry to help teams collaborate and manage vtss. In addition to vts management,
`🦄️ Yatai <https://github.com/vtsserving/Yatai>`_ project has since been expanded into a platform for deploying large scale model
serving workloads on Kubernetes. Yatai standardizes VtsServing deployment and provides UI for managing all your ML models and deployments
in one place, and enables advanced GitOps and CI/CD workflow.


🎉 Ta-da, you have migrated your project to VtsServing 1.0.0. Have more questions?
`Join the VtsServing Slack community <https://l.linklyhq.com/l/ktPp>`_.
