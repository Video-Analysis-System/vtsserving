===============
Deploying Vts
===============


Deployment Overview
-------------------

The three most common deployment options with VtsServing are:

- 🐳 Generate container images from Vts for custom docker deployment
- `🦄️ Yatai <https://github.com/vtsserving/Yatai>`_: Model Deployment at scale on Kubernetes
- `🚀 vtsctl <https://github.com/vtsserving/vtsctl>`_: Fast model deployment on any cloud platform


Containerize Vtss
-------------------

Containerizing vtss as Docker images allows users to easily distribute and deploy
vtss. Once services are built as vtss and saved to the vts store, we can
containerize saved vtss with the CLI command `vtsserving containerize`.

Start the Docker engine. Verify using `docker info`.

.. code:: bash

    > docker info

Run :code:`vtsserving list` to view available vtss in the store.

.. code:: bash

    > vtsserving list

    Tag                               Size        Creation Time        Path
    iris_classifier:ejwnswg5kw6qnuqj  803.01 KiB  2022-05-27 00:37:08  ~/vtsserving/vtss/iris_classifier/ejwnswg5kw6qnuqj
    iris_classifier:h4g6jmw5kc4ixuqj  644.45 KiB  2022-05-27 00:02:08  ~/vtsserving/vtss/iris_classifier/h4g6jmw5kc4ixuqj


Run :code:`vtsserving containerize` to start the containerization process.

.. code:: bash

    > vtsserving containerize iris_classifier:latest                                                                                                                                             02:10:47

    INFO [cli] Building docker image for Vts(tag="iris_classifier:ejwnswg5kw6qnuqj")...
    [+] Building 21.2s (20/20) FINISHED
    ...
    INFO [cli] Successfully built docker image "iris_classifier:ejwnswg5kw6qnuqj"


.. dropdown:: For Mac with Apple Silicon
   :icon: cpu

   Specify the :code:`--platform` to avoid potential compatibility issues with some
   Python libraries.

   .. code:: bash

      vtsserving containerize --platform=linux/amd64 iris_classifier:latest


View the built Docker image:

.. code:: bash

    > docker images

    REPOSITORY          TAG                 IMAGE ID       CREATED         SIZE
    iris_classifier     ejwnswg5kw6qnuqj    669e3ce35013   1 minutes ago   1.12GB

Run the generated docker image:

.. code:: bash

    > docker run -p 3000:3000 iris_classifier:ejwnswg5kw6qnuqj

.. todo::

    - Add sample code for working with GPU and --gpu flag
    - Add a further reading section
    - Explain buildx requirement
    - Explain multi-platform build


Deploy with Yatai
-----------------

Yatai helps ML teams to deploy large scale model serving workloads on Kubernetes. It
standardizes VtsServing deployment on Kubernetes, provides UI and APis for managing all
your ML models and deployments in one place, and enables advanced GitOps and CI/CD
workflows.

Yatai is Kubernetes native, integrates well with other cloud native tools in the K8s
eco-system.

To get started, get an API token from Yatai Web UI and login from your :code:`vtsserving`
CLI command:

.. code:: bash

    vtsserving yatai login --api-token {YOUR_TOKEN_GOES_HERE} --endpoint http://yatai.127.0.0.1.sslip.io

Push your local Vtss to yatai:

.. code:: python

    vtsserving push iris_classifier:latest

.. tip::
    Yatai will automatically start building container images for a new Vts pushed.


Deploy via Web UI
^^^^^^^^^^^^^^^^^

Although not always recommended for production workloads, Yatai offers an easy-to-use
web UI for quickly creating deployments. This is convenient for data scientists to test
out Vts deployments end-to-end from a development or testing environment:

.. image:: /_static/img/yatai-deployment-creation.png
    :alt: Yatai Deployment creation UI

The web UI is also very helpful for viewing system status, monitoring services, and
debugging issues.

.. image:: /_static/img/yatai-deployment-details.png
    :alt: Yatai Deployment Details UI

Commonly we recommend using APIs or Kubernetes CRD objects to automate the deployment
pipeline for production workloads.

Deploy via API
^^^^^^^^^^^^^^

Yatai's REST API specification can be found under the :code:`/swagger` endpoint. If you
have Yatai deployed locally with minikube, visit:
http://yatai.127.0.0.1.sslip.io/swagger/. The Swagger API spec covers all core Yatai
functionalities ranging from model/vts management, cluster management to deployment
automation.

.. note::

    Python APIs for creating deployment on Yatai is on our roadmap. See :issue:`2405`.
    Current proposal looks like this:

    .. code:: python

        yatai_client = vtsserving.YataiClient.from_env()

        vts = yatai_client.get_vts('my_svc:v1')
        assert vts and vts.status.is_ready()

        yatai_client.create_deployment('my_deployment', vts.tag, ...)

        # For updating a deployment:
        yatai_client.update_deployment('my_deployment', vts.tag)

        # check deployment_info.status
        deployment_info = yatai_client.get_deployment('my_deployment')


Deploy via kubectl and CRD
^^^^^^^^^^^^^^^^^^^^^^^^^^

For DevOps managing production model serving workloads along with other kubernetes
resources, the best option is to use :code:`kubectl` and directly create
:code:`VtsDeployment` objects in the cluster, which will be handled by the Yatai
deployment CRD controller.

.. code:: yaml

    # my_deployment.yaml
    apiVersion: serving.yatai.ai/v1alpha2
    kind: VtsDeployment
    metadata:
      name: demo
    spec:
      vts_tag: iris_classifier:3oevmqfvnkvwvuqj
      resources:
        limits:
          cpu: 1000m
        requests:
          cpu: 500m

.. code:: bash

    kubectl apply -f my_deployment.yaml



Deploy with vtsctl
--------------------

:code:`vtsctl` is a CLI tool for deploying Vtss to run on any cloud platform. It
supports all major cloud providers, including AWS, Azure, Google Cloud, and many more.

Underneath, :code:`vtsctl` is powered by Terraform. :code:`vtsctl` adds required
modifications to Vts or service configurations, and then generate terraform templates
for the target deploy platform for easy deployment.

The :code:`vtsctl` deployment workflow is optimized for CI/CD and GitOps. It is highly
customizable, users can fine-tune all configurations provided by the cloud platform. It
is also extensible, for users to define additional terraform templates to be attached
to a deployment.

Quick Tour
^^^^^^^^^^

Install aws-lambda plugin for :code:`vtsctl` as an example:

.. code:: bash

    vtsctl operator install aws-lambda

Initialize a vtsctl project. This enters an interactive mode asking users for related
deployment configurations:

.. code:: bash

    > vtsctl init

    Vtsctl Interactive Deployment Config Builder
    ...

    deployment config generated to: deployment_config.yaml
    ✨ generated template files.
      - vtsctl.tfvars
      - main.tf


Deployment config will be saved to :code:`./deployment_config.yaml`:

.. code:: yaml

    api_version: v1
    name: quickstart
    operator:
        name: aws-lambda
    template: terraform
    spec:
        region: us-west-1
        timeout: 10
        memory_size: 512

Now, we are ready to build the deployable artifacts required for this deployment. In
most cases, this step will product a new docker image specific to the target deployment
configuration:


.. code:: bash

    vtsctl build -b iris_classifier:btzv5wfv665trhcu -f ./deployment_config.yaml

Next step, use :code:`terraform` CLI command to apply the generated deployment configs
to AWS. This will require user setting up AWS credentials on the environment.


.. code:: bash

    > terraform init
    > terraform apply -var-file=vtsctl.tfvars --auto-approve

    ...
    base_url = "https://ka8h2p2yfh.execute-api.us-west-1.amazonaws.com/"
    function_name = "quickstart-function"
    image_tag = "192023623294.dkr.ecr.us-west-1.amazonaws.com/quickstart:btzv5wfv665trhcu"


Testing the endpoint deployed:

.. code:: bash

    URL=$(terraform output -json | jq -r .base_url.value)classify
    curl -i \
        --header "Content-Type: application/json" \
        --request POST \
        --data '[5.1, 3.5, 1.4, 0.2]' \
        $URL


Supported Cloud Platforms
^^^^^^^^^^^^^^^^^^^^^^^^^

- AWS Lambda: https://github.com/vtsserving/aws-lambda-deploy
- AWS SageMaker: https://github.com/vtsserving/aws-sagemaker-deploy
- AWS EC2: https://github.com/vtsserving/aws-ec2-deploy
- Google Cloud Run: https://github.com/vtsserving/google-cloud-run-deploy
- Google Compute Engine: https://github.com/vtsserving/google-compute-engine-deploy
- Azure Functions: https://github.com/vtsserving/azure-functions-deploy
- Azure Container Instances: https://github.com/vtsserving/azure-container-instances-deploy
- Heroku: https://github.com/vtsserving/heroku-deploy

.. TODO::
    Explain limitations of each platform, e.g. GPU support
    Explain how to customize the terraform workflow


About Horizontal Auto-scaling
-----------------------------

Auto-scaling is one of the most sought-after features when it comes to deploying models. Autoscaling helps optimize resource usage and cost by automatically provisioning up and scaling down depending on incoming traffic.

Among deployment options introduced in this guide, Yatai on Kubernetes is the
recommended approach if auto-scaling and resource efficiency are required for your team’s workflow.
Yatai enables users to fine-tune resource requirements and
auto-scaling policy at the Runner level, which inherently improves interoperability between auto-scaling and data aggregated at Runner's adaptive batching layer in real-time.

Many of vtsctl’s deployment targets also come with a certain level of auto-scaling
capabilities, including AWS EC2 and AWS Lambda.