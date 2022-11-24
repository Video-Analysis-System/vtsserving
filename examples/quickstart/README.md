# VtsServing Scikit-Learn Tutorial

This is a sample project demonstrating basic usage of [VtsServing](https://github.com/vtsserving) with
Scikit-learn.

In this project, we will train a classifier model using Scikit-learn and the Iris dataset, build
an prediction service for serving the trained model via an HTTP server, and containerize the 
model server as a docker image for production deployment.

This project is also available to run from a notebook: https://github.com/vtsserving/VtsServing/blob/main/examples/quickstart/iris_classifier.ipynb

### Install Dependencies

Install python packages required for running this project:
```bash
pip install -r ./requirements.txt
```

### Model Training

First step, train a classification model with sklearn's built-in iris dataset and save the model
with VtsServing:

```bash
import vtsserving
from sklearn import svm, datasets

# Load training data
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Model Training
clf = svm.SVC()
clf.fit(X, y)

# Save model to VtsServing local model store
vtsserving.sklearn.save_model("iris_clf", clf)
```

This will save a new model in the VtsServing local model store, a new version tag is automatically
generated when the model is saved. You can see all model revisions from CLI via `vtsserving models`
commands:

```bash
vtsserving models get iris_clf:latest

vtsserving models list

vtsserving models --help
```

To verify that the saved model can be loaded correctly, run the following:

```python
import vtsserving

loaded_model = vtsserving.sklearn.load_model("iris_clf:latest")

loaded_model.predict([[5.9, 3. , 5.1, 1.8]])  # => array(2)
```

In VtsServing, the recommended way of running ML model inference in serving is via Runner, which 
gives VtsServing more flexibility in terms of how to schedule the inference computation, how to 
batch inference requests and take advantage of hardware resoureces available. Saved models can
be loaded as Runner instance as shown below:

```python
import vtsserving

# Create a Runner instance:
iris_clf_runner = vtsserving.sklearn.get("iris_clf:latest").to_runner()

# Runner#init_local initializes the model in current process, this is meant for development and testing only:
iris_clf_runner.init_local()

# This should yield the same result as the loaded model:
iris_clf_runner.predict.run([[5.9, 3., 5.1, 1.8]])
```


### Serving the model

A simple VtsServing Service that serves the model saved above look like this:

```python
import numpy as np
import vtsserving
from vtsserving.io import NumpyNdarray

iris_clf_runner = vtsserving.sklearn.get("iris_clf:latest").to_runner()

svc = vtsserving.Service("iris_classifier", runners=[iris_clf_runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
async def classify(input_series: np.ndarray) -> np.ndarray:
    return await iris_clf_runner.predict.async_run(input_series)
```

Copy this to a `service.py` file, and run your service with Vts Server locally:

```bash
vtsserving serve service.py:svc --reload
```

Open your web browser at http://127.0.0.1:3000 to view the Vts UI for sending test requests.

You may also send request with `curl` command or any HTTP client, e.g.:

```bash
curl -X POST -H "content-type: application/json" --data "[[5.9, 3, 5.1, 1.8]]" http://127.0.0.1:3000/classify
```


### Build Vts for deployment

Vts is the distribution format in VtsServing which captures all the source code, model files, config
files and dependency specifications required for running the service for production deployment. Think 
of it as Docker/Container designed for machine learning models.

To begin with building Vts, create a `vtsfile.yaml` under your project directory:

```yaml
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
```

Next, run `vtsserving build` from current directory to start the Vts build:

```
> vtsserving build

05/05/2022 19:19:16 INFO     [cli] Building VtsServing service "iris_classifier:5wtigdwm4kwzduqj" from build context "/Users/vtsserving/workspace/gallery/quickstart"
05/05/2022 19:19:16 INFO     [cli] Packing model "iris_clf:4i7wbngm4crhpuqj" from "/Users/vtsserving/vtsserving/models/iris_clf/4i7wbngm4crhpuqj"
05/05/2022 19:19:16 INFO     [cli] Successfully saved Model(tag="iris_clf:4i7wbngm4crhpuqj",
                             path="/var/folders/bq/gdsf0kmn2k1bf880r_l238600000gn/T/tmp26dx354uvtsserving_vts_iris_classifier/models/iris_clf/4i7wbngm4crhpuqj/")
05/05/2022 19:19:16 INFO     [cli] Locking PyPI package versions..
05/05/2022 19:19:17 INFO     [cli]
                             ██████╗░███████╗███╗░░██╗████████╗░█████╗░███╗░░░███╗██╗░░░░░
                             ██╔══██╗██╔════╝████╗░██║╚══██╔══╝██╔══██╗████╗░████║██║░░░░░
                             ██████╦╝█████╗░░██╔██╗██║░░░██║░░░██║░░██║██╔████╔██║██║░░░░░
                             ██╔══██╗██╔══╝░░██║╚████║░░░██║░░░██║░░██║██║╚██╔╝██║██║░░░░░
                             ██████╦╝███████╗██║░╚███║░░░██║░░░╚█████╔╝██║░╚═╝░██║███████╗
                             ╚═════╝░╚══════╝╚═╝░░╚══╝░░░╚═╝░░░░╚════╝░╚═╝░░░░░╚═╝╚══════╝

05/05/2022 19:19:17 INFO     [cli] Successfully built Vts(tag="iris_classifier:5wtigdwm4kwzduqj") at "/Users/vtsserving/vtsserving/vtss/iris_classifier/5wtigdwm4kwzduqj/"
```

A new Vts is now built and saved to local Vts store. You can view and manage it via 
`vtsserving list`,`vtsserving get` and `vtsserving delete` CLI command.


### Containerize and Deployment

Vts is designed to be deployed to run efficiently in a variety of different environments.
And there are lots of deployment options and tools as part of the VtsServing eco-system, such as 
[Yatai](https://github.com/vtsserving/Yatai) and [vtsctl](https://github.com/vtsserving/vtsctl) for
direct deployment to cloud platforms.

In this guide, we will show you the most basic way of deploying a Vts, which is converting a Vts
into a Docker image containing the HTTP model server.

Make sure you have docker installed and docker deamon running, and run the following commnand:

```bash
vtsserving containerize iris_classifier:latest
```

This will build a new docker image with all source code, model files and dependencies in place,
and ready for production deployment. To start a container with this docker image locally, run:

```bash
docker run -p 3000:3000 iris_classifier:invwzzsw7li6zckb2ie5eubhd 
```

## What's Next?

- 👉 [Pop into our Slack community!](https://l.linklyhq.com/l/ktO8) We're happy to help with any issue you face or even just to meet you and hear what you're working on.
- Dive deeper into the [Core Concepts](https://docs.vtsserving.org/en/latest/concepts/index.html) in VtsServing
- Learn how to use VtsServing with other ML Frameworks at [Frameworks Guide](https://docs.vtsserving.org/en/latest/frameworks/index.html) or check out other [gallery projects](https://github.com/vtsserving/VtsServing/tree/main/examples)
- Learn more about model deployment options for Vts:
  - [🦄️ Yatai](https://github.com/vtsserving/Yatai): Model Deployment at scale on Kubernetes
  - [🚀 vtsctl](https://github.com/vtsserving/vtsctl): Fast model deployment on any cloud platform

