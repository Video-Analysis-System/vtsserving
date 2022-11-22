# VtsServing monitoring example for classification tasks

This is a sample project demonstrating basic monitoring usage of [VtsServing](https://github.com/vtsserving).

In this project, we will train a classifier model using Scikit-learn and the Iris dataset, build
an prediction service for serving the trained model with monitoring enabled, and deploy the
model server as a docker image for production deployment.

### Install Dependencies

Install python packages required for running this project:
```bash
pip install -r ./requirements.txt
```

### Model Training

Create an Iris classifier and save it with `vtsserving.sklearn`:

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

### Serving the model
Draft a `service.py` file with monitoring data collection lines, and run your service with Bento Server locally:

```python
import numpy as np

import vtsserving
from vtsserving.io import Text
from vtsserving.io import NumpyNdarray

CLASS_NAMES = ["setosa", "versicolor", "virginica"]

iris_clf_runner = vtsserving.sklearn.get("iris_clf:latest").to_runner()
svc = vtsserving.Service("iris_classifier", runners=[iris_clf_runner])


@svc.api(
    input=NumpyNdarray.from_sample(np.array([4.9, 3.0, 1.4, 0.2], dtype=np.double)),
    output=Text(),
)
async def classify(features: np.ndarray) -> str:
    with vtsserving.monitor("iris_classifier_prediction") as mon:
        mon.log(features[0], name="sepal length", role="feature", data_type="numerical")
        mon.log(features[1], name="sepal width", role="feature", data_type="numerical")
        mon.log(features[2], name="petal length", role="feature", data_type="numerical")
        mon.log(features[3], name="petal width", role="feature", data_type="numerical")

        results = await iris_clf_runner.predict.async_run([features])
        result = results[0]
        category = CLASS_NAMES[result]

        mon.log(category, name="pred", role="prediction", data_type="categorical")
    return category
```

```bash
vtsserving serve service.py:svc --reload
```

Open your web browser at http://127.0.0.1:3000 to view the Bento UI for sending test requests.

You may also send request with `curl` command or any HTTP client, e.g.:

```bash
curl -X POST -H "content-type: application/json" --data "[[5.9, 3, 5.1, 1.8]]" http://127.0.0.1:3000/classify
```


Then you can find the exported data under the `./monitoring/<monitor_name>/data` directory.
Here's the example output:

```json
{"timestamp": "2022-11-02T12:38:38.701396", "request_id": 8781503815303167270, "sepal length": 5.9, "sepal width": 3.0, "petal length": 1.4, "petal width": 0.2, "pred": "0"}
{"timestamp": "2022-11-02T12:38:48.345552", "request_id": 14419506828678509143, "sepal length": 4.9, "sepal width": 3.0, "petal length": 1.4, "petal width": 0.2, "pred": "0"}
```


### Customizing the monitoring

You can customize the monitoring by modifying the config file of vtsserving. The default is:

```yaml
monitoring:
  enabled: true
  type: default
  options:
    output_dir: ./monitoring
```

You can draft your own vtsserving config file `deployment.yaml` and change the `output_dir` to any directory you want. You can also use other monitoring solutions by changing the `type` to your desired handler. For example, if you want to use the `arize` handler, you can change the config to:

```yaml
monitoring:
  enabled: true
  type: vtsserving_plugins.arize.ArizeMonitor
  options:
    api_key: <your_api_key>
    space_key: <your_space_key>
```

Then you can specify the config file through environment variable `VTSSERVING_CONFIG`:
```bash
VTSSERVING_CONFIG=deployment.yaml vtsserving serve service.py:svc
```


### Containerized Serving with monitoring

Bento is the distribution format in VtsServing which captures all the source code, model files, config
files and dependency specifications required for running the service for production deployment. Think 
of it as Docker/Container designed for machine learning models.

To begin with building Bento, create a `vtsfile.yaml` under your project directory:

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

Next, run `vtsserving build` from current directory to start the Bento build:

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

05/05/2022 19:19:17 INFO     [cli] Successfully built Bento(tag="iris_classifier:5wtigdwm4kwzduqj") at "/Users/vtsserving/vtsserving/vtss/iris_classifier/5wtigdwm4kwzduqj/"
```

A new Bento is now built and saved to local Bento store. You can view and manage it via 
`vtsserving list`,`vtsserving get` and `vtsserving delete` CLI command.

Then we will convert a Bento into a Docker image containing the HTTP model server.

Make sure you have docker installed and docker deamon running, and run the following commnand:

```bash
vtsserving containerize iris_classifier:latest
```

This will build a new docker image with all source code, model files and dependencies in place,
and ready for production deployment. To start a container with this docker image locally, run:

```bash
docker run -p 3000:3000 iris_classifier:invwzzsw7li6zckb2ie5eubhd --mount type=bind,source=<your directory>,target=/vts/monitoring
```

## What's Next?

- 👉 [Pop into our Slack community!](https://l.linklyhq.com/l/ktO8) We're happy to help with any issue you face or even just to meet you and hear what you're working on.
- Dive deeper into the [Core Concepts](https://docs.vtsserving.org/en/latest/concepts/index.html) in VtsServing
- Learn how to use VtsServing with other ML Frameworks at [Frameworks Guide](https://docs.vtsserving.org/en/latest/frameworks/index.html) or check out other [gallery projects](https://github.com/vtsserving/VtsServing/tree/main/examples)
- Learn more about model deployment options for Bento:
  - [🦄️ Yatai](https://github.com/vtsserving/Yatai): Model Deployment at scale on Kubernetes
  - [🚀 vtsctl](https://github.com/vtsserving/vtsctl): Fast model deployment on any cloud platform

