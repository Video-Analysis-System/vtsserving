# VTSServing

## Installation
In your terminal, run the following script:
```bat
pip install git+https://github.com/tungedng2710/vtsserving.git
```

<!-- [<img src="https://raw.githubusercontent.com/vtsserving/VtsServing/main/docs/source/_static/img/vtsserving-readme-header.jpeg" width="600px" margin-left="-5px">](https://github.com/vtsserving/VtsServing)
<br> -->

<!-- # The Unified Model Serving Framework [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=VtsServing:%20The%20Unified%20Model%20Serving%20Framework%20&url=https://github.com/vtsserving&via=vtsservingai&hashtags=mlops,vtsserving) -->

<!-- [![pypi_status](https://img.shields.io/pypi/v/vtsserving.svg)](https://pypi.org/project/VtsServing)
[![downloads](https://pepy.tech/badge/vtsserving)](https://pepy.tech/project/vtsserving)
[![actions_status](https://github.com/vtsserving/vtsserving/workflows/CI/badge.svg)](https://github.com/vtsserving/vtsserving/actions)
[![documentation_status](https://readthedocs.org/projects/vtsserving/badge/?version=latest)](https://docs.vtsserving.org/)
[![join_slack](https://badgen.net/badge/Join/VtsServing%20Slack/cyan?icon=slack)](https://join.slack.vtsserving.org) -->

<!-- VtsServing makes it easy to create Machine Learning services that are ready to deploy and scale.

👉 [Join our Slack community today!](https://l.vtsserving.com/join-slack)

✨ Looking deploy your ML service quickly? Checkout [VtsServing Cloud](https://www.vtsserving.com/vtsserving-cloud/)
for the easiest and fastest way to deploy your vts.

## Getting Started

- [Documentation](https://docs.vtsserving.org/) - Overview of the VtsServing docs and related resources
- [Tutorial: Intro to VtsServing](https://docs.vtsserving.org/en/latest/tutorial.html) - Learn by doing! In under 10 minutes, you'll serve a model via REST API and generate a docker image for deployment.
- [Main Concepts](https://docs.vtsserving.org/en/latest/concepts/index.html) - A step-by-step tour for learning main concepts in VtsServing
- [Examples](https://github.com/vtsserving/VtsServing/tree/main/examples) - Gallery of sample projects using VtsServing
- [ML Framework Guides](https://docs.vtsserving.org/en/latest/frameworks/index.html) - Best practices and example usages by the ML framework of your choice
- [Advanced Guides](https://docs.vtsserving.org/en/latest/guides/index.html) - Learn about VtsServing's internals, architecture and advanced features
- Need help? [Join VtsServing Community Slack 💬](https://l.linklyhq.com/l/ktOh)

---

## Highlights

🍭 Unified Model Serving API
pip install git+https://github.com/tungedng2710/vtsserving.git
- Framework-agnostic model packaging for Tensorflow, PyTorch, XGBoost, Scikit-Learn, ONNX, and [many more](https://docs.vtsserving.org/en/latest/frameworks/index.html)!
- Write **custom Python code** alongside model inference for pre/post-processing and business logic
- Apply the **same code** for online(REST API or gRPC), offline batch, and streaming inference
- Simple abstractions for building **multi-model inference** pipelines or graphs

🚂 **Standardized process** for a frictionless transition to production

- Build [Vts](https://docs.vtsserving.org/en/latest/concepts/vts.html) as the standard deployable artifact for ML services
- Automatically **generate docker images** with the desired dependencies
- Easy CUDA setup for inference with GPU
- Rich integration with the MLOps ecosystem, including Kubeflow, Airflow, MLFlow, Triton

🏹 **_Scalable_** with powerful performance optimizations

- [Adaptive batching](https://docs.vtsserving.org/en/latest/guides/batching.html) dynamically groups inference requests on server-side optimal performance
- [Runner](https://docs.vtsserving.org/en/latest/concepts/runner.html) abstraction scales model inference separately from your custom code
- [Maximize your GPU](https://docs.vtsserving.org/en/latest/guides/gpu.html) and multi-core CPU utilization with automatic provisioning

🎯 Deploy anywhere in a **DevOps-friendly** way

- Streamline production deployment workflow via:
  - [☁️ VtsServing Cloud](https://www.vtsserving.com/vtsserving-cloud/): the fastest way to deploy your vts, simple and at scale
  - [🦄️ Yatai](https://github.com/vtsserving/yatai): Model Deployment at scale on Kubernetes
  - [🚀 vtsctl](https://github.com/vtsserving/vtsctl): Fast model deployment on AWS SageMaker, Lambda, ECE, GCP, Azure, Heroku, and more!
- Run offline batch inference jobs with Spark or Dask
- Built-in support for Prometheus metrics and OpenTelemetry
- Flexible APIs for advanced CI/CD workflows -->

## How it works

Save your trained model with VtsServing:

```python
import vtsserving

saved_model = vtsserving.pytorch.save_model(
    "demo_mnist", # model name in the local model store
    model, # model instance being saved
)

print(f"Model saved: {saved_model}")
# Model saved: Model(tag="demo_mnist:3qee3zd7lc4avuqj", path="~/vtsserving/models/demo_mnist/3qee3zd7lc4avuqj/")
```

Define a prediction service in a `service.py` file:

```python
import numpy as np
import vtsserving
from vtsserving.io import NumpyNdarray, Image
from PIL.Image import Image as PILImage

mnist_runner = vtsserving.pytorch.get("demo_mnist:latest").to_runner()

svc = vtsserving.Service("pytorch_mnist", runners=[mnist_runner])

@svc.api(input=Image(), output=NumpyNdarray(dtype="int64"))
def predict(input_img: PILImage):
    img_arr = np.array(input_img)/255.0
    input_arr = np.expand_dims(img_arr, 0).astype("float32")
    output_tensor = mnist_runner.predict.run(input_arr)
    return output_tensor.numpy()
```

Create a `vtsfile.yaml` build file for your ML service:

```yaml
service: "service:svc"
include:
  - "*.py"
python:
  packages:
    - numpy
    - torch
    - Pillow
```

Now, run the prediction service:

```bash
vtsserving serve
```

Sent a prediction request:

```bash
curl -F 'image=@samples/1.png' http://127.0.0.1:3000/predict_image
```

Build a Vts and generate a docker image:

```bash
$ vtsserving build
Successfully built Vts(tag="pytorch_mnist:4mymorgurocxjuqj") at "~/vtsserving/vtss/pytorch_mnist/4mymorgurocxjuqj/"

$ vtsserving containerize pytorch_mnist:4mymorgurocxjuqj
Successfully built docker image "pytorch_mnist:4mymorgurocxjuqj"

$ docker run -p 3000:3000 pytorch_mnist:4mymorgurocxjuqj
Starting production VtsServer from "pytorch_mnist:4mymorgurocxjuqj" running on http://0.0.0.0:3000
```

<!-- For a more detailed user guide, check out the [VtsServing Tutorial](https://docs.vtsserving.org/en/latest/tutorial.html).

---

## Community

- For general questions and support, join the [community slack](https://l.linklyhq.com/l/ktOh).
- To receive release notification, star & watch the VtsServing project on [GitHub](https://github.com/vtsserving/VtsServing).
- To report a bug or suggest a feature request, use [GitHub Issues](https://github.com/vtsserving/VtsServing/issues/new/choose).
- To stay informed with community updates, follow the [VtsServing Blog](http://modelserving.com) and [@vtsservingai](http://twitter.com/vtsservingai) on Twitter.

## Contributing

There are many ways to contribute to the project:

- If you have any feedback on the project, share it under the `#vtsserving-contributors` channel in the [community slack](https://l.linklyhq.com/l/ktOh).
- Report issues you're facing and "Thumbs up" on issues and feature requests that are relevant to you.
- Investigate bugs and reviewing other developer's pull requests.
- Contributing code or documentation to the project by submitting a GitHub pull request. Check out the [Development Guide](https://github.com/vtsserving/VtsServing/blob/main/DEVELOPMENT.md).
- Learn more in the [contributing guide](https://github.com/vtsserving/VtsServing/blob/main/CONTRIBUTING.md).

### Contributors

Thanks to all of our amazing contributors!

<a href="https://github.com/vtsserving/VtsServing/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=vtsserving/VtsServing" />
</a>

---

### Usage Reporting

VtsServing collects usage data that helps our team to improve the product.
Only VtsServing's internal API calls are being reported. We strip out as much potentially
sensitive information as possible, and we will never collect user code, model data, model names, or stack traces.
Here's the [code](./src/vtsserving/_internal/utils/analytics/usage_stats.py) for usage tracking.
You can opt-out of usage tracking by the `--do-not-track` CLI option:

```bash
vtsserving [command] --do-not-track
```

Or by setting environment variable `VTSSERVING_DO_NOT_TRACK=True`:

```bash
export VTSSERVING_DO_NOT_TRACK=True
```

---

### License

[Apache License 2.0](https://github.com/vtsserving/VtsServing/blob/main/LICENSE)

[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fvtsserving%2FVtsServing.svg?type=small)](https://app.fossa.com/projects/git%2Bgithub.com%2Fvtsserving%2FVtsServing?ref=badge_small) -->
## Examples
Some examples are placed under the ```examples``` folder, please checkout it.

For instance, with ```torch_hub_yolov5```, run the following commands
```bat
cd examples/torch_hub_yolov5
pip install -r requirements.txt
vtsserving serve service.py:svc --reload
```
In the first run, the YOLOv5s model will be downloaded from ```torch.hub```. Then open the web page to try it out (default address: http://0.0.0.0:3000). I also provided some samples in the ```test_images``` folder. For other example, you may have to train the model before serving them.
