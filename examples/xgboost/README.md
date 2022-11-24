# VtsServing XGBoost Demo

This is a sample project demonstrating basic usage of VtsServing with XGBoost.

### Install Dependencies

Install python packages required for running this project:

```bash
pip install -r ./requirements.txt
```

### Model Training

First step, train a classification model with the UCI Machine Learning Repository Agaricus mushroom
dataset and save the model with VtsServing:

```bash
python train.py
```

This should save a new model in the VtsServing local model store:

```bash
vtsserving models list agaricus
```

Verify that the model can be loaded as runner from Python shell:

```python
import vtsserving

runner = vtsserving.xgboost.load_runner("agaricus:latest")

runner.run([0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1])  # => array(0.01241208, dtype=float32)
```

### Create ML Service

The ML Service code is defined in the `agaricus.py` file:

```python
# agaricus.py
import typing

import vtsserving
import xgboost
from vtsserving.io import NumpyNdarray, File

if typing.TYPE_CHECKING:
    import numpy as np

agaricus_runner = vtsserving.xgboost.load_runner("agaricus:latest")

svc = vtsserving.Service("agaricus", runners=[agaricus_runner])


@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_data: "np.ndarray") -> "np.ndarray":
    return agaricus_runner.run(input_data)
```

Start an API server locally to test the service code above:

```bash
vtsserving serve agaricus:svc --reload
```

With the `--reload` flag, the API server will automatically restart when the source
file `agaricus.py` is being edited, to boost your development productivity.

Verify the endpoint can be accessed locally:

```bash
curl -X POST -H "content-type: application/json" --data "[[0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]]" http://127.0.0.1:3000/classify
```

### Build Vts for deployment

A `vtsfile` for the agaricus service is also contained in this directory:

```yaml
service: "agaricus:svc"
description: "file: ./README.md"
labels:
  owner: vtsserving-team
  stage: demo
include:
  - "*.py"
exclude:
  - "locustfile.py"
python:
  packages:
    - xgboost
```

Simply run `vtsserving build` from this directory to build a Vts with the latest version of the
`agaricus` model. This may take a while when running for the first time, as VtsServing needs to resolve
all dependency versions:

```
> vtsserving build

03/07/2022 12:25:16 PM INFO     [cli] Building VtsServing service "agaricus:uvkv7d46cgnvgeb5" from build context "/home/user/devel/gallery/xgboost"
03/07/2022 12:25:16 PM INFO     [cli] Packing model "agaricus:3t4533c6ufi6zcz2ca6rzl235" from "/home/user/vtsserving/models/agaricus/3t4533c6ufi6zcz2ca6rzl235"
03/07/2022 12:25:16 PM INFO     [cli] Successfully saved Model(tag="agaricus:3t4533c6ufi6zcz2ca6rzl235",
                                path="/run/user/1000/tmpw8lyba_svtsserving_vts_agaricus/models/agaricus/3t4533c6ufi6zcz2ca6rzl235/")
03/07/2022 12:25:16 PM INFO     [cli] Locking PyPI package versions..
03/07/2022 12:25:17 PM INFO     [cli]
                                ██████╗░███████╗███╗░░██╗████████╗░█████╗░███╗░░░███╗██╗░░░░░
                                ██╔══██╗██╔════╝████╗░██║╚══██╔══╝██╔══██╗████╗░████║██║░░░░░
                                ██████╦╝█████╗░░██╔██╗██║░░░██║░░░██║░░██║██╔████╔██║██║░░░░░
                                ██╔══██╗██╔══╝░░██║╚████║░░░██║░░░██║░░██║██║╚██╔╝██║██║░░░░░
                                ██████╦╝███████╗██║░╚███║░░░██║░░░╚█████╔╝██║░╚═╝░██║███████╗
                                ╚═════╝░╚══════╝╚═╝░░╚══╝░░░╚═╝░░░░╚════╝░╚═╝░░░░░╚═╝╚══════╝

03/07/2022 12:25:17 PM INFO     [cli] Successfully built Vts(tag="agaricus:uvkv7d46cgnvgeb5") at "/home/user/vtsserving/vtss/agaricus/uvkv7d46cgnvgeb5/"
```

This Vts can now be served:

```bash
vtsserving serve agaricus:latest --production
```

The Vts directory contains all code, files, models and configuration required to run this service.
VtsServing standarizes this file structure, enabling serving runtimes and deployment tools to be built
on top of it. By default, Bentos are managed under the `~/vtsserving/vtss` directory:

```
> cd $(vtsserving get agaricus:latest -o path)

> tree
.
├── apis
│   └── openapi.yaml
├── vts.yaml
├── env
│   ├── conda
│   ├── docker
│   │   ├── Dockerfile
│   │   ├── entrypoint.sh
│   │   └── init.sh
│   └── python
│       ├── requirements.lock.txt
│       ├── requirements.txt
│       └── version.txt
├── models
│   └── agaricus
│       ├── 3t4533c6ufi6zcz2ca6rzl235
│       │   ├── model.yaml
│       │   └── saved_model.json
│       └── latest
├── README.md
└── src
    ├── agaricus.py
    └── train.py

9 directories, 14 files
```

### Containerize Vts for deployment

Make sure you have docker installed and the docker daemon is running. The following command will use
your local docker environment to build a new docker image containing the Vts:

```bash
vtsserving containerize agaricus:latest
```

To test out the newly created docker image:

```bash
docker run agaricus:<vts tag output> -p 3000:3000
```
