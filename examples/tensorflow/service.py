import numpy as np
from PIL.Image import Image as PILImage

import vtsserving
from vtsserving.io import Image
from vtsserving.io import NumpyNdarray

mnist_runner = vtsserving.tensorflow.get("tensorflow_mnist:latest").to_runner()

svc = vtsserving.Service(
    name="tensorflow_mnist_demo",
    runners=[mnist_runner],
)


@svc.api(input=Image(), output=NumpyNdarray(dtype="float32"))
async def predict_image(f: PILImage) -> "np.ndarray":
    assert isinstance(f, PILImage)
    arr = np.array(f) / 255.0
    assert arr.shape == (28, 28)

    # We are using greyscale image and our PyTorch model expect one
    # extra channel dimension
    arr = np.expand_dims(arr, (0, 3)).astype("float32")  # reshape to [1, 28, 28, 1]
    return await mnist_runner.async_run(arr)
