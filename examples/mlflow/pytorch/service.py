import vtsserving

mnist_runner = vtsserving.mlflow.get("mlflow_pytorch_mnist:latest").to_runner()

svc = vtsserving.Service("mlflow_pytorch_mnist_demo", runners=[mnist_runner])

input_spec = vtsserving.io.NumpyNdarray(
    dtype="float32",
    shape=[-1, 1, 28, 28],
    enforce_dtype=True,
)


@svc.api(input=input_spec, output=vtsserving.io.NumpyNdarray())
async def predict(input_arr):
    return await mnist_runner.predict.async_run(input_arr)
