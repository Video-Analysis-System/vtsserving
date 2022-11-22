import vtsserving

# import vtsserving.sklearn
# from vtsserving.io import NumpyNdarray

# iris_model_runner = vtsserving.sklearn.get('iris_classifier:latest').to_runner()
svc = vtsserving.Service(
    "test.simplevts",
    # runners=[iris_model_runner]
)

# @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
# def predict(request_data: np.ndarray):
#     return iris_model_runner.predict(request_data)

# For simple use cases, only models list is required:
# svc.vts_options.models = []
# svc.vts_files.include = ["*"]
# svc.vts_env.pip_install = "./requirements.txt"
