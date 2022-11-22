import numpy as np
from flask import Flask
from flask import jsonify
from flask import request

import vtsserving
from vtsserving.io import NumpyNdarray

vts_model = vtsserving.sklearn.get("iris_clf:latest")
iris_clf_runner = vts_model.to_runner()

svc = vtsserving.Service("iris_flask_demo", runners=[iris_clf_runner])


@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
async def predict_vtsserving(input_series: np.ndarray) -> np.ndarray:
    return await iris_clf_runner.predict.async_run(input_series)


flask_app = Flask(__name__)
svc.mount_wsgi_app(flask_app)


@flask_app.route("/metadata")
def metadata():
    return {"name": vts_model.tag.name, "version": vts_model.tag.version}


# For demo purpose, here's an identical inference endpoint implemented via FastAPI
@flask_app.route("/predict_flask", methods=["POST"])
def predict():
    content_type = request.headers.get("Content-Type")
    if content_type == "application/json":
        input_arr = np.array(request.json, dtype=float)
        return jsonify(iris_clf_runner.predict.run(input_arr).tolist())
    else:
        return "Content-Type not supported!"
