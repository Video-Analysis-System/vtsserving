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
