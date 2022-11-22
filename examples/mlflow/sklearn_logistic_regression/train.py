import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression

import vtsserving

if __name__ == "__main__":
    X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
    y = np.array([0, 0, 1, 1, 1, 0])
    lr = LogisticRegression()
    lr.fit(X, y)
    score = lr.score(X, y)
    print("Score: %s" % score)
    mlflow.log_metric("score", score)
    logged_model = mlflow.sklearn.log_model(lr, "model")
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)

    # Import logged mlflow model to VtsServing model store for serving:
    vts_model = vtsserving.mlflow.import_model(
        "logistic_regression_model", logged_model.model_uri
    )
    print("Model imported to VtsServing: %s" % vts_model)
