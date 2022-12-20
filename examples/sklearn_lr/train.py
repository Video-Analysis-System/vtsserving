from sklearn import linear_model

import vtsserving

if __name__ == "__main__":
    reg = linear_model.LinearRegression()
    reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])

    print("coef: ", reg.coef_)
    vts_model = vtsserving.sklearn.save_model("linear_reg", reg)
    print(f"Model saved: {vts_model}")

    # Test running inference with VtsServing runner
    test_runner = vtsserving.sklearn.get("linear_reg:latest").to_runner()
    test_runner.init_local()
    assert test_runner.predict.run([[1, 1]]) == reg.predict([[1, 1]])
