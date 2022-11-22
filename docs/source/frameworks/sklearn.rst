============
Scikit-Learn
============

Below is a simple example of using scikit-learn with VtsServing:

.. code:: python

    import vtsserving

    from sklearn.datasets import load_iris
    from sklearn.neighbors import KNeighborsClassifier

    model = KNeighborsClassifier()
    iris = load_iris()
    X = iris.data[:, :4]
    Y = iris.target
    model.fit(X, Y)

    # `save` a given classifier and retrieve coresponding tag:
    tag = vtsserving.sklearn.save_model('kneighbors', model)

    # retrieve metadata with `vtsserving.models.get`:
    metadata = vtsserving.models.get(tag)

    # load the model back:
    loaded = vtsserving.sklearn.load_model("kneighbors:latest")

    # Run a given model under `Runner` abstraction with `to_runner`
    runner = vtsserving.sklearn.get(tag).to_runner()
    runner.init_local()
    runner.run([[1,2,3,4,5]])

.. note::

   You can find more examples for **scikit-learn** in our `vtsserving/examples https://github.com/vtsserving/VtsServing/tree/main/examples`_ directory.
