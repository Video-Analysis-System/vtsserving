===========
Metrics API
===========

VtsServing provides metrics API that uses `Prometheus <https://prometheus.io/>`_ under the hood.

VtsServing's ``vtsserving.metrics`` is a drop-in replacement for ``prometheus_client`` that should be used in VtsServing services:

.. code-block:: diff

   diff --git a/service.py b/service.py
   index acd8467e..0f3e6e77 100644
   --- a/service.py
   +++ b/service.py
   @@ -1,11 +1,10 @@
   -from prometheus_client import Summary
   +from vtsserving.metrics import Summary
    import random
    import time

   REQUEST_TIME = Summary("request_processing_seconds", "Time spent processing request")

   @REQUEST_TIME.time()
   def process_request(t):
       """A function that takes some time."""

While ``vtsserving.metrics`` contains all API that is offered by ``prometheus_client``,
users should always use ``vtsserving.metrics`` instead of ``prometheus_client`` in your service definition.

The reason is that VtsServing's ``vtsserving.metrics`` will construct metrics lazily and
ensure `multiprocessing mode <https://github.com/prometheus/client_python#multiprocess-mode-eg-gunicorn>`_. are correctly configured.

.. note::

   ``prometheus_client`` shouldn't be imported in VtsServing services, otherwise it will
   break multiprocessing mode.

.. note::

   All metrics from ``vtsserving.metrics`` will set up ``registry`` to handle multiprocess mode,
   which means you **SHOULD NOT** pass in ``registry`` argument to metrics initialization:

   .. code-block:: python
      :caption: service.py

      # THIS WILL NOT WORK
      from vtsserving.metrics import Summary, CollectorRegistry
      from vtsserving.metrics import multiprocess

      registry = CollectorRegistry()
      multiprocess.MultiProcessCollector(registry)
      REQUEST_TIME = Summary(
         "request_processing_seconds", "Time spent processing request", registry=registry
      )

   instead:

   .. code-block:: python
      :caption: service.py

      # THIS WILL WORK
      from vtsserving.metrics import Summary

      REQUEST_TIME = Summary("request_processing_seconds", "Time spent processing request")

-----

The following section will go over the most commonly used metrics API in
``vtsserving.metrics``:

.. currentmodule:: vtsserving.metrics

.. autofunction:: vtsserving.metrics.generate_latest

.. autofunction:: vtsserving.metrics.text_string_to_metric_families

.. autofunction:: vtsserving.metrics.Histogram

.. autofunction:: vtsserving.metrics.Counter

.. autofunction:: vtsserving.metrics.Summary

.. autofunction:: vtsserving.metrics.Gauge
