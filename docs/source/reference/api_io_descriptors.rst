==================
API IO Descriptors
==================

IO Descriptors are used for describing the input and output spec of a Service API.
Here's a list of built-in IO Descriptors and APIs for extending custom IO Descriptor.

NumPy ``ndarray``
-----------------

.. note::

   The :code:`numpy` package is required to use the :obj:`vtsserving.io.NumpyNdarray`.

   Install it with ``pip install numpy`` and add it to your :code:`vtsfile.yaml`'s under either Python or Conda packages list.

   Refer to :ref:`Build Options <concepts/vts:Vts Build Options>`.

   .. tab-set::

      .. tab-item:: pip

         .. code-block:: yaml
            :caption: `vtsfile.yaml`

            ...
            python:
              packages:
                - numpy

      .. tab-item:: conda

         .. code-block:: yaml
            :caption: `vtsfile.yaml`

            ...
            conda:
              channels:
                - conda-forge
              dependencies:
                - numpy


.. autoclass:: vtsserving.io.NumpyNdarray
.. automethod:: vtsserving.io.NumpyNdarray.from_sample
.. automethod:: vtsserving.io.NumpyNdarray.from_proto
.. automethod:: vtsserving.io.NumpyNdarray.from_http_request
.. automethod:: vtsserving.io.NumpyNdarray.to_proto
.. automethod:: vtsserving.io.NumpyNdarray.to_http_response


Tabular Data with Pandas
------------------------

To use the IO descriptor, install vtsserving with extra ``io-pandas`` dependency:

.. code-block:: bash

    pip install "vtsserving[io-pandas]"

.. note::

   The :code:`pandas` package is required to use the :obj:`vtsserving.io.PandasDataFrame`
   or :obj:`vtsserving.io.PandasSeries`. 

   Install it with ``pip install pandas`` and add it to your :code:`vtsfile.yaml`'s under either Python or Conda packages list.

   Refer to :ref:`Build Options <concepts/vts:Vts Build Options>`.

   .. tab-set::

      .. tab-item:: pip

         .. code-block:: yaml
            :caption: `vtsfile.yaml`

            ...
            python:
              packages:
                - pandas

      .. tab-item:: conda

         .. code-block:: yaml
            :caption: `vtsfile.yaml`

            ...
            conda:
              channels:
                - conda-forge
              dependencies:
                - pandas

.. autoclass:: vtsserving.io.PandasDataFrame
.. automethod:: vtsserving.io.PandasDataFrame.from_sample
.. automethod:: vtsserving.io.PandasDataFrame.from_proto
.. automethod:: vtsserving.io.PandasDataFrame.from_http_request
.. automethod:: vtsserving.io.PandasDataFrame.to_proto
.. automethod:: vtsserving.io.PandasDataFrame.to_http_response
.. autoclass:: vtsserving.io.PandasSeries
.. automethod:: vtsserving.io.PandasSeries.from_sample
.. automethod:: vtsserving.io.PandasSeries.from_proto
.. automethod:: vtsserving.io.PandasSeries.from_http_request
.. automethod:: vtsserving.io.PandasSeries.to_proto
.. automethod:: vtsserving.io.PandasSeries.to_http_response


Structured Data with JSON
-------------------------
.. note::

   For common structure data, we **recommend** using the :obj:`JSON` descriptor, as it provides
   the most flexibility. Users can also define a schema of the JSON data via a
   `Pydantic <https://pydantic-docs.helpmanual.io/>`_ model, and use it to for data
   validation.

   To use the IO descriptor with pydantic, install vtsserving with extra ``io-json`` dependency:

   .. code-block:: bash

      pip install "vtsserving[io-json]"

   This will include VtsServing with `Pydantic <https://pydantic-docs.helpmanual.io/>`_
   alongside with VtsServing

   Then proceed to add it to your :code:`vtsfile.yaml`'s under either Python or Conda packages list.

   Refer to :ref:`Build Options <concepts/vts:Vts Build Options>`.

   .. tab-set::

      .. tab-item:: pip

         .. code-block:: yaml
            :caption: `vtsfile.yaml`

            ...
            python:
              packages:
                - pydantic

      .. tab-item:: conda

         .. code-block:: yaml
            :caption: `vtsfile.yaml`

            ...
            conda:
              channels:
                - conda-forge
              dependencies:
                - pydantic

   Refers to :ref:`Build Options <concepts/vts:Vts Build Options>`.

   .. tab-set::

      .. tab-item:: pip

         .. code-block:: yaml
            :caption: `vtsfile.yaml`

            ...
            python:
              packages:
                - pydantic

      .. tab-item:: conda

         .. code-block:: yaml
            :caption: `vtsfile.yaml`

            ...
            conda:
              channels:
                - conda-forge
              dependencies:
                - pydantic

.. autoclass:: vtsserving.io.JSON
.. automethod:: vtsserving.io.JSON.from_sample
.. automethod:: vtsserving.io.JSON.from_proto
.. automethod:: vtsserving.io.JSON.from_http_request
.. automethod:: vtsserving.io.JSON.to_proto
.. automethod:: vtsserving.io.JSON.to_http_response

Texts
-----
:code:`vtsserving.io.Text` is commonly used for NLP Applications:

.. autoclass:: vtsserving.io.Text
.. automethod:: vtsserving.io.Text.from_proto
.. automethod:: vtsserving.io.Text.from_http_request
.. automethod:: vtsserving.io.Text.to_proto
.. automethod:: vtsserving.io.Text.to_http_response

Images
------

To use the IO descriptor, install vtsserving with extra ``io-image`` dependency:


.. code-block:: bash

    pip install "vtsserving[io-image]"

.. note::

   The :code:`Pillow` package is required to use the :obj:`vtsserving.io.Image`.

   Install it with ``pip install Pillow`` and add it to your :code:`vtsfile.yaml`'s under either Python or Conda packages list.

   Refer to :ref:`Build Options <concepts/vts:Vts Build Options>`.

   .. tab-set::

      .. tab-item:: pip

         .. code-block:: yaml
            :caption: `vtsfile.yaml`

            ...
            python:
              packages:
                - Pillow

      .. tab-item:: conda

         .. code-block:: yaml
            :caption: `vtsfile.yaml`

            ...
            conda:
              channels:
                - conda-forge
              dependencies:
                - Pillow

.. autoclass:: vtsserving.io.Image
.. automethod:: vtsserving.io.Image.from_proto
.. automethod:: vtsserving.io.Image.from_http_request
.. automethod:: vtsserving.io.Image.to_proto
.. automethod:: vtsserving.io.Image.to_http_response

Files
-----

.. autoclass:: vtsserving.io.File
.. automethod:: vtsserving.io.File.from_proto
.. automethod:: vtsserving.io.File.from_http_request
.. automethod:: vtsserving.io.File.to_proto
.. automethod:: vtsserving.io.File.to_http_response

Multipart Payloads
------------------

.. note::
    :code:`io.Multipart` makes it possible to compose a multipart payload from multiple
    other IO Descriptor instances. For example, you may create a Multipart input that
    contains a image file and additional metadata in JSON.

.. autoclass:: vtsserving.io.Multipart
.. automethod:: vtsserving.io.Multipart.from_proto
.. automethod:: vtsserving.io.Multipart.from_http_request
.. automethod:: vtsserving.io.Multipart.to_proto
.. automethod:: vtsserving.io.Multipart.to_http_response

Custom IODescriptor
-------------------

.. note::
    The IODescriptor base class can be extended to support custom data format for your
    APIs, if the built-in descriptors does not fit your needs.

.. autoclass:: vtsserving.io.IODescriptor
