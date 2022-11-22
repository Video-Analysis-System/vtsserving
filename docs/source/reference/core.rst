===============
Core Components
===============

vtsserving.Service
---------------

.. autoclass:: vtsserving.Service
    :members: api, runners, apis, mount_asgi_app, mount_wsgi_app, add_asgi_middleware
    :undoc-members:

.. autofunction:: vtsserving.load

.. TODO::
    Add docstring to the following classes/functions

vtsserving.build
-------------

.. autofunction:: vtsserving.vtss.build

.. autofunction:: vtsserving.vtss.build_vtsfile

.. autofunction:: vtsserving.vtss.containerize


vtsserving.Bento
-------------

.. autoclass:: vtsserving.Bento
    :members: tag, info, path, path_of, doc
    :undoc-members:

vtsserving.Runner
--------------

.. autoclass:: vtsserving.Runner

vtsserving.Runnable
----------------

.. autoclass:: vtsserving.Runnable
    :members: method
    :undoc-members:

Tag
---

.. autoclass:: vtsserving.Tag

Model
-----

.. autoclass:: vtsserving.Model
    :members: to_runner, to_runnable, info, path, path_of, with_options
    :undoc-members:


YataiClient
-----------

.. autoclass:: vtsserving.YataiClient
