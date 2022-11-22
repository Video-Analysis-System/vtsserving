===============================
Unified Model Serving Framework
===============================

|github_stars| |pypi_status| |downloads| |actions_status| |documentation_status| |join_slack|

----

What is VtsServing?
----------------

`VtsServing <https://github.com/vtsserving/VtsServing>`_ makes it easy to create ML-powered prediction services that are ready to deploy and scale.

Data Scientists and ML Engineers use VtsServing to:

* Accelerate and standardize the process of taking ML models to production
* Build scalable and high performance prediction services
* Continuously deploy, monitor, and operate prediction services in production

Learn VtsServing
-------------

.. grid:: 1 2 2 2
    :gutter: 3
    :margin: 0
    :padding: 3 4 0 0

    .. grid-item-card:: :doc:`💻 Tutorial: Intro to VtsServing <tutorial>`
        :link: tutorial
        :link-type: doc

        A simple example of using VtsServing in action. In under 10 minutes, you'll be able to serve your ML model over an HTTP API endpoint, and build a docker image that is ready to be deployed in production.

    .. grid-item-card:: :doc:`📖 Main Concepts <concepts/index>`
        :link: concepts/index
        :link-type: doc

        A step-by-step tour of VtsServing's components and introduce you to its philosophy. After reading, you will see what drives VtsServing's design, and know what `vts` and `runner` stands for.

    .. grid-item-card:: :doc:`🧮 ML Framework Guides <frameworks/index>`
        :link: frameworks/index
        :link-type: doc

        Best practices and example usages by the ML framework used for model training.

    .. grid-item-card:: `🎨 Gallery Projects <https://github.com/vtsserving/VtsServing/tree/main/examples>`_
        :link: https://github.com/vtsserving/VtsServing/tree/main/examples
        :link-type: url

        Example projects demonstrating VtsServing usage in a variety of different scenarios.

    .. grid-item-card:: :doc:`💪 Advanced Guides <guides/index>`
        :link: guides/index
        :link-type: doc

        Dive into VtsServing's advanced features, internals, and architecture, including GPU support, inference graph, monitoring, and performance optimization.

    .. grid-item-card:: :doc:`⚙️ Integrations & Ecosystem <integrations/index>`
        :link: integrations/index
        :link-type: doc

        Learn how VtsServing works together with other tools and products in the Data/ML ecosystem

    .. grid-item-card:: `💬 VtsServing Community <https://l.linklyhq.com/l/ktOX>`_
        :link: https://l.linklyhq.com/l/ktOX
        :link-type: url

        Join us in our Slack community where hundreds of ML practitioners are contributing to the project, helping other users, and discuss all things MLOps.


Beyond Model Serving
--------------------

.. grid:: 1 2 2 2
    :gutter: 3
    :margin: 0
    :padding: 3 4 0 0

    .. grid-item-card:: `🦄️ Yatai <https://github.com/vtsserving/Yatai>`_
        :link: https://github.com/vtsserving/Yatai
        :link-type: url

        Model Deployment at scale on Kubernetes.

    .. grid-item-card:: `🚀 vtsctl <https://github.com/vtsserving/vtsctl>`_
        :link: https://github.com/vtsserving/vtsctl
        :link-type: url

        Fast model deployment on any cloud platform.


Staying Informed
----------------

The `VtsServing Blog <http://modelserving.com>`_ and `@vtsservingai <http://twitt
er.com/vtsservingai>`_ on Twitter are the official source for
updates from the VtsServing team. Anything important, including major releases and announcements, will be posted there. We also frequently
share tutorials, case studies, and community updates there.

To receive release notification, star & watch the `VtsServing project on GitHub <https://github.com/vtsserving/vtsserving>`_. For release
notes and detailed changelog, see the `Releases <https://github.com/vtsserving/VtsServing/releases>`_ page.

----

Why are we building VtsServing?
----------------------------

Model deployment is one of the last and most important stages in the machine learning
life cycle: only by putting a machine learning model into a production environment and
making predictions for end applications, the full potential of ML can be realized.

Sitting at the intersection of data science and engineering, **model deployment
introduces new operational challenges between these teams**. Data scientists, who are
typically responsible for building and training the model, often don’t have the
expertise to bring it into production. At the same time, engineers, who aren’t used to
working with models that require continuous iteration and improvement, find it
challenging to leverage their know-how and common practices (like CI/CD) to deploy them.
As the two teams try to meet halfway to get the model over the finish line,
time-consuming and error-prone workflows can often be the result, slowing down the pace
of progress.

We at VtsServing want to **get your ML models shipped in a fast, repeatable, and scalable
way**. VtsServing is designed to streamline the handoff to production deployment, making it
easy for developers and data scientists alike to test, deploy, and integrate their
models with other systems.

With VtsServing, data scientists can focus primarily on creating and improving their
models, while giving deployment engineers peace of mind that nothing in the deployment
logic is changing and that production service is stable.

----

Getting Involved
----------------

VtsServing has a thriving open source community where hundreds of ML practitioners are
contributing to the project, helping other users and discuss all things MLOps.
`👉 Join us on slack today! <https://l.linklyhq.com/l/ktOX>`_


.. toctree::
   :hidden:

   installation
   tutorial
   concepts/index
   frameworks/index
   guides/index
   integrations/index
   reference/index
   Community <https://l.linklyhq.com/l/ktOX>
   GitHub <https://github.com/vtsserving/VtsServing>
   Blog <https://modelserving.com>


.. spelling::

.. |pypi_status| image:: https://img.shields.io/pypi/v/vtsserving.svg?style=flat-square
   :target: https://pypi.org/project/VtsServing
.. |downloads| image:: https://pepy.tech/badge/vtsserving?style=flat-square
   :target: https://pepy.tech/project/vtsserving
.. |actions_status| image:: https://github.com/vtsserving/vtsserving/workflows/CI/badge.svg
   :target: https://github.com/vtsserving/vtsserving/actions
.. |documentation_status| image:: https://readthedocs.org/projects/vtsserving/badge/?version=latest&style=flat-square
   :target: https://docs.vtsserving.org/
.. |join_slack| image:: https://badgen.net/badge/Join/VtsServing%20Slack/cyan?icon=slack&style=flat-square
   :target: https://l.linklyhq.com/l/ktOX
.. |github_stars| image:: https://img.shields.io/github/stars/vtsserving/VtsServing?color=%23c9378a&label=github&logo=github&style=flat-square
   :target: https://github.com/vtsserving/vtsserving
