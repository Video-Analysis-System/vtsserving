============
Apache Flink
============

Apache Flink DataStream
-----------------------

VtsServing support stream model inferencing in 
`Apache Flink DataStream API <https://nightlies.apache.org/flink/flink-docs-master/docs/dev/datastream/overview/>`_ 
through either embedded runners or remote calls to a separated deployed Vts Service. This guide assumes prior knowledge 
on using runners and service APIs.

Embedded Model Runners
^^^^^^^^^^^^^^^^^^^^^^
In VtsServing, a :ref:`Runner <concepts/runner:Using Runners>` 
represents a unit of computation, such as model inferencing, that can executed on either a remote runner process or an 
embedded runner instance. If available system resources allow loading the ML model in memory, invoking runners as embedded 
instances can typically achieve a better performance by avoiding the overhead incurred in the interprocess communication.

Runners can be initialized as embedded instances by calling `init_local()`. Once a runner is initialized, inference functions 
can be invoked on the runners.

.. code:: python

    import vtsserving

    iris_runner = vtsserving.transformers.get("text-classification:latest")).to_runner()
    iris_runner.init_local()
    iris_runner.predict.run(INPUT_TEXT)


To integrate with Flink DataRunners API, runners can be used in `ProcessWindowFunction`` for iterative inferencing or a 
`WindowFunction` for batched inferencing.

.. code:: python

    import vtsserving

    from pyflink.datastream import StreamExecutionEnvironment
    from pyflink.datastream.functions import RuntimeContext, MapFunction

    class ClassifyFunction(MapFunction):
        def open(self, runtime_context: RuntimeContext):
            self.runner = vtsserving.transformers.get(
                "text-classification:latest"
            ).to_runner()
            self.runner.init_local()

        def map(self, data):
            # transform(data)
            return data[0], self.runner.run(data[1])

The following is an end-to-end word classification example of using embedded runners in a Flink DataStream program. 
For simplicity, the input stream and output sink are abstracted out using in-memory collections and stdout sink.

.. code:: python

    import vtsserving
    import logging
    import sys

    from pyflink.datastream import StreamExecutionEnvironment
    from pyflink.datastream.functions import RuntimeContext, MapFunction


    class ClassifyFunction(MapFunction):
        def open(self, runtime_context: RuntimeContext):
            self.runner = vtsserving.transformers.get("text-classification:latest").to_runner()
            self.runner.init_local()

        def map(self, data):
            # transform(data)
            return data[0], self.runner.run(data[1])


    def classify_tweets():
        # Create a StreamExecutionEnvironment
        env = StreamExecutionEnvironment.get_execution_environment()
        env.set_parallelism(1)

        # Create source DataStream, e.g. Kafka, Table, etc. Example uses data collection for simplicity.
        ds = env.from_collection(
            collection=[
                (1, "VtsServing: Create an ML Powered Prediction Service in Minutes via @TDataScience https://buff.ly/3srhTw9 #Python #MachineLearning #VtsServing"),
                (2, "Top MLOps Serving frameworks — 2021 https://link.medium.com/5Elq6Aw52ib #mlops #TritonInferenceServer #opensource #nvidia #machincelearning  #serving #tensorflow #PyTorch #Bodywork #VtsServing #KFServing #kubeflow #Cortex #Seldon #Sagify #Syndicai"),
                (3, "#MLFlow provides components for experimentation management, ML project management. #VtsServing only focuses on serving and deploying trained models"),
                (4, "2000 and beyond #OpenSource #vtsserving"),
                (5, "Model Serving Made Easy https://github.com/vtsserving/VtsServing ⭐ 1.1K #Python #Vtsml #VtsServing #Modelserving #Modeldeployment #Modelmanagement #Mlplatform #Mlinfrastructure #Ml #Ai #Machinelearning #Awssagemaker #Awslambda #Azureml #Mlops #Aiops #Machinelearningoperations #Turn")
            ]
        )

        # Define the execution logic
        ds = ds.map(ClassifyFunction())
        
        # Create sink and emit result to sink, e.g. Kafka, File, Table, etc. Example prints to stdout for simplicity.
        ds.print()

        # Submit for execution
        env.execute()


    if __name__ == '__main__':
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
        classify_tweets()


Remote Vts Service
^^^^^^^^^^^^^^^^^^^^

Model runners can also be invoked remotely as a separately deployed Vts Service. Calling a remote Vts Service may be 
preferred if the model cannot be loaded into memory of the Flink DataStream program. This options is also advantageous because 
model runners can be scaled more easily with deployment frameworks like :ref:`Yatai <concepts/deploy:Deploy with Yatai>`.

To send a prediction request to a remotely deployed Vts Service in the DataStream program, you can use any HTTP client 
implementation of your choice inside the `MapFunction` or `ProcessWindowFunction`.


.. code:: python

    class ClassifyFunction(MapFunction):
        def map(self, data):
            return requests.post(
                "http://127.0.0.1:3000/classify",
                headers={"content-type": "text/plain"},
                data=TEXT_INPUT,
            ).text


Using a client with asynchronous IO support combined with Flink AsyncFunction is recommended to handle requests and responses 
concurrent and minimize IO waiting time of calling a remote Vts Service.
