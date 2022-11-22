Since there is no easy way to add additional proto files, we will have to clone some
repositories and copy the proto files into our project:

1. :github:`protocolbuffers/protobuf` - the official repository for the Protocol Buffers. We will need protobuf files that lives under ``src/google/protobuf``:

.. code-block:: bash

   » mkdir -p thirdparty && cd thirdparty
   » git clone --depth 1 https://github.com/protocolbuffers/protobuf.git

2. :github:`vtsserving/vtsserving` - We need the ``service.proto`` under ``vtsserving/grpc/`` to build the client, therefore, we will perform
   a `sparse checkout <https://github.blog/2020-01-17-bring-your-monorepo-down-to-size-with-sparse-checkout/>`_ to only checkout ``vtsserving/grpc`` directory:

.. code-block:: bash

   » mkdir vtsserving && pushd vtsserving
   » git init
   » git remote add -f origin https://github.com/vtsserving/VtsServing.git
   » git config core.sparseCheckout true
   » cat <<EOT >|.git/info/sparse-checkout
   src/vtsserving/grpc
   EOT
   » git pull origin main && mv src/vtsserving/grpc .
   » popd
