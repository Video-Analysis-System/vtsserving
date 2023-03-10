#!/bin/bash

if [ "$#" -eq 1 ]; then
	VTSSERVING_VERSION=$1
else
	VTSSERVING_VERSION=$(python -c "import vtsserving; print(vtsserving.__version__)")
	echo "Releasing with current VtsServing Version $VTSSERVING_VERSION"
fi

export DOCKER_BUILDKIT=1

docker buildx build --platform=linux/arm64,linux/amd64 -t vtsserving/quickstart:$VTSSERVING_VERSION -t vtsserving/quickstart:latest --pull -o type=image,push=True -f- . <<EOF
FROM jupyter/minimal-notebook:python-3.9.13

# ./start.sh requires root permission to set up notebook user and ensure access to home directory 
USER root
WORKDIR /home/vtsserving


COPY ../examples/quickstart .
RUN pip install -U pip "vtsserving[grpc]==${VTSSERVING_VERSION}" && pip install -r ./requirements.txt

# For jupyter notebook UI
EXPOSE 8888
# For accessing VtsServer
EXPOSE 3000
EXPOSE 3001

ENV NB_USER=vtsserving \
    NB_UID=1101 \
    NB_GID=1101 \
    CHOWN_HOME=yes \ 
    CHOWN_HOME_OPTS="-R" \
    GRANT_SUDO=yes \
    DOCKER_STACKS_JUPYTER_CMD=notebook \ 
    NOTEBOOK_ARGS="./iris_classifier.ipynb" \
    VTSSERVING_HOST=0.0.0.0
EOF
