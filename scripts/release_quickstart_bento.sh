#!/bin/bash

if [ "$#" -eq 1 ]; then
	VTSSERVING_VERSION=$1
else
	VTSSERVING_VERSION=$(python -c "import vtsserving; print(vtsserving.__version__)")
	echo "Releasing with current VtsServing Version $VTSSERVING_VERSION"
fi

GIT_ROOT=$(git rev-parse --show-toplevel)
cd "$GIT_ROOT" || exit 1

docker run --platform=linux/amd64 \
	-v $GIT_ROOT:/vtsserving \
	-v $HOME/.aws:/root/.aws \
	python:3.8-slim /bin/bash -c """\
pip install -U pip
pip install "vtsserving[grpc]==$VTSSERVING_VERSION"
cd /vtsserving/examples/quickstart
pip install -r ./requirements.txt
python train.py
vtsserving build
pip install fs-s3fs
vtsserving export iris_classifier:latest s3://vtsserving.com/quickstart/iris_classifier.vts
"""
