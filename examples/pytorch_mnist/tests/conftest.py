# pylint: disable=redefined-outer-name

import typing as t

import pytest

from vtsserving.testing.server import host_vts


def pytest_configure(config):  # pylint: disable=unused-argument
    import os
    import sys
    import subprocess

    cmd = f"{sys.executable} {os.path.join(os.getcwd(), 'train.py')} --k-folds=0"
    subprocess.run(cmd, shell=True, check=True)


@pytest.fixture(scope="session")
def host() -> t.Generator[str, None, None]:
    import vtsserving

    vtsserving.build("service:svc")

    with host_vts(vts="pytorch_mnist_demo:latest") as host:
        yield host
