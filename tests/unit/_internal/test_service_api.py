# pylint: disable=redefined-outer-name
# type: ignore[no-untyped-def]

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

import vtsserving
from vtsserving.io import Text
from vtsserving.exceptions import VtsServingException

if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture


svc = vtsserving.Service(name="general_features_routes")


@svc.api(input=Text(), output=Text(), route="api/v1/test_without_prefix")
async def without_prefix(obj: str) -> int:
    return 1


@svc.api(input=Text(), output=Text(), route="/api/v1/with_prefix")
async def with_prefix(obj: str) -> int:
    return 1


def test_routes_prefix():
    assert all(i.path.startswith("/") for i in svc.asgi_app.routes)


def test_invalid_api_naming():
    svc = vtsserving.Service(name="invalid_api_naming")

    @svc.api(name="similar", input=Text(), output=Text())
    def r1(obj: str) -> int:
        return 1

    with pytest.raises(VtsServingException) as excinfo:

        @svc.api(name="similar", input=Text(), output=Text())
        def r2(obj: str) -> int:
            return 1

    assert "is already defined in Service" in str(excinfo.value)


def test_get_valid_service_name(caplog: LogCaptureFixture):
    from vtsserving._internal.service.service import get_valid_service_name

    with caplog.at_level(logging.WARNING):
        get_valid_service_name("NOT_LOWER")
    assert "to lowercase:" in caplog.text
