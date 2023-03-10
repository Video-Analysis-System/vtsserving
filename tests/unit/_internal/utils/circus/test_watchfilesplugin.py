from __future__ import annotations

import os
import typing as t
import logging
from typing import TYPE_CHECKING
from pathlib import Path
from unittest import skipUnless
from unittest.mock import patch

import pytest
from circus.tests.support import TestCircus

from vtsserving._internal.utils.pkg import source_locations
from vtsserving._internal.utils.circus.watchfilesplugin import ServiceReloaderPlugin

if TYPE_CHECKING:
    from unittest import TestCase
    from unittest.mock import MagicMock

    from watchfiles.main import FileChange


def requires_watchfiles(test_case: t.Type[TestCase]) -> t.Callable[..., t.Any]:
    return skipUnless(
        source_locations("watchfiles") is not None,
        "Requires 'watchfiles' to be installed.",
    )(test_case)


@pytest.mark.usefixtures("reload_directory")
@requires_watchfiles
class TestServiceReloaderPlugin(TestCircus):
    reload_directory: Path

    def setUp(self) -> None:
        super().setUp()
        self.plugin_kwargs: dict[str, t.Any] = {
            "vts_identifier": ".",
            "working_dir": self.reload_directory.__fspath__(),
            "vtsserving_home": self.reload_directory.__fspath__(),
        }
        self.mock_vtsserving_install_from_source()
        self.mock_vtsserving_component_context()
        self.mock_vtsserving_server_logging()

    def test_logging_info(self) -> None:
        with self.assertLogs("vtsserving", level=logging.INFO) as log:
            self.make_plugin(ServiceReloaderPlugin, **self.plugin_kwargs)
            self.assertIn("adding source root", log.output[0])
            self.assertIn("Watching directories", log.output[1])

    def test_reloader_params_is_required(self) -> None:
        self.assertRaises(AssertionError, self.make_plugin, ServiceReloaderPlugin)  # type: ignore (unfinished circus type)

    def mock_vtsserving_install_from_source(self) -> MagicMock:
        patcher = patch(
            "vtsserving._internal.utils.circus.watchfilesplugin.is_pypi_installed_vtsserving"
        )
        mock = patcher.start()
        mock.return_value = False
        self.addCleanup(patcher.stop)

    def mock_vtsserving_component_context(self) -> MagicMock:
        from vtsserving._internal.context import _ComponentContext

        # prevent error from double setting component name
        _ComponentContext.component_name = None

    def mock_vtsserving_server_logging(self) -> MagicMock:
        patcher = patch(
            "vtsserving._internal.utils.circus.watchfilesplugin.configure_server_logging"
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        import vtsserving._internal.log as log

        log.configure_server_logging = lambda: None

    def setup_watch_mock(self, watch_return: set[FileChange]) -> MagicMock:
        # changes: 1 -> added, 2 -> modified, 3 -> deleted
        patcher = patch(f"{ServiceReloaderPlugin.__module__}.watch")
        watch_mock = patcher.start()
        self.addCleanup(patcher.stop)
        # return value {(<Change.added: 1>, 'path/to/file.txt'), ...}
        watch_mock.return_value = iter([watch_return])  # type: ignore
        return watch_mock

    def setup_call_mock(self, watcher_name: str) -> MagicMock:
        patcher = patch.object(ServiceReloaderPlugin, "call")
        call_mock = patcher.start()
        self.addCleanup(patcher.stop)
        call_mock.side_effect = [
            {"watchers": [watcher_name]},
            {"options": {"cmd": watcher_name}},
            None,
        ]
        return call_mock

    def test_look_after_trigger_restart(self) -> None:
        from watchfiles.main import Change

        file = self.reload_directory.joinpath("file.txt").__fspath__()

        call_mock = self.setup_call_mock(watcher_name="reloader")
        self.setup_watch_mock(watch_return={(Change(1), file)})
        plugin = self.make_plugin(ServiceReloaderPlugin, **self.plugin_kwargs)
        Path(file).touch()

        with self.assertLogs("vtsserving", level=logging.WARNING) as log:
            plugin.look_after()
            call_mock.assert_called_with("restart", name="*")
            self.assertIn("ADDED", log.output[0])

    def test_look_after_trigger_restart_on_deletion(self):
        from watchfiles.main import Change

        file = self.reload_directory.joinpath("train.py").__fspath__()

        call_mock = self.setup_call_mock(watcher_name="reloader")
        self.setup_watch_mock(watch_return={(Change(3), file)})
        plugin = self.make_plugin(ServiceReloaderPlugin, **self.plugin_kwargs)
        os.remove(file)

        with self.assertLogs("vtsserving", level=logging.WARNING) as log:
            plugin.look_after()
            call_mock.assert_called_with("restart", name="*")
            self.assertIn("DELETED", log.output[0])
