import shutil
from pathlib import Path

import setuptools

# Remove stale vtsserving.egg-info directory to avoid https://github.com/pypa/pip/issues/5466
git_root = Path(__file__).parent
stale_egg_info = git_root / "vtsserving.egg-info"
if stale_egg_info.exists():
    print(
        """\
Warning: %s exists.

We recently moved the location of source code to follow src-layout convention.
This is to avoid the adhoc REPL imports where if you are at %s, it would import the folder instead of the editable package.

This directory is automatically generated by Python's packaging tools when you install vtsserving in editable mode.
See https://github.com/pypa/pip/issues/5466. I will remove it now.
"""
        % (
            stale_egg_info.absolute(),
            git_root.absolute(),
        )
    )
    shutil.rmtree(stale_egg_info)

setuptools.setup()
