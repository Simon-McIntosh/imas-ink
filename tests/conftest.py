"""pytest configuration and session-scoped fixtures for imas-ink tests.

Tests requiring real IMAS HDF5 data use the ``iter_imas_data`` fixture.
They are automatically skipped when the data path is unavailable.

Set either environment variable to point at a directory containing ITER
IMAS HDF5 test data (e.g. the ``tests/data/imas/ITER/135013`` directory
from the efit++ repository):

    export IMAS_INK_ITER_DATA=/path/to/efitpp/tests/data/imas/ITER/135013
    # or, for smooth migration from efit++:
    export EFIT_INK_ITER_DATA=/path/to/efitpp/tests/data/imas/ITER/135013
"""

from __future__ import annotations

import os
import pathlib

import pytest


@pytest.fixture(scope="session")
def iter_imas_data() -> pathlib.Path:
    """Return the path to the ITER 135013 IMAS test data directory.

    Checks ``IMAS_INK_ITER_DATA`` first, then ``EFIT_INK_ITER_DATA`` for
    smooth migration from the efit++ test suite.  Skips the test if
    neither variable is set or the path does not exist.
    """
    for env in ("IMAS_INK_ITER_DATA", "EFIT_INK_ITER_DATA"):
        val = os.environ.get(env)
        if val and pathlib.Path(val).exists():
            return pathlib.Path(val)
    pytest.skip("No ITER IMAS test data available (set IMAS_INK_ITER_DATA)")
