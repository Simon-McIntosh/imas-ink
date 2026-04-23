"""Integration test: render the ITER coilset from real IMAS data.

Gated on the ``iter_imas_data`` fixture — skips cleanly when
``IMAS_INK_ITER_DATA`` is not set.
"""

from __future__ import annotations

import pytest


@pytest.mark.render
@pytest.mark.imas_data
def test_render_iter_coilset(iter_imas_data, tmp_path):
    """Render the ITER coilset and verify a non-trivial PNG is produced."""
    from imas_ink.three_d.scene import render_coilset

    uri = f"imas:hdf5?path={iter_imas_data}/"
    outfile = tmp_path / "iter.png"

    render_coilset(uri, outfile=outfile, off_screen=True)

    assert outfile.exists()
    size = outfile.stat().st_size
    assert size > 1024, f"PNG too small ({size} bytes) — likely blank"
