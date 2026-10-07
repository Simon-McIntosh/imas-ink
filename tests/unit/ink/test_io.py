"""Saved figures are reproducible: the same figure always yields the same bytes.

A vector format would otherwise stamp the save time into its metadata, and an
SVG would carry random element ids, so every re-render of an unchanged figure
would show up as a diff in a figure committed beside the code that draws it.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from imas_ink import render_to_bytes


def _figure():
    fig, ax = plt.subplots()
    t = np.linspace(0.0, 1.0, 200)
    ax.plot(t, np.sin(2 * np.pi * t))
    ax.fill_between(t, 0.0, np.sin(2 * np.pi * t), alpha=0.3)
    return fig


@pytest.mark.parametrize("fmt", ["svg", "pdf"])
def test_same_figure_saves_to_identical_bytes(fmt):
    """Two saves of one figure are byte-identical."""
    assert render_to_bytes(_figure(), format=fmt) == render_to_bytes(_figure(), format=fmt)


def test_svg_carries_no_save_date():
    """The SVG metadata block holds no date."""
    assert b"<dc:date>" not in render_to_bytes(_figure(), format="svg")
