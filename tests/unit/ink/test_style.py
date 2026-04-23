"""Tests for efit.ink.style — InkStyle and DEFAULT_STYLE."""

from __future__ import annotations

import dataclasses

import pytest

from imas_ink.style import DEFAULT_STYLE, InkStyle


class TestInkStyleDefaults:
    """Verify DEFAULT_STYLE key values match the specification."""

    def test_is_inkstyle(self):
        assert isinstance(DEFAULT_STYLE, InkStyle)

    def test_flux_color(self):
        assert DEFAULT_STYLE.flux_color == "#3366cc"

    def test_flux_linewidth(self):
        assert DEFAULT_STYLE.flux_linewidth == 0.7

    def test_flux_linestyle(self):
        assert DEFAULT_STYLE.flux_linestyle == "solid"

    def test_flux_n_levels(self):
        assert DEFAULT_STYLE.flux_n_levels == 6

    def test_sep_color(self):
        assert DEFAULT_STYLE.sep_color == "#cc0000"

    def test_sep_linewidth(self):
        assert DEFAULT_STYLE.sep_linewidth == 1.5

    def test_wall_color(self):
        assert DEFAULT_STYLE.wall_color == "#000000"

    def test_wall_linewidth(self):
        assert DEFAULT_STYLE.wall_linewidth == 1.0

    def test_coil_edgecolor(self):
        assert DEFAULT_STYLE.coil_edgecolor == "#888888"

    def test_coil_facecolor(self):
        assert DEFAULT_STYLE.coil_facecolor == "none"

    def test_axis_marker(self):
        assert DEFAULT_STYLE.axis_marker == "."

    def test_axis_color(self):
        assert DEFAULT_STYLE.axis_color == "#cc0000"

    def test_xpt_marker(self):
        assert DEFAULT_STYLE.xpt_marker == "x"

    def test_xpt_color(self):
        assert DEFAULT_STYLE.xpt_color == "#cc0000"

    def test_label_fontsize(self):
        assert DEFAULT_STYLE.label_fontsize == 8.0

    def test_figure_facecolor(self):
        assert DEFAULT_STYLE.figure_facecolor == "white"

    def test_figure_dpi(self):
        assert DEFAULT_STYLE.figure_dpi == 120

    def test_zorder_ordering(self):
        """Z-order values should be in logical render order."""
        assert DEFAULT_STYLE.zorder_flux < DEFAULT_STYLE.zorder_coils
        assert DEFAULT_STYLE.zorder_coils < DEFAULT_STYLE.zorder_wall
        assert DEFAULT_STYLE.zorder_wall < DEFAULT_STYLE.zorder_sep
        assert DEFAULT_STYLE.zorder_sep < DEFAULT_STYLE.zorder_markers
        assert DEFAULT_STYLE.zorder_markers < DEFAULT_STYLE.zorder_label


class TestInkStyleFrozen:
    """InkStyle is a frozen dataclass — immutable after creation."""

    def test_cannot_set_attribute(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            DEFAULT_STYLE.flux_color = "#ff0000"

    def test_cannot_delete_attribute(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            del DEFAULT_STYLE.flux_color


class TestInkStyleCustom:
    """Custom InkStyle instances."""

    def test_override_single_field(self):
        custom = InkStyle(flux_color="#ff0000")
        assert custom.flux_color == "#ff0000"
        # Other fields remain default
        assert custom.sep_color == "#cc0000"

    def test_override_multiple_fields(self):
        custom = InkStyle(flux_color="#ff0000", sep_linewidth=3.0, figure_dpi=300)
        assert custom.flux_color == "#ff0000"
        assert custom.sep_linewidth == 3.0
        assert custom.figure_dpi == 300

    def test_replace(self):
        """dataclasses.replace creates a new instance with overrides."""
        custom = dataclasses.replace(DEFAULT_STYLE, sep_color="#00cc00")
        assert custom.sep_color == "#00cc00"
        assert custom.flux_color == "#3366cc"  # unchanged
        # Original not affected
        assert DEFAULT_STYLE.sep_color == "#cc0000"

    def test_custom_is_also_frozen(self):
        custom = InkStyle(flux_color="#aabbcc")
        with pytest.raises(dataclasses.FrozenInstanceError):
            custom.flux_color = "#ffffff"

    def test_label_bbox_default(self):
        """label_bbox should be a dict with expected keys."""
        bbox = DEFAULT_STYLE.label_bbox
        assert isinstance(bbox, dict)
        assert "facecolor" in bbox
        assert "alpha" in bbox
        assert bbox["facecolor"] == "white"
        assert bbox["alpha"] == 0.9
