"""Tests for extract_signal_traces — dynamic signals read straight from an IDS.

The IDS objects are built in memory with ``imas.IDSFactory``: no file, no
corpus data.  These tests pin the contract that a caller hands imas-ink the
IDS and the unindexed data-dictionary path, and imas-ink names the signal and
its unit itself.
"""

from __future__ import annotations

import imas
import numpy as np
import pytest
from numpy.testing import assert_allclose

from imas_ink import extract_signal_traces


@pytest.fixture
def factory() -> imas.IDSFactory:
    return imas.IDSFactory()


def _pf_ids_with_three_coils(factory: imas.IDSFactory):
    """A pf_active with CS1, CS2 fed and CS3 unfed, under homogeneous time 1."""
    pf = factory.pf_active()
    pf.ids_properties.homogeneous_time = 1
    pf.time = np.array([0.0, 0.1, 0.2])
    pf.coil.resize(3)
    pf.coil[0].name = "CS1"
    pf.coil[0].current.data = np.array([1.0, 2.0, 3.0])
    pf.coil[1].name = "CS2"
    pf.coil[1].current.data = np.array([4.0, 5.0, 6.0])
    pf.coil[2].name = "CS3"  # unfed — current.data left empty
    return pf


def test_pf_active_coils_yield_one_series_per_fed_coil(factory):
    pf = _pf_ids_with_three_coils(factory)

    series = extract_signal_traces(pf, "coil/current")

    assert len(series) == 2
    assert {s.ylabel for s in series} == {"CS1", "CS2"}
    assert all(s.units == "A" for s in series)
    for s in series:
        assert_allclose(s.time, pf.time)


def test_magnetics_ip_yields_one_series_named_by_the_path_component(factory):
    mag = factory.magnetics()
    mag.ids_properties.homogeneous_time = 1
    mag.time = np.array([0.0, 1.0, 2.0])
    mag.ip.resize(1)
    mag.ip[0].data = np.array([1.0e6, 1.1e6, 1.2e6])

    series = extract_signal_traces(mag, "ip")

    assert len(series) == 1
    assert series[0].ylabel == "ip"
    assert series[0].units == "A"
    assert_allclose(series[0].time, mag.time)


def test_flux_loops_take_each_nodes_own_time_under_homogeneous_time_zero(factory):
    mag = factory.magnetics()
    mag.ids_properties.homogeneous_time = 0
    # A distinct IDS time is present; under homogeneous time 0 the series must
    # take each flux loop's own flux.time instead of it.
    mag.time = np.array([100.0, 200.0, 300.0])
    mag.flux_loop.resize(2)
    flux_times = {
        "FL1": np.array([0.0, 0.5, 1.0]),
        "FL2": np.array([0.0, 0.25]),
    }
    for i, (name, t) in enumerate(flux_times.items()):
        mag.flux_loop[i].name = name
        mag.flux_loop[i].flux.time = t
        mag.flux_loop[i].flux.data = np.full(t.shape, float(i + 1))

    series = extract_signal_traces(mag, "flux_loop/flux")

    assert len(series) == 2
    by_label = {s.ylabel: s for s in series}
    assert set(by_label) == {"FL1", "FL2"}
    for name, t in flux_times.items():
        assert_allclose(by_label[name].time, t)
        assert by_label[name].units == "Wb"


def test_homogeneous_time_two_raises_naming_the_path(factory):
    mag = factory.magnetics()
    mag.ids_properties.homogeneous_time = 2
    mag.ip.resize(1)
    mag.ip[0].data = np.array([1.0])

    with pytest.raises(ValueError, match="ip"):
        extract_signal_traces(mag, "ip")


def test_indexed_path_raises_naming_the_unindexed_form(factory):
    pf = _pf_ids_with_three_coils(factory)

    with pytest.raises(ValueError) as excinfo:
        extract_signal_traces(pf, "coil[0]/current")

    assert "coil/current" in str(excinfo.value)


def test_no_series_carries_a_label(factory):
    series = extract_signal_traces(_pf_ids_with_three_coils(factory), "coil/current")
    assert series  # sanity: the fixture produces series at all
    assert all(s.label == "" for s in series)
