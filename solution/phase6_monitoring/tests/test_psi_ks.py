"""PSI / KS primitives - the drift-detection maths."""
from __future__ import annotations

import numpy as np
import pandas as pd

from capstone import monitoring as mon


def test_identical_numeric_distributions_have_near_zero_psi_and_ks():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(size=5000))
    b = pd.Series(rng.normal(size=5000))
    assert mon.psi(a, b) < 0.05
    ks_stat, ks_p = mon.ks(a, b)
    assert ks_stat < 0.05
    assert ks_p > 0.05


def test_shifted_numeric_distribution_registers_significant_psi():
    rng = np.random.default_rng(1)
    ref = pd.Series(rng.normal(0, 1, size=5000))
    shifted = pd.Series(rng.normal(1.5, 1, size=5000))
    value = mon.psi(ref, shifted)
    assert value > mon.PSI_SIGNIFICANT
    assert mon.psi_band(value) == "significant"
    assert mon.ks(ref, shifted)[0] > 0.3


def test_categorical_psi_tracks_share_change():
    ref = pd.Series(["A"] * 700 + ["B"] * 200 + ["C"] * 100)
    same = pd.Series(["A"] * 350 + ["B"] * 100 + ["C"] * 50)
    moved = pd.Series(["A"] * 200 + ["B"] * 200 + ["C"] * 600)
    assert mon.psi(ref, same) < 0.02
    assert mon.psi(ref, moved) > mon.PSI_SIGNIFICANT


def test_psi_is_bounded_when_a_bucket_is_empty_on_one_side():
    ref = pd.Series(list(range(100)))
    disjoint = pd.Series(list(range(1000, 1100)))
    value = mon.psi(ref, disjoint)
    assert np.isfinite(value)
    assert value < 20        # Laplace-smoothed, not a blow-up


def test_psi_band_edges():
    assert mon.psi_band(0.0) == "stable"
    assert mon.psi_band(0.10) == "moderate"
    assert mon.psi_band(0.2499) == "moderate"
    assert mon.psi_band(0.25) == "significant"
