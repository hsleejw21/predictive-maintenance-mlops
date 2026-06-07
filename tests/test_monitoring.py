"""Monitoring — PSI 드리프트 지표 + SPC 관리한계/플래그 계약."""
import numpy as np
import pandas as pd

from pdm.config import SENSORS
from pdm.monitoring import fit_spc, psi, spc_flags


def test_psi_zero_for_identical_distribution():
    rng = np.random.default_rng(0)
    base = rng.normal(40, 10, 4000)
    assert psi(base, base) < 0.01


def test_psi_rises_for_shifted_distribution():
    rng = np.random.default_rng(0)
    base = rng.normal(40, 10, 4000)
    shifted = base + 16          # 시나리오의 토크 시프트 크기
    assert psi(base, shifted) > 0.2


def test_psi_guard_small_sample():
    assert psi(np.arange(100), np.arange(5)) == 0.0


def test_fit_spc_limits_are_mean_plus_minus_3sigma():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({c: rng.normal(0, 1, 3000) for c in SENSORS})
    limits = fit_spc(df)
    for c in SENSORS:
        lim = limits[c]
        assert np.isclose(lim["ucl"], lim["center"] + 3 * lim["sigma"])
        assert np.isclose(lim["lcl"], lim["center"] - 3 * lim["sigma"])


def test_spc_flags_any_ooc():
    df = pd.DataFrame({c: [0.0, 0.0] for c in SENSORS})
    # 한 센서만 관리한계를 크게 벗어나도록 첫 행을 밀어둔다
    df.loc[0, SENSORS[0]] = 100.0
    limits = {c: {"center": 0.0, "sigma": 1.0, "ucl": 3.0, "lcl": -3.0} for c in SENSORS}
    flags = spc_flags(df, limits)
    assert bool(flags["any_ooc"].iloc[0]) is True
    assert bool(flags["any_ooc"].iloc[1]) is False
