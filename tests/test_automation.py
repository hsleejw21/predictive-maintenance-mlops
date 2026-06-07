"""Automation — 드리프트 트리거 탐지 + 카나리 검증 게이트(승격/롤백) 분기."""
import numpy as np
import pandas as pd

from pdm.automation import detect_trigger, validate_and_promote


def test_detect_trigger_first_crossing_after_drift_start():
    psi = np.zeros(100)
    psi[60:] = 0.3                      # 60부터 임계 초과
    assert detect_trigger(psi, drift_start=50, threshold=0.2) == 60


def test_detect_trigger_ignores_crossing_before_drift_start():
    psi = np.zeros(100)
    psi[10] = 0.5                       # drift_start 이전 스파이크 → 무시
    psi[70:] = 0.3
    assert detect_trigger(psi, drift_start=50, threshold=0.2) == 70


def test_detect_trigger_none_when_no_crossing():
    assert detect_trigger(np.zeros(100), drift_start=50, threshold=0.2) is None


class _StubModel:
    """predict가 고정 라벨 배열을 돌려주는 더미(검증 게이트 분기만 시험)."""
    def __init__(self, preds):
        self._preds = np.asarray(preds)

    def predict(self, X):
        return self._preds[:len(X)]


def _make_xy(n):
    X = pd.DataFrame({"f": np.arange(n)})
    y = pd.Series(["Normal", "HDF"] * (n // 2))
    return X, y


def test_validate_and_promote_promotes_better_model():
    X, y = _make_xy(20)
    lo = 0
    yt = y.iloc[lo:10].values
    old = _StubModel(["Normal"] * 10)          # 고장 전부 놓침 → recall 0
    new = _StubModel(yt)                        # 완벽 예측 → recall 1
    res = validate_and_promote(old, new, X, y, trigger_t=0, n_total=20, window=10)
    assert res["promoted"] is True
    assert res["v2"]["fail_recall"] >= res["v1"]["fail_recall"]


def test_validate_and_promote_rolls_back_worse_model():
    X, y = _make_xy(20)
    yt = y.iloc[0:10].values
    old = _StubModel(yt)                        # 완벽
    new = _StubModel(["Normal"] * 10)           # 더 나쁨
    res = validate_and_promote(old, new, X, y, trigger_t=0, n_total=20, window=10)
    assert res["promoted"] is False
