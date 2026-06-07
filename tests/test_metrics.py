"""평가 지표 — 롤링 helper 정확성 + failure_scores/per_mode_recall 계약."""
import numpy as np

from pdm.metrics import failure_scores, per_mode_recall, rolling_recall_accuracy


def test_rolling_recall_accuracy_basic():
    y = np.array(["Normal", "HDF", "Normal", "HDF", "Normal", "HDF"])
    pred = np.array(["Normal", "HDF", "Normal", "Normal", "Normal", "HDF"])
    rec, acc = rolling_recall_accuracy(y, pred, window=2)
    # window 이전은 NaN
    assert np.isnan(rec[0]) and np.isnan(rec[1])
    # t=4 구간 [2:4]: 실제 고장 1건(idx3) 놓침 → recall 0, 정확도 1/2
    assert rec[4] == 0.0
    assert acc[4] == 0.5


def test_rolling_recall_nan_when_no_failures_in_window():
    y = np.array(["Normal"] * 5)
    pred = np.array(["Normal"] * 5)
    rec, acc = rolling_recall_accuracy(y, pred, window=2)
    assert np.isnan(rec[3])      # 구간에 실제 고장 없음 → recall NaN
    assert acc[3] == 1.0


def test_failure_scores():
    y = np.array(["Normal", "HDF", "HDF", "Normal"])
    pred = np.array(["Normal", "HDF", "Normal", "Normal"])
    acc, rec, f1 = failure_scores(y, pred)
    assert acc == 0.75
    assert rec == 0.5            # 고장 2건 중 1건 탐지


def test_per_mode_recall_support_and_value():
    y = np.array(["HDF", "HDF", "PWF", "Normal"])
    pred = np.array(["HDF", "Normal", "PWF", "Normal"])
    out = per_mode_recall(y, pred, modes=["HDF", "PWF"])
    assert out["HDF"]["support"] == 2
    assert out["HDF"]["recall"] == 0.5
    assert out["PWF"]["recall"] == 1.0
