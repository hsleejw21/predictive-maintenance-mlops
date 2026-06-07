"""회귀 테스트 — 시나리오의 핵심 불변식을 고정.

정확한 소수 지표(recall 0.904 등)는 scikit-learn/numpy 버전에 따라 RandomForest 예측이
미세하게 달라져 환경 의존적이다(docs/architecture.md 의 수치는 '참조 환경 실측'). 따라서
여기서는 **버전에 강건한 불변식**을 검증한다:
  - 드리프트 트리거 시점 t* (PSI·셔플 기반 → 버전 안정)
  - v2 가 카나리에서 승격되는가
  - 재학습(운영)이 미재학습보다 우수하며, 높은 수준으로 회복되는가
값이 크게 벗어나면(예: 트리거 이동, 재학습이 더 나빠짐) 즉시 실패한다.
"""
import pytest

from pdm.metrics import summarize_drift_performance


def test_retrain_trigger_point(sim):
    # t* 와 PSI 는 토크 드리프트 주입 + np.quantile/histogram 기반이라 버전에 안정적
    assert sim["retrain_t"] == 6700
    assert sim["psi_series"][sim["retrain_t"]] == pytest.approx(0.201, abs=0.01)


def test_canary_promotes_v2(sim):
    prom = sim["promotion"]
    assert prom["promoted"] is True
    # 카나리 구간에서 v2 가 v1 이상 (승격 조건과 일치)
    assert prom["v2"]["fail_recall"] >= prom["v1"]["fail_recall"]
    assert prom["v2"]["fail_recall"] >= 0.80


def test_drift_region_performance(sim):
    drift_mask = (sim["stream"]["phase"] == "drift").values
    perf = summarize_drift_performance(
        sim["y_all"], sim["pred_live"], sim["pred_noretrain"], drift_mask)
    live, nore = perf["live"], perf["noretrain"]
    # 핵심 명제: 자동 재학습이 미재학습보다 우수하며 높은 수준으로 회복
    assert live["fail_recall"] >= nore["fail_recall"]
    assert live["accuracy"] >= nore["accuracy"]
    assert live["fail_recall"] >= 0.85      # 운영 recall 회복 (참조 실측 0.904)
    assert live["accuracy"] >= 0.93         # 운영 accuracy (참조 실측 0.959)
    assert nore["fail_recall"] >= 0.70      # 미재학습 하한 (참조 실측 0.788)
