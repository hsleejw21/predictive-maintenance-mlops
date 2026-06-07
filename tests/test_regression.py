"""회귀 테스트 — 문서화된 시나리오 불변값을 고정(seed=42 결정론).

docs/architecture.md의 '검증 기대값'이 코드 변경에도 유지되는지 자동 보증한다.
값이 바뀌면 의도된 변경인지 즉시 드러난다.
"""
import pytest

from pdm.metrics import summarize_drift_performance


def test_retrain_trigger_point(sim):
    assert sim["retrain_t"] == 6700
    assert sim["psi_series"][sim["retrain_t"]] == pytest.approx(0.2009, abs=2e-3)


def test_canary_promotes_v2(sim):
    prom = sim["promotion"]
    assert prom["promoted"] is True
    assert prom["v1"]["fail_recall"] == pytest.approx(0.7692, abs=2e-3)
    assert prom["v2"]["fail_recall"] == pytest.approx(0.8846, abs=2e-3)


def test_drift_region_performance(sim):
    drift_mask = (sim["stream"]["phase"] == "drift").values
    perf = summarize_drift_performance(
        sim["y_all"], sim["pred_live"], sim["pred_noretrain"], drift_mask)
    # 운영(재학습) — 회복된 성능
    assert perf["live"]["fail_recall"] == pytest.approx(0.9042, abs=2e-3)
    assert perf["live"]["accuracy"] == pytest.approx(0.9587, abs=2e-3)
    # 미재학습 — 반례(성능 저하)
    assert perf["noretrain"]["fail_recall"] == pytest.approx(0.7881, abs=2e-3)
    assert perf["noretrain"]["accuracy"] == pytest.approx(0.9213, abs=2e-3)
    # 재학습이 항상 더 낫거나 같아야 함
    assert perf["live"]["fail_recall"] >= perf["noretrain"]["fail_recall"]
