"""MLflow 추적 — 비활성 가드 + (설치 시) file 스토어에 run 1건 기록 확인."""
import pytest

from pdm.tracking import log_run, tracking_enabled


def test_disabled_via_env(monkeypatch):
    """PDM_TRACKING=0 이면 mlflow 설치 여부와 무관하게 비활성."""
    monkeypatch.setenv("PDM_TRACKING", "0")
    assert tracking_enabled() is False
    assert log_run({}) is None       # 비활성이면 result를 건드리지 않고 None


def test_log_run_records_a_run(tmp_path, monkeypatch, sim):
    """mlflow 설치 시: 임시 sqlite 스토어에 run 1건이 지표와 함께 기록되어야 한다."""
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("PDM_TRACKING", "1")
    monkeypatch.chdir(tmp_path)      # 아티팩트(mlartifacts/)를 임시 디렉터리로 격리
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")

    run_id = log_run(sim, experiment="test-pdm")
    assert run_id is not None

    run = mlflow.get_run(run_id)
    # 핵심 지표가 기록됐는지 확인
    assert run.data.metrics["live__fail_recall"] == pytest.approx(0.9042, abs=2e-3)
    assert run.data.metrics["retrain_t"] == 6700
    assert "rf__n_estimators" in run.data.params
