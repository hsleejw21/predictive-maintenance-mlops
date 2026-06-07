"""MLflow 실험 추적 — 수제 JSON 레지스트리 위에 얹는 업계 표준 추적 레이어.

`run_simulation` 결과를 받아 파라미터·지표·산출물(모델/SPC/PSI)을 한 run으로 기록한다.
**비침습 설계**: mlflow가 없거나 환경변수 `PDM_TRACKING=0`이면 조용히 건너뛰어, 핵심
파이프라인(streamlit/mlflow 불필요 경로)이 절대 깨지지 않는다.

로컬에서 비교 보기:
    mlflow ui          # → http://localhost:5000 에서 run별 지표/아티팩트 비교
"""
import os

from pdm import config
from pdm.config import MODELS_DIR
from pdm.metrics import summarize_drift_performance
from pdm.models.classifier import RF_PARAMS

# 레지스트리에 저장되는 부가 산출물(있으면 함께 아티팩트로 기록)
_ARTIFACTS = [
    "rf_v1.joblib", "rf_v1.json", "rf_v2.joblib", "rf_v2.json",
    "spc_limits.json", "psi_baseline.json",
]


def tracking_enabled() -> bool:
    """PDM_TRACKING=0 이 아니고 mlflow가 설치돼 있을 때만 활성."""
    if os.environ.get("PDM_TRACKING", "1") == "0":
        return False
    try:
        import mlflow  # noqa: F401
        return True
    except ImportError:
        return False


def _ensure_tracking_uri(mlflow):
    """추적 백엔드 확정. 사용자가 지정한 게 있으면 존중하고, 미지정 시(=구식 file 스토어)
    mlflow 3.x가 거부하므로 repo 내 sqlite 로 자동 대체해 즉시 동작하게 한다."""
    if os.environ.get("MLFLOW_TRACKING_URI"):
        return
    uri = mlflow.get_tracking_uri()
    if uri.split(":", 1)[0] in ("sqlite", "http", "https", "postgresql", "mysql", "databricks"):
        return
    mlflow.set_tracking_uri(f"sqlite:///{config.ROOT_DIR / 'mlflow.db'}")


def log_run(result, experiment: str = "predictive-maintenance"):
    """시뮬레이션 결과 한 건을 MLflow run으로 기록하고 run_id를 반환(비활성/실패 시 None).

    persist=True로 모델 산출물이 models/ 에 이미 저장된 상태에서 호출하면 그 파일들을
    아티팩트로 함께 업로드한다. **추적은 부가 기능**이므로 어떤 예외도 핵심 파이프라인을
    중단시키지 않는다(경고만 남기고 None 반환).
    """
    if not tracking_enabled():
        return None
    try:
        return _log_run(result, experiment)
    except Exception as e:   # 추적 실패는 파이프라인을 깨지 않는다
        import sys
        print(f"[tracking] MLflow 기록 생략(추적 비핵심): {e}", file=sys.stderr)
        return None


def _log_run(result, experiment: str):
    import mlflow

    _ensure_tracking_uri(mlflow)
    mlflow.set_experiment(experiment)
    drift_mask = (result["stream"]["phase"] == "drift").values
    perf = summarize_drift_performance(
        result["y_all"], result["pred_live"], result["pred_noretrain"], drift_mask)

    with mlflow.start_run() as run:
        # 파라미터: 하이퍼파라미터 + 시나리오 상수
        mlflow.log_params({f"rf__{k}": v for k, v in RF_PARAMS.items()})
        mlflow.log_params(config.scenario_dict())

        # 지표: 운영(재학습) vs 미재학습 드리프트 구간 성능
        for split in ("live", "noretrain"):
            for k, v in perf[split].items():
                mlflow.log_metric(f"{split}__{k}", float(v))

        # 자동화: 트리거 시점/PSI, 카나리 검증 결과
        if result.get("retrain_t") is not None:
            mlflow.log_metric("retrain_t", int(result["retrain_t"]))
            mlflow.log_metric("trigger_psi", float(result["psi_series"][result["retrain_t"]]))
        prom = result.get("promotion")
        if prom:
            mlflow.log_metric("canary_v1_recall", float(prom["v1"]["fail_recall"]))
            mlflow.log_metric("canary_v2_recall", float(prom["v2"]["fail_recall"]))
            mlflow.set_tag("promoted", str(prom["promoted"]))

        # 아티팩트: 레지스트리 산출물(있는 것만)
        for fname in _ARTIFACTS:
            p = MODELS_DIR / fname
            if p.exists():
                mlflow.log_artifact(str(p), artifact_path="registry")

        # 모델 등록(sklearn flavor) — 트래킹 스토어가 지원하지 않으면 조용히 생략
        try:
            import mlflow.sklearn
            for ver in ("rf_v1", "rf_v2"):
                model = result.get(ver)
                if model is not None:
                    mlflow.sklearn.log_model(model, name=ver.replace("rf_", "model_"))
        except Exception:
            pass

        return run.info.run_id
