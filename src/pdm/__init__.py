"""pdm — 설비 고장 모드 분류 + SPC 이중 레이어 MLOps 패키지.

5개 MLOps 레이어를 서브패키지로 분리한다:
  pdm.data        — Data        (적재·고장규칙·드리프트·Feature Store)
  pdm.models      — Model       (RandomForest + 버전 레지스트리)
  pdm.serving     — Serving     (전체 시뮬레이션 오케스트레이션)
  pdm.monitoring  — Monitoring  (SPC 3σ + PSI)
  pdm.automation  — Automation  (드리프트 트리거 + 자동 재학습)

대시보드(app.py)는 이 패키지를 `import pdm as P`로 불러 쓴다. 아래에서 config 상수와
주요 함수를 평면적으로 재노출해 기존 호출부(P.SENSORS, P.run_simulation 등)를 그대로 유지한다.
"""
from pdm.automation import detect_trigger, retrain, validate_and_promote
from pdm.config import (
    CANARY_WINDOW,
    DATA_PATH,
    DRIFT_FULL,
    DRIFT_START,
    MODE_KR,
    MODE_PRIORITY,
    MODELS_DIR,
    MODES,
    N_BASELINE,
    N_STABLE_END,
    PSI_THRESHOLD,
    PSI_WINDOW,
    SENSOR_KR,
    SENSORS,
    SHUFFLE_SEED,
    TORQUE_SHIFT_MAX,
)
from pdm.data import (
    build_stream,
    load_raw,
    make_features,
    make_target,
    physics_failures,
    torque_shift_at,
)
from pdm.metrics import (
    failure_scores,
    per_mode_recall,
    rolling_metric,
    rolling_recall_accuracy,
    summarize_drift_performance,
)
from pdm.models import registry, train_rf
from pdm.monitoring import fit_spc, psi, rolling_psi, spc_flags
from pdm.serving import run_simulation

__all__ = [
    # config 상수
    "SENSORS", "SENSOR_KR", "MODES", "MODE_KR", "MODE_PRIORITY",
    "N_BASELINE", "N_STABLE_END", "DRIFT_START", "DRIFT_FULL",
    "TORQUE_SHIFT_MAX", "PSI_THRESHOLD", "PSI_WINDOW", "CANARY_WINDOW", "SHUFFLE_SEED",
    "DATA_PATH", "MODELS_DIR",
    # 함수
    "load_raw", "physics_failures", "make_target",
    "torque_shift_at", "build_stream", "make_features",
    "fit_spc", "spc_flags", "psi", "rolling_psi",
    "train_rf", "registry",
    "detect_trigger", "retrain", "validate_and_promote",
    "run_simulation",
    "rolling_metric", "rolling_recall_accuracy", "failure_scores",
    "per_mode_recall", "summarize_drift_performance",
]
