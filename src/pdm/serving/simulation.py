"""전체 시뮬레이션 오케스트레이션 — 5개 MLOps 레이어를 한 흐름으로 실행.

Data → (SPC/RF 학습) → Serving(추론) → Monitoring(PSI) → Automation(재학습) 순으로
스트림 전체를 사전 계산해 대시보드가 즉시 인덱싱하도록 반환한다. 학습된 v1/v2와
SPC 한계·PSI 기준 분포는 모델 레지스트리(models/)에 영속화된다.
"""

from pdm import config
from pdm.automation import detect_trigger, retrain, validate_and_promote
from pdm.config import (
    DRIFT_START,
    N_BASELINE,
    PSI_THRESHOLD,
    PSI_WINDOW,
)
from pdm.data import build_stream, load_raw, make_features
from pdm.metrics import summarize_drift_performance
from pdm.models import registry, train_rf
from pdm.monitoring import fit_spc, rolling_psi, spc_flags


def run_simulation(persist=True):
    """스트림 전체를 사전 계산해 대시보드가 즉시 인덱싱하도록 반환.

    persist=True면 학습한 모델·SPC 한계·PSI 기준을 models/ 레지스트리에 저장한다.
    """
    # ── Data 레이어 ──────────────────────────────────────────────────────
    raw = load_raw()
    stream = build_stream(raw)
    X_all = make_features(stream)
    y_all = stream["target"]
    feature_names = list(X_all.columns)

    # ── 기준기간으로 Layer1(SPC) + Layer2(RF v1) 학습 ────────────────────
    base = stream.iloc[:N_BASELINE]
    limits = fit_spc(base)
    rf_v1 = train_rf(X_all.iloc[:N_BASELINE], y_all.iloc[:N_BASELINE])

    # ── Monitoring: 전체 SPC 플래그 + 롤링 PSI(토크 기준) ────────────────
    flags = spc_flags(stream, limits)
    base_torque = base["Torque [Nm]"].values
    psi_series = rolling_psi(base_torque, stream["Torque [Nm]"].values, PSI_WINDOW)

    # ── Automation: PSI 임계 돌파 → 재학습 트리거 ────────────────────────
    retrain_t = detect_trigger(psi_series, DRIFT_START, PSI_THRESHOLD)

    # 재학습: 트리거 시점까지 본 전체 데이터로 RF v2 학습(데이터 누적 전략)
    rf_v2 = None
    promotion = None
    if retrain_t is not None:
        rf_v2 = retrain(X_all, y_all, retrain_t)
        # 검증 게이트: 카나리 구간에서 v2를 v1과 비교 → 더 나을 때만 승격(아니면 롤백)
        promotion = validate_and_promote(
            rf_v1, rf_v2, X_all, y_all, retrain_t, len(stream))

    # ── 예측: 운영(재학습 반영) vs 미재학습 반례 ────────────────────────
    pred_v1 = rf_v1.predict(X_all)          # 끝까지 v1만 (재학습 안 한 경우)
    proba_v1 = rf_v1.predict_proba(X_all).max(axis=1)
    pred_live = pred_v1.copy()              # 운영(검증 통과 시 트리거 후 v2 적용)
    proba_live = proba_v1.copy()
    if rf_v2 is not None and promotion["promoted"]:
        seg = slice(retrain_t, len(stream))
        pred_live[seg] = rf_v2.predict(X_all.iloc[seg])
        proba_live[seg] = rf_v2.predict_proba(X_all.iloc[seg]).max(axis=1)

    result = {
        "stream": stream,
        "X_all": X_all,
        "y_all": y_all.values,
        "limits": limits,
        "flags": flags,
        "psi_series": psi_series,
        "retrain_t": retrain_t,
        "pred_live": pred_live,        # 운영 모델 예측 (재학습 반영)
        "proba_live": proba_live,
        "pred_noretrain": pred_v1,     # 재학습 안 했을 때 예측 (반례)
        "rf_v1": rf_v1,
        "rf_v2": rf_v2,
        "promotion": promotion,        # 검증 게이트 결과 (v1 vs v2, 승격 여부)
        "feature_names": feature_names,
    }

    if persist:
        _persist(result)
    return result


def _persist(result):
    """학습 산출물을 모델 레지스트리에 저장(설계문서 Model 레이어의 버전 보관)."""
    scenario = config.scenario_dict()
    drift_mask = (result["stream"]["phase"] == "drift").values
    perf = summarize_drift_performance(
        result["y_all"], result["pred_live"], result["pred_noretrain"], drift_mask)

    # RF v1 (기준기간 학습) / v2 (재학습) 저장
    registry.save_model(
        result["rf_v1"], "v1",
        metrics={"drift_region": perf["noretrain"]},   # v1만 쓰면 = 미재학습 성능
        train_range=(0, N_BASELINE),
        feature_names=result["feature_names"], scenario=scenario,
    )
    if result["rf_v2"] is not None:
        registry.save_model(
            result["rf_v2"], "v2",
            metrics={"drift_region": perf["live"],
                     "canary_validation": result["promotion"]},   # 검증 게이트 기록
            train_range=(0, result["retrain_t"]),
            feature_names=result["feature_names"], scenario=scenario,
            extra={"trigger_t": result["retrain_t"],
                   "trigger_psi": float(result["psi_series"][result["retrain_t"]]),
                   "promoted": result["promotion"]["promoted"]},
        )

    # SPC 한계 / PSI 기준 분포도 함께 저장 → 모니터링 기준 재현
    registry.save_json_artifact("spc_limits", {
        c: {k: float(v) for k, v in lim.items()}
        for c, lim in result["limits"].items()
    })
    registry.save_json_artifact("psi_baseline", {
        "feature": "Torque [Nm]",
        "window": PSI_WINDOW,
        "threshold": PSI_THRESHOLD,
        "baseline_values": [float(v) for v in
                            result["stream"]["Torque [Nm]"].values[:N_BASELINE]],
    })


def _cli():
    """`pdm-run` 콘솔 스크립트 진입점 — scripts/run_pipeline.py 와 동일한 진단 출력."""
    import warnings

    from pdm.report import print_diagnostics
    from pdm.tracking import log_run
    warnings.filterwarnings("ignore")
    result = run_simulation()
    print_diagnostics(result)
    run_id = log_run(result)   # 실험 추적(옵션) — mlflow 미설치 시 자동 skip
    if run_id:
        print(f"\nMLflow run 기록 → {run_id}  ( `mlflow ui` 로 비교 )")
