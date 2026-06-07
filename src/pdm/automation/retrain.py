"""자동 재학습 — PSI 임계 돌파 시점을 트리거로 새 모델을 학습/검증/등록(설계문서 7장 Automation).

데이터 누적 전략: 트리거 시점까지 본 전체 데이터로 재학습한다. 희귀 고장 모드 예시를
잃지 않으면서 드리프트 구간을 함께 학습 → 모든 지표 개선. 학습한 v2는 곧바로 배포하지
않고, 트리거 직후 카나리 구간에서 기존 v1과 비교 검증해 더 나을 때만 승격한다(아니면 롤백).
"""
import numpy as np

from pdm.config import CANARY_WINDOW, DRIFT_START, PSI_THRESHOLD
from pdm.metrics import failure_scores
from pdm.models.classifier import train_rf


def detect_trigger(psi_series, drift_start=DRIFT_START, threshold=PSI_THRESHOLD):
    """드리프트 시작 이후 PSI가 임계치를 처음 넘는 시점 = 재학습 트리거(없으면 None)."""
    psi_series = np.asarray(psi_series)
    cand = np.where((np.arange(len(psi_series)) >= drift_start) &
                    (psi_series >= threshold))[0]
    return int(cand[0]) if len(cand) else None


def retrain(X, y, upto_t, version=None, seed=42, registry=None,
            metrics=None, feature_names=None, scenario=None):
    """트리거 시점(upto_t)까지의 누적 데이터로 RF 재학습. registry가 주어지면 저장한다."""
    model = train_rf(X.iloc[:upto_t], y.iloc[:upto_t], seed=seed)
    if registry is not None and version is not None:
        registry.save_model(
            model, version,
            metrics=metrics, train_range=(0, upto_t),
            feature_names=feature_names, scenario=scenario,
        )
    return model


def validate_and_promote(rf_old, rf_new, X, y, trigger_t, n_total,
                         window=CANARY_WINDOW, metric="fail_recall"):
    """카나리(섀도) 검증 — 트리거 직후 구간에서 신·구 모델을 공정 비교 후 승격 판정.

    검증 구간 [trigger_t : trigger_t+window] 는 두 모델 모두 학습에 쓰지 않은 미래 구간이라
    공정한 out-of-sample 비교가 된다(현실의 '지연 라벨로 backtest' 패턴). 신 모델 지표가
    구 모델 이상이면 승격(promoted=True), 미달이면 롤백(False).

    반환: {"v1": {...}, "v2": {...}, "promoted": bool, "metric": str, "window": [lo, hi]}
    """
    lo, hi = trigger_t, min(trigger_t + window, n_total)
    yt = y.iloc[lo:hi].values
    Xv = X.iloc[lo:hi]

    def score(model):
        acc, rec, f1 = failure_scores(yt, model.predict(Xv))
        return {"accuracy": acc, "fail_recall": rec, "macro_f1": f1}

    s_old, s_new = score(rf_old), score(rf_new)
    promoted = s_new[metric] >= s_old[metric]
    return {"v1": s_old, "v2": s_new, "promoted": promoted,
            "metric": metric, "window": [lo, hi]}
