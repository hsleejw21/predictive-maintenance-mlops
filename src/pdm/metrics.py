"""평가 지표 — 롤링 성능 시계열과 드리프트 구간 전/후 성능 요약(설계문서 9장).

고장 탐지 recall을 가장 중시한다(고장을 놓치는 것이 가장 치명적). 정확도/macro-F1도 함께 본다.
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from pdm.config import MODES


def rolling_recall_accuracy(y_true, y_pred, window=500):
    """롤링 고장탐지 recall / accuracy 시계열 (cumsum 기반 O(n) 고속 계산).

    각 시점 t에서 직전 window개 구간 [t-window:t]의 성능을 매긴다. window 이전 구간과
    분모(실제 고장 수)가 0인 구간은 NaN. 대시보드처럼 전체 스트림을 한 번에 계산할 때
    sklearn 루프(rolling_metric)보다 훨씬 빠르다. 반환: (rec, acc) — 둘 다 길이 n 배열."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    n = len(yt)
    yt_fail = (yt != "Normal").astype(float)            # 실제 고장
    hit = ((yp != "Normal") & (yt != "Normal")).astype(float)  # 고장을 고장으로 탐지
    correct = (yp == yt).astype(float)
    # 누적합(앞에 0 패딩) → 임의 구간 합을 차분으로 O(1)에 구함
    chit = np.concatenate([[0.0], np.cumsum(hit)])
    cfail = np.concatenate([[0.0], np.cumsum(yt_fail)])
    ccor = np.concatenate([[0.0], np.cumsum(correct)])
    rec = np.full(n, np.nan)
    acc = np.full(n, np.nan)
    for t in range(window, n):
        lo = t - window
        fail_sum = cfail[t] - cfail[lo]
        rec[t] = (chit[t] - chit[lo]) / fail_sum if fail_sum > 0 else np.nan
        acc[t] = (ccor[t] - ccor[lo]) / window
    return rec, acc


def rolling_metric(y_true, y_pred, window=500):
    """롤링 정확도 / 고장탐지 recall / macro-F1 시계열."""
    n = len(y_true)
    f1 = np.full(n, np.nan)
    rec, acc = rolling_recall_accuracy(y_true, y_pred, window)
    for t in range(window, n):
        sl = slice(t - window, t)
        f1[t] = f1_score(y_true[sl], y_pred[sl], average="macro",
                         labels=MODES, zero_division=0)
    return acc, rec, f1


def failure_scores(y_true, y_pred):
    """단일 구간 성능: (accuracy, 고장탐지 recall, macro-F1)."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    acc = accuracy_score(yt, yp)
    tf = (yt != "Normal")
    pf = (yp != "Normal")
    rec = (pf & tf).sum() / tf.sum() if tf.sum() else 0.0
    f1 = f1_score(yt, yp, average="macro", labels=MODES, zero_division=0)
    return float(acc), float(rec), float(f1)


def per_mode_recall(y_true, y_pred, modes=None):
    """고장 모드별 탐지율(recall) — 실제 그 모드인 건 중 같은 모드로 맞힌 비율.

    예지보전에서 가장 중요한 지표(고장을 놓치면 치명적). 모드별 실제 건수도 함께 반환.
    반환: {mode: {"recall": float, "support": int}}
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    modes = modes or [m for m in MODES if m != "Normal"]
    out = {}
    for m in modes:
        mask = (yt == m)
        sup = int(mask.sum())
        rec = float((yp[mask] == m).mean()) if sup else float("nan")
        out[m] = {"recall": rec, "support": sup}
    return out


def summarize_drift_performance(y_all, pred_live, pred_noretrain, drift_mask):
    """드리프트 구간에서 운영(재학습) vs 미재학습 성능을 비교 요약."""
    yt = np.asarray(y_all)[drift_mask]
    out = {}
    for name, pred in [("live", pred_live), ("noretrain", pred_noretrain)]:
        acc, rec, f1 = failure_scores(yt, np.asarray(pred)[drift_mask])
        out[name] = {"accuracy": acc, "fail_recall": rec, "macro_f1": f1}
    return out
