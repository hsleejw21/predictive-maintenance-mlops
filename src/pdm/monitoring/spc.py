"""Layer 1 — SPC (Shewhart 개별값 관리도, 3σ).

설계문서 4장. 기준기간 데이터로 센서별 평균±3σ 관리한계를 만들고, 값이 띠를 벗어나면
(OOC) 즉시 알람한다. 라벨이 없어도 동작하며, 동시에 ML 모델의 입력 드리프트를 감시하는
역할을 겸한다.
"""
import pandas as pd

from pdm.config import SENSORS


def fit_spc(baseline_df):
    """기준기간 데이터로 센서별 관리한계 산출. center ± 3σ."""
    limits = {}
    for c in SENSORS:
        mu = baseline_df[c].mean()
        sigma = baseline_df[c].std(ddof=1)
        limits[c] = {"center": mu, "sigma": sigma,
                     "ucl": mu + 3 * sigma, "lcl": mu - 3 * sigma}
    return limits


def spc_flags(df, limits):
    """행별·센서별 관리이탈(OOC) 여부 + 임의 센서 OOC(=Layer1 알람)."""
    flags = pd.DataFrame(index=df.index)
    for c in SENSORS:
        lim = limits[c]
        flags[c] = (df[c] > lim["ucl"]) | (df[c] < lim["lcl"])
    flags["any_ooc"] = flags[SENSORS].any(axis=1)
    return flags
