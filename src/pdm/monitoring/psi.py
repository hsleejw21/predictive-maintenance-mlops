"""PSI(Population Stability Index) — 기준 분포 대비 현재 분포 변화량(설계문서 7장).

PSI<0.1 안정, 0.1~0.2 주의, >0.2 유의한 드리프트로 본다. 우리는 0.2를 재학습 신호로 쓴다.
SPC가 "즉시 이상"을 잡는다면 PSI는 "분포가 통째로 이동했는가"를 한 숫자로 잡는다.
"""
import numpy as np

from pdm.config import PSI_WINDOW


def psi(expected, actual, bins=10):
    """expected(기준) 대비 actual(현재) 분포 차이. >0.2면 유의한 드리프트."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if len(actual) < 20:
        return 0.0
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e_perc = np.histogram(expected, edges)[0] / len(expected)
    a_perc = np.histogram(actual, edges)[0] / len(actual)
    e_perc = np.clip(e_perc, 1e-4, None)
    a_perc = np.clip(a_perc, 1e-4, None)
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))


def rolling_psi(expected, series, window=PSI_WINDOW):
    """기준 분포(expected) 대비 series의 롤링 PSI 시계열을 계산.

    각 시점 t에서 최근 window개 구간 [t-window+1 .. t]의 PSI를 매긴다.
    드리프트가 시작되면 이 값이 서서히 상승해 임계(0.2)를 돌파한다.
    """
    series = np.asarray(series, dtype=float)
    out = np.zeros(len(series))
    for t in range(len(series)):
        lo = max(0, t - window + 1)
        out[t] = psi(expected, series[lo:t + 1])
    return out
