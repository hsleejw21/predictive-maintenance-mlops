"""드리프트 주입 — 베어링 마모로 토크가 점점 커지는 상황을 흉내(설계문서 8장).

원본 데이터는 시간 흐름에 따른 변화가 없으므로, 드리프트 구간(6000~)에 토크 시프트를
주입하고 물리 규칙으로 고장 라벨을 재계산한다. 셔플은 각 행이 독립 제품 스냅샷이라는
점에 근거하며 baseline 대표성을 확보한다.
"""
import numpy as np

from pdm.config import (
    DRIFT_FULL,
    DRIFT_START,
    N_BASELINE,
    N_STABLE_END,
    SHUFFLE_SEED,
    TORQUE_SHIFT_MAX,
)
from pdm.data.physics import make_target, physics_failures


def torque_shift_at(idx):
    """행 인덱스별 토크 시프트량(Nm). 6000부터 선형 증가, 8000부터 최대 유지."""
    if idx < DRIFT_START:
        return 0.0
    if idx >= DRIFT_FULL:
        return TORQUE_SHIFT_MAX
    frac = (idx - DRIFT_START) / (DRIFT_FULL - DRIFT_START)
    return TORQUE_SHIFT_MAX * frac


def build_stream(df):
    """셔플 후 스트림으로 사용. 드리프트 구간에 토크 시프트 주입 후
    물리 규칙으로 고장 라벨 재계산. phase / 라벨 / 타깃 컬럼 추가.
    셔플 이유: 원본은 온도 센서가 random-walk라 앞 3000행이 비대표적 →
    baseline 기준 SPC 한계가 정상 구간을 대량 오탐. 각 행은 독립 제품
    스냅샷이므로 셔플은 정당하며, '제품 도착 순서' 시뮬레이션으로 해석."""
    s = df.sample(frac=1, random_state=SHUFFLE_SEED).reset_index(drop=True)

    # 드리프트 주입 (토크 상승) — 물리적으로 베어링 마모/부하 증가를 의미
    shift = np.array([torque_shift_at(i) for i in range(len(s))])
    s["Torque [Nm]"] = s["Torque [Nm]"] + shift
    s["_torque_shift"] = shift

    # 시프트된 센서로 고장 라벨 재계산
    hdf, pwf, osf = physics_failures(s)
    s["HDF"], s["PWF"], s["OSF"] = hdf, pwf, osf
    s["Machine failure"] = ((s[["TWF", "HDF", "PWF", "OSF", "RNF"]].sum(axis=1)) > 0).astype(int)
    s["target"] = make_target(s)

    # phase 표시
    phase = np.where(np.arange(len(s)) < N_BASELINE, "baseline",
             np.where(np.arange(len(s)) < N_STABLE_END, "stable", "drift"))
    s["phase"] = phase
    return s
