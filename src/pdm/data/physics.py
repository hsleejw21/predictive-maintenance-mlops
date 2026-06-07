"""고장 규칙 — AI4I 2020 공식 문서의 결정론적 물리 규칙으로 라벨 재계산.

드리프트 주입으로 센서 값을 바꿔도, 같은 규칙으로 고장 여부를 정직하게 다시 매길 수
있다(설계문서 8장 "라벨 100% 재현"). HDF/PWF/OSF는 규칙으로 재계산하고,
TWF/RNF는 원본 무작위 라벨을 유지한다.
"""
import numpy as np
import pandas as pd

from pdm.config import MODE_PRIORITY


def physics_failures(df):
    """AI4I 2020 공식 문서의 결정론적 고장 규칙으로 HDF/PWF/OSF 재계산.
    (검증 결과 원본 라벨을 100% 재현함) TWF/RNF는 원본 무작위 라벨 유지."""
    air = df["Air temperature [K]"]
    proc = df["Process temperature [K]"]
    rpm = df["Rotational speed [rpm]"]
    torque = df["Torque [Nm]"]
    wear = df["Tool wear [min]"]
    typ = df["Type"]

    hdf = ((proc - air).abs() < 8.6) & (rpm < 1380)
    power = torque * (rpm * 2 * np.pi / 60.0)      # 기계적 일률 [W]
    pwf = (power < 3500) | (power > 9000)
    osf_thr = typ.map({"L": 11000, "M": 12000, "H": 13000})
    osf = (wear * torque) > osf_thr
    return hdf.astype(int), pwf.astype(int), osf.astype(int)


def make_target(df):
    """5개 고장 플래그 → 단일 라벨 다중분류 타깃(Normal + 5모드)."""
    target = pd.Series(["Normal"] * len(df), index=df.index)
    for mode in reversed(MODE_PRIORITY):     # 우선순위 낮은 것부터 덮어씀
        target[df[mode] == 1] = mode
    # TWF/RNF 는 우선순위 마지막 → 위 루프에 포함되도록 처리
    for mode in ["RNF", "TWF"]:
        mask = (df[mode] == 1) & (target == "Normal")
        target[mask] = mode
    return target
