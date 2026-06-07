"""Feature Store — 학습·서빙에서 동일하게 적용되는 특징 변환(설계문서 5장·7장 Data).

원값 5센서 + 제품타입 + 파생특징(temp_diff, power)을 만든다. 파생특징은 고장 물리
규칙에 직접 등장하는 값이라 모델 학습을 쉽게 만든다. 학습과 추론이 같은 함수를 쓰는 것이
훈련/서빙 스큐(train-serving skew)를 막는 핵심.
"""
import numpy as np
import pandas as pd

from pdm.config import SENSORS


def make_features(df):
    X = pd.DataFrame(index=df.index)
    X["type_code"] = df["Type"].map({"L": 0, "M": 1, "H": 2})
    for c in SENSORS:
        X[c] = df[c]
    X["temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    X["power"] = df["Torque [Nm]"] * (df["Rotational speed [rpm]"] * 2 * np.pi / 60.0)
    return X
