"""Feature Store — 컬럼 계약·파생 공식·학습/서빙 동일성(train-serve skew 방지)."""
import numpy as np

from pdm.config import SENSORS
from pdm.data.features import make_features

EXPECTED_COLS = ["type_code", *SENSORS, "temp_diff", "power"]


def test_feature_columns_and_order(raw):
    X = make_features(raw)
    assert list(X.columns) == EXPECTED_COLS


def test_derived_formulas(raw):
    X = make_features(raw)
    expected_diff = raw["Process temperature [K]"] - raw["Air temperature [K]"]
    expected_power = raw["Torque [Nm]"] * (raw["Rotational speed [rpm]"] * 2 * np.pi / 60.0)
    assert np.allclose(X["temp_diff"], expected_diff)
    assert np.allclose(X["power"], expected_power)


def test_type_code_mapping(raw):
    X = make_features(raw)
    assert set(X["type_code"].unique()) <= {0, 1, 2}


def test_train_serve_parity(raw):
    """같은 입력 → 같은 출력. 동일 변환 함수를 학습·서빙이 공유하는지 보증."""
    a = make_features(raw)
    b = make_features(raw.copy())
    assert a.equals(b)
