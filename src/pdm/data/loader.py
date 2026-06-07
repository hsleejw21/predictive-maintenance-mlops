"""원본 데이터 적재 — UCI AI4I 2020 Predictive Maintenance (10,000건)."""
import pandas as pd

from pdm.config import DATA_PATH


def load_raw(path=None):
    """원본 CSV를 DataFrame으로 로드. path 미지정 시 config.DATA_PATH 사용."""
    return pd.read_csv(path or DATA_PATH)
