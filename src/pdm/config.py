"""프로젝트 전역 상수 — 센서/고장모드 정의 및 드리프트 시나리오 파라미터.

설계문서(scenario_design.html) 0장(센서·고장모드)과 8장(드리프트 시나리오 구간)에 대응.
모든 레이어가 이 값을 공유하므로, 시나리오를 바꾸려면 여기만 수정하면 된다.
"""
from pathlib import Path

# ── 경로 ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[2]   # 저장소 루트
DATA_PATH = ROOT_DIR / "data" / "ai4i2020.csv"
MODELS_DIR = ROOT_DIR / "models"                 # 모델 레지스트리 산출물 위치

# ── 5개 센서 (Layer 1 SPC 감시 대상) ────────────────────────────────────────
SENSORS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
SENSOR_KR = {
    "Air temperature [K]": "공기 온도",
    "Process temperature [K]": "공정 온도",
    "Rotational speed [rpm]": "회전 속도",
    "Torque [Nm]": "토크",
    "Tool wear [min]": "공구 마모",
}

# ── 고장 모드 (Layer 2 RF 다중분류 타깃) ────────────────────────────────────
MODES = ["Normal", "TWF", "HDF", "PWF", "OSF", "RNF"]
MODE_KR = {
    "Normal": "정상",
    "TWF": "공구마모 고장",
    "HDF": "열발산 고장",
    "PWF": "전력 고장",
    "OSF": "과부하 고장",
    "RNF": "무작위 고장",
}
# 다중 라벨 충돌 시 우선순위 (위쪽이 우선)
MODE_PRIORITY = ["OSF", "PWF", "HDF", "TWF", "RNF"]

# ── 드리프트 시나리오 구간 (행 인덱스 기준) ─────────────────────────────────
N_BASELINE = 3000      # 0 ~ 2999  : 기준기간 (RF 학습 + SPC 한계 + PSI 기준)
N_STABLE_END = 6000    # 3000~5999 : 안정 가동
DRIFT_START = 6000     # 6000~     : 드리프트 시작
DRIFT_FULL = 8000      # 8000~     : 드리프트 최대치 유지
TORQUE_SHIFT_MAX = 16.0  # Nm, 베어링 마모로 인한 부하 증가 시뮬레이션
PSI_THRESHOLD = 0.20   # 드리프트 판정 임계치
PSI_WINDOW = 500       # 롤링 PSI 윈도우
CANARY_WINDOW = 500    # 재학습 검증(카나리) 윈도우: 트리거 직후 구간에서 신·구 모델 비교
SHUFFLE_SEED = 42      # 각 행은 독립 제품 스냅샷 → 셔플해 baseline 대표성 확보
                       # (원본은 온도 센서가 random-walk라 앞 구간이 비대표적)


def scenario_dict():
    """재현성 메타데이터로 저장할 시나리오 파라미터 묶음."""
    return {
        "N_BASELINE": N_BASELINE,
        "N_STABLE_END": N_STABLE_END,
        "DRIFT_START": DRIFT_START,
        "DRIFT_FULL": DRIFT_FULL,
        "TORQUE_SHIFT_MAX": TORQUE_SHIFT_MAX,
        "PSI_THRESHOLD": PSI_THRESHOLD,
        "PSI_WINDOW": PSI_WINDOW,
        "CANARY_WINDOW": CANARY_WINDOW,
        "SHUFFLE_SEED": SHUFFLE_SEED,
    }
