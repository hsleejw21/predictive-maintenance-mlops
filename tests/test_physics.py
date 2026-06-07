"""물리 고장 규칙 — 원본 라벨 100% 재현 + make_target 우선순위 검증."""
import pandas as pd

from pdm.data.physics import make_target, physics_failures


def test_physics_reproduces_original_labels(raw):
    """결정론적 물리 규칙이 원본 CSV의 HDF/PWF/OSF 라벨을 100% 재현해야 한다.

    드리프트 주입 후에도 같은 규칙으로 라벨을 정직하게 다시 매길 수 있다는 전제의 근거.
    """
    hdf, pwf, osf = physics_failures(raw)
    for name, calc in [("HDF", hdf), ("PWF", pwf), ("OSF", osf)]:
        assert (calc.values == raw[name].values).all(), f"{name} 라벨 재현 실패"


def test_make_target_priority():
    """다중 라벨 충돌 시 OSF가 PWF보다 우선(MODE_PRIORITY)."""
    df = pd.DataFrame({
        "TWF": [0, 0, 0],
        "HDF": [0, 0, 0],
        "PWF": [1, 0, 1],   # 0행: PWF+OSF 충돌 → OSF 우선
        "OSF": [1, 0, 0],
        "RNF": [0, 1, 0],   # 1행: RNF만
    })
    target = make_target(df)
    assert target.tolist() == ["OSF", "RNF", "PWF"]


def test_make_target_normal_when_no_failure():
    df = pd.DataFrame({c: [0] for c in ["TWF", "HDF", "PWF", "OSF", "RNF"]})
    assert make_target(df).tolist() == ["Normal"]
