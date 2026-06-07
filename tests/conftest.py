"""공유 픽스처 — 무거운 산출물(원본 데이터·전체 시뮬레이션)을 세션 단위로 1회만 계산."""
import warnings

import pytest

warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def raw():
    import pdm as P
    return P.load_raw()


@pytest.fixture(scope="session")
def sim():
    """persist=False — 실제 models/ 산출물을 건드리지 않고 시뮬레이션 결과만 반환."""
    import pdm as P
    return P.run_simulation(persist=False)
