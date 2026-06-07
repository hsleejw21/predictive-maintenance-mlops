# -*- coding: utf-8 -*-
"""[호환 파사드] 기존 `import pipeline as P` 호출을 위해 유지.

핵심 로직은 src/pdm/ 패키지의 5개 MLOps 레이어로 이전되었다(docs/architecture.md 참고).
새 코드는 `import pdm` 을 직접 사용할 것. 이 모듈은 하위호환을 위해 공개 API를 재노출만 한다.
"""
from pdm import *  # noqa: F401,F403  (config 상수 + 주요 함수)
from pdm import config  # noqa: F401  (P.config 접근 대비)

if __name__ == "__main__":
    # 기존 `python pipeline.py` 진단 출력과 동일 (이제 scripts/run_pipeline.py 권장)
    import warnings

    from pdm.report import print_diagnostics
    warnings.filterwarnings("ignore")
    print_diagnostics(run_simulation())   # noqa: F405
