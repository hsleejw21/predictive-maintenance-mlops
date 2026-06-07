# -*- coding: utf-8 -*-
"""CLI 진입점 — 전체 시뮬레이션을 실행하고 진단 요약을 출력 + 모델 레지스트리 영속화.

streamlit/plotly 없이 numpy/pandas/sklearn/joblib만으로 동작하는 빠른 검증 경로.

실행:
    python scripts/run_pipeline.py
    # 또는 (pip install -e . 후)
    pdm-run
"""
import sys
import warnings
from pathlib import Path

# src 레이아웃을 editable 설치 없이도 import 가능하게(개발 편의)
SRC = Path(__file__).resolve().parents[1] / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pdm.config import MODELS_DIR  # noqa: E402
from pdm.models import registry  # noqa: E402
from pdm.report import print_diagnostics  # noqa: E402
from pdm.serving.simulation import run_simulation  # noqa: E402


def main():
    warnings.filterwarnings("ignore")
    result = run_simulation(persist=True)
    print_diagnostics(result)

    print()
    print("=== 모델 레지스트리 (models/) ===")
    for meta in registry.list_models():
        m = meta.get("metrics", {}).get("drift_region", {})
        rec = m.get("fail_recall")
        rec_str = f"  drift recall={rec*100:5.1f}%" if rec is not None else ""
        print(f"  rf_{meta['version']:3s}  학습범위={meta['train_range']}"
              f"  trained_at={meta['trained_at']}{rec_str}")
    print("산출물 위치 →", MODELS_DIR)

    # 실험 추적(옵션) — mlflow 설치 + PDM_TRACKING!=0 일 때만 기록
    from pdm.tracking import log_run
    run_id = log_run(result)
    if run_id:
        print(f"MLflow run 기록 → {run_id}  ( `mlflow ui` 로 비교 )")


if __name__ == "__main__":
    main()
