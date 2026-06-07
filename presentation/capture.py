# -*- coding: utf-8 -*-
"""대시보드 주요 화면을 PNG로 캡처 (PPT 삽입용).

사전에 대시보드를 띄워둬야 한다:  streamlit run app.py   (기본 포트 8501)
환경변수로 포트 변경 가능:        DASHBOARD_PORT=8533 python presentation/capture.py
"""
import os, time
from pathlib import Path
from playwright.sync_api import sync_playwright

PORT = os.environ.get("DASHBOARD_PORT", "8501")
BASE = f"http://localhost:{PORT}"
OUT = str(Path(__file__).resolve().parent / "shots")   # 스크립트 기준
os.makedirs(OUT, exist_ok=True)

# (filename, t, tab label, note) — tab label must match app.py's st.tabs(...) exactly
SHOTS = [
    ("01_spc_stable",   3500, "Real-time Monitoring (SPC)",        "Stable-phase SPC (in control)"),
    ("02_spc_drift",    9500, "Real-time Monitoring (SPC)",        "Drift-phase SPC (torque excursion)"),
    ("03_rf_diagnosis", 9500, "Failure Diagnosis (Random Forest)", "RF failure diagnosis + feature importance"),
    ("04_drift_retrain",9500, "Drift & Auto-Retraining",           "PSI, recall recovery, validation gate"),
    ("05_business",     9500, "Business Value",                    "Before/after retraining effect"),
]

with sync_playwright() as pw:
    browser = pw.chromium.launch()
    page = browser.new_page(viewport={"width": 1680, "height": 1000},
                            device_scale_factor=2)
    first = True
    for fname, t, tab, desc in SHOTS:
        page.goto(f"{BASE}/?t={t}", wait_until="networkidle")
        if first:
            time.sleep(10)   # 첫 로드: 모델 학습 캐시 대기
            first = False
        else:
            time.sleep(2)
        # 탭 클릭
        try:
            page.get_by_role("tab", name=tab).click()
        except Exception:
            page.click(f"text={tab}")
        time.sleep(4)   # 플롯 렌더 대기
        page.screenshot(path=os.path.join(OUT, fname + ".png"), full_page=False)
        print(f"캡처: {fname}.png  (t={t}, {desc})")
    browser.close()
print("완료 →", os.path.abspath(OUT))
