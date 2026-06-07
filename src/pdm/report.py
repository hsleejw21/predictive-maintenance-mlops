"""진단 리포트 — 시뮬레이션 결과를 콘솔에 요약 출력(설계문서 8·9장 검증용).

기존 pipeline.py의 __main__ 진단 블록을 옮긴 것. CLI(scripts/run_pipeline.py)와
콘솔 스크립트(pdm-run)에서 공유한다.
"""
from sklearn.metrics import accuracy_score, f1_score

from pdm.config import MODES


def print_diagnostics(r):
    s = r["stream"]
    print("=== 시뮬레이션 진단 ===")
    print("스트림 길이:", len(s))
    print("phase별 행수:", s["phase"].value_counts().to_dict())
    print("재학습 트리거 시점 t* =", r["retrain_t"],
          f"(PSI={r['psi_series'][r['retrain_t']]:.3f})" if r["retrain_t"] else "")
    prom = r.get("promotion")
    if prom:
        gate = "승격(배포)" if prom["promoted"] else "롤백(v1 유지)"
        print(f"검증 게이트(카나리 {prom['window']}): "
              f"v1 recall={prom['v1']['fail_recall']:.3f} vs "
              f"v2 recall={prom['v2']['fail_recall']:.3f} → {gate}")
    print()
    print("타깃 분포(전체):", s["target"].value_counts().to_dict())
    print()
    # phase별 고장률
    for ph in ["baseline", "stable", "drift"]:
        sub = s[s["phase"] == ph]
        fr = (sub["target"] != "Normal").mean()
        print(f"  {ph:9s} 고장률 {fr*100:5.2f}%   토크평균 {sub['Torque [Nm]'].mean():.1f}")
    print()
    # 드리프트 구간 성능: 운영 vs 미재학습
    drift_mask = (s["phase"] == "drift").values
    yt = r["y_all"][drift_mask]
    print("드리프트 구간 성능 (운영 재학습 vs 미재학습):")
    for name, pred in [("운영(재학습)", r["pred_live"]), ("미재학습", r["pred_noretrain"])]:
        pv = pred[drift_mask]
        acc = accuracy_score(yt, pv)
        ytf = (yt != "Normal"); pvf = (pv != "Normal")
        rec = (pvf & ytf).sum() / ytf.sum() if ytf.sum() else 0
        f1 = f1_score(yt, pv, average="macro", labels=MODES, zero_division=0)
        print(f"  {name:12s}  acc={acc:.3f}  고장recall={rec:.3f}  macroF1={f1:.3f}")
    print()
    # SPC 알람률
    for ph in ["baseline", "stable", "drift"]:
        m = (s["phase"] == ph).values
        print(f"  {ph:9s} SPC 알람률 {r['flags']['any_ooc'].values[m].mean()*100:5.2f}%")
