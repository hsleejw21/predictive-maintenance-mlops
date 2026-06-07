# 아키텍처 & 모듈 참고자료

> 설계문서 [scenario_design.html](../scenario_design.html) 의 개념을 실제 코드 구조로 매핑한 참고 문서.
> "어디서 무엇을 하는가"를 빠르게 찾기 위한 지도.

---

## 1. 한눈에 보기 — 5개 MLOps 레이어 ↔ 코드 모듈

설계문서 7장은 시스템을 5개 레이어(Data / Model / Serving / Monitoring / Automation)로 나눈다.
코드도 동일하게 `src/pdm/` 아래 서브패키지로 1:1 분리했다.

| MLOps 레이어 | 코드 모듈 | 핵심 함수 | 설계문서 절 |
|---|---|---|---|
| **Data** | [`pdm/data/`](../src/pdm/data) | `load_raw` · `physics_failures`/`make_target` · `build_stream` · `make_features` | 0·5·8장 |
| **Model** | [`pdm/models/`](../src/pdm/models) | `train_rf` · `registry.save_model`/`load_model`/`list_models` | 5·7장 |
| **Serving** | [`pdm/serving/`](../src/pdm/serving) | `run_simulation`(배치 오케스트레이션) · `api.py`(온라인 REST 추론) | 7장 |
| **Monitoring** | [`pdm/monitoring/`](../src/pdm/monitoring) | `fit_spc`/`spc_flags` (Layer 1) · `psi`/`rolling_psi` | 4·7장 |
| **Automation** | [`pdm/automation/`](../src/pdm/automation) | `detect_trigger` · `retrain` | 6·7장 |
| **Tracking** | [`pdm/tracking/`](../src/pdm/tracking) | `log_run`(MLflow 실험 추적, 옵션·비침습) | 7장 |
| (공통) | [`pdm/config.py`](../src/pdm/config.py) · [`pdm/metrics.py`](../src/pdm/metrics.py) · [`pdm/report.py`](../src/pdm/report.py) | 상수 · 평가지표 · 진단출력 | 0·9장 |

### 모듈별 책임

```
src/pdm/
├── config.py              5개 센서·6개 고장모드·드리프트 시나리오 구간/임계치 (전 레이어 공유)
├── data/
│   ├── loader.py          load_raw                — 원본 CSV 적재
│   ├── physics.py         physics_failures        — 공식 물리 규칙으로 HDF/PWF/OSF 라벨 재계산
│   │                      make_target             — 5플래그 → 단일 다중분류 타깃
│   ├── drift.py           build_stream            — 셔플 + 토크 드리프트 주입 + 라벨 재계산 + phase
│   └── features.py        make_features           — Feature Store(학습·서빙 동일 변환): 원값+파생(temp_diff,power)
├── monitoring/
│   ├── spc.py             fit_spc / spc_flags     — Layer1 Shewhart 3σ 관리한계·OOC 플래그
│   └── psi.py             psi / rolling_psi       — 입력 드리프트 지표(토크 분포 변화)
├── models/
│   ├── classifier.py      train_rf / RF_PARAMS    — Layer2 RandomForest(class_weight=balanced)
│   └── registry.py        save/load/list/latest   — 버전별 모델+메타데이터 디스크 영속화
├── automation/
│   └── retrain.py         detect_trigger          — PSI 임계 돌파 시점 = 재학습 트리거
│                          retrain                 — 누적 데이터로 재학습
│                          validate_and_promote    — 카나리 구간에서 v2 vs v1 검증 → 승격/롤백
├── serving/
│   ├── simulation.py      run_simulation          — 위 전부를 한 흐름으로 실행 + 레지스트리 저장
│   └── api.py             FastAPI app             — 온라인 추론(/predict): 레지스트리 모델 로드 +
│                          (make_features 재사용 → train-serve skew 없음) + SPC 플래그
├── tracking/
│   └── mlflow_logger.py   log_run                 — params/metrics/artifacts/모델을 MLflow run으로
│                          (가드: mlflow 미설치 또는 PDM_TRACKING=0 이면 조용히 skip)
├── metrics.py             rolling_recall_accuracy — cumsum 기반 고속 롤링 recall/acc(단일 출처)
│                          rolling_metric          — 롤링 성능(+macroF1)
│                          per_mode_recall         — 고장 모드별 탐지율
│                          summarize_drift_*       — 드리프트 구간 전/후 비교
└── report.py              print_diagnostics       — 콘솔 진단 요약(검증용)
```

---

## 2. 데이터 흐름

```
 data/ai4i2020.csv (10,000행)
        │  load_raw
        ▼
 ┌─ Data 레이어 ──────────────────────────────────────────────┐
 │ build_stream:  셔플 → 토크 드리프트 주입(6000~, +16Nm)      │
 │                → physics 규칙으로 고장 라벨 재계산 → phase   │
 │ make_features: 제품타입 + 5센서 + 파생(temp_diff, power)     │
 └────────────────────────────────────────────────────────────┘
        │  X_all, y_all, stream
        ▼
 기준기간(0~3000)으로 학습 ──► fit_spc(SPC 한계)  +  train_rf(RF v1)
        │
        ▼
 ┌─ Monitoring ───────────────┐     ┌─ Serving(추론) ───────────┐
 │ spc_flags  : 행별 OOC 알람 │     │ RF v1 으로 전체 예측        │
 │ rolling_psi: 토크 PSI 시계열│     │ (pred_noretrain = 반례)     │
 └────────────┬───────────────┘     └─────────────┬─────────────┘
              │ PSI ≥ 0.20 (드리프트 시작 이후)                  │
              ▼  detect_trigger → t*=6700                        │
 ┌─ Automation ───────────────────────────────────┐            │
 │ retrain: 0~t* 누적 데이터로 RF v2 학습            │            │
 │ validate_and_promote: 카나리 [t*:t*+500]에서      │            │
 │   v2 vs v1 비교 → 더 우수하면 승격, 아니면 롤백    │            │
 └────────────┬───────────────────────────────────┘            │
              ▼  승격 시 t* 이후 구간 v2로 교체(롤백 시 v1 유지)   ▼
        pred_live (운영) ◄──────────────────────────────────────┘
        │
        ▼  metrics / report  →  드리프트 구간: recall 78.8%(미재학습) → 90.4%(운영)
```

설계문서 6장의 **SPC × RF 교차검증 매트릭스**와 8장의 **연쇄 반응**(토크↑ → SPC 이탈↑ → PSI↑ →
실제 고장↑ → v1 성능↓ → 재학습 → 회복)이 이 흐름에서 그대로 재현된다.

---

## 3. 모델 레지스트리 & 재현성 (Model 레이어)

`run_simulation(persist=True)` 는 학습 산출물을 `models/` 에 저장한다:

| 파일 | 내용 |
|---|---|
| `rf_v1.joblib` / `rf_v1.json` | 기준기간(0~3000) 학습 모델 + 메타데이터 |
| `rf_v2.joblib` / `rf_v2.json` | 재학습(0~6700) 모델 + 메타데이터(`trigger_t`·`trigger_psi`·`promoted` 포함) |
| `spc_limits.json` | 센서별 center/σ/UCL/LCL — 모니터링 기준 재현 |
| `psi_baseline.json` | PSI 기준 분포(토크 baseline) + window/threshold |

`*.json` 메타데이터는 `trained_at`, `train_range`, `params`(하이퍼파라미터), `classes`,
`feature_names`, `feature_importances`, `metrics`(드리프트 구간 성능 + `canary_validation` 검증 게이트
결과), `scenario`(시나리오 상수)를 담아 **"그때 그 모델"을 그대로 재현/감사**할 수 있게 한다
(설계문서 7장 Model 레이어 요구사항).

```python
from pdm.models import registry
registry.list_models()      # 저장된 모든 버전 메타데이터(학습시점순)
registry.load_model("v2")   # joblib 로드
registry.load_meta("v2")    # 메타데이터 dict
```

---

## 4. 대시보드 구성 (app.py · 4탭)

| 탭 | 내용 | 설계문서 |
|---|---|---|
| **실시간 모니터링 (SPC)** | 5개 센서 관리도(±3σ)·OOC 빨간점·OOC 비율 추이 | 4장 |
| **고장 진단 (RF)** | 모드별 실제 vs 진단 · **특징 중요도**(진단 근거) · **모드별 탐지율** · 최근 진단 내역 | 5·9장 |
| **드리프트 · 자동 재학습** | PSI 시계열 · recall 회복(운영 vs 미재학습) · **검증 게이트(v2 vs v1)** · SPC×RF 교차검증 · **모델 레지스트리** | 6·7장 |
| **비즈니스 가치** | 드리프트 구간 재학습 전/후 정량 효과 · 일반 ML 대비 비교 | 9장 |

상단 상태 바는 현재 시점·구간·운영 모델 버전(v1/v2)·SPC 상태·PSI를 실시간 표시한다.

---

## 5. 실행 방법

```bash
# 0) 환경 (최초 1회)
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                 # 핵심 파이프라인 의존성
pip install -e ".[dashboard]"    # 대시보드까지(streamlit, plotly)

# 1) 핵심 파이프라인 (streamlit 불필요) — 진단 + 모델 영속화
python scripts/run_pipeline.py   # 또는: pdm-run

# 2) 대시보드
streamlit run app.py             # http://localhost:8501
```

참조 환경(macOS, scikit-learn 1.8.0) 실측: **t*=6700(PSI 0.201), 드리프트 구간 운영 recall
90.4% / 미재학습 78.8%, acc 95.9% / 92.1%**. t*·PSI 는 환경에 안정적이나, RandomForest
지표는 OS/CPU·라이브러리 버전에 따라 ±0.5%p 미세 변동이 있다(Linux 컨테이너 ≈ 90.2%).
Docker(`requirements-lock.txt`)는 버전을 고정해 컨테이너 간 동일 값을 보장한다.

---

## 6. 참고 / 알려진 사항

- **하위호환**: 루트 [`pipeline.py`](../pipeline.py) 는 `from pdm import *` 파사드로 축소됨.
  기존 `import pipeline as P` 코드도 계속 동작하나, 새 코드는 `import pdm` 을 직접 쓸 것.
- **app.py**: `import pdm as P` 한 줄만 변경, 4탭 대시보드 로직은 그대로.
- **발표 자료**: `presentation/` 에 모음(`build_ppt.py`·`capture.py`·`*.pptx`·`shots/`). 두 스크립트는
  경로를 스크립트 위치 기준으로 해석하므로 어디서 실행하든 동작한다. `capture.py` 는 기본 포트 `8501`
  을 캡처하며 `DASHBOARD_PORT` 환경변수로 변경할 수 있다.

---

## 7. 운영 보강 — 테스트 / CI / 온라인 서빙 / 실험 추적

핵심 시나리오는 그대로 두고, "MLOps"로서의 신뢰성·운영성을 보강한 레이어들이다.
신규 의존성은 모두 **optional-dependencies 그룹 + 가드 임포트**라 `pip install -e .` 만 한
환경(핵심 파이프라인)은 영향받지 않는다.

| 보강 | 위치 | 핵심 |
|---|---|---|
| **테스트** | [`tests/`](../tests) | 각 레이어 단위 계약 + **회귀 테스트**(버전-강건 불변식: t*=6700, v2 승격, 재학습>미재학습, recall/acc 하한). `run_simulation(persist=False)`·tmp 경로로 실제 산출물 미오염 |
| **CI** | [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) | push/PR마다 `ruff check` + `pytest` (Python 3.10/3.11) + Docker 빌드·헬스체크 |
| **컨테이너** | [`Dockerfile`](../Dockerfile) · [`docker-compose.yml`](../docker-compose.yml) | 핀 고정 의존성([`requirements-lock.txt`](../requirements-lock.txt)) + 빌드 시 모델 내장. `docker compose up` 으로 대시보드(8501)+API(8000) 동시 기동, 컨테이너 간 동일 수치 보장 |
| **온라인 서빙** | [`pdm/serving/api.py`](../src/pdm/serving/api.py) | FastAPI `/predict`(단건·배치) — 레지스트리 모델 로드, **학습과 동일한 `make_features` 재사용**, 응답에 클래스별 확률 + SPC OOC 플래그. `GET /models`로 버전 조회. `pdm-serve` 로 기동 |
| **실험 추적** | [`pdm/tracking/mlflow_logger.py`](../src/pdm/tracking/mlflow_logger.py) | `log_run`이 params·metrics·artifacts·모델을 한 MLflow run으로 기록. CLI(`pdm-run`/`run_pipeline`) 끝에서 가드 호출. 추적 백엔드 미지정 시 repo 내 sqlite로 자동 대체, 실패해도 파이프라인 무중단 |

**설계 원칙**: 수제 JSON 레지스트리(3장)는 경량 재현 스토리로 유지하고, MLflow는 그 위에 얹는
표준 추적 레이어다. 온라인 추론이 학습과 **같은 변환 함수**를 쓰는 것이 train-serve skew를 막는
핵심이며(5장 Data 레이어), 이를 API가 실제로 재사용해 시연한다.
