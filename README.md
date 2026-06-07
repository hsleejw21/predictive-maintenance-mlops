# 설비 고장 모드 분류 + SPC 관리도 이중 레이어 MLOps

UCI **AI4I 2020 Predictive Maintenance** 데이터로,
**SPC 관리도(3σ) + Random Forest 다중분류**의 이중 레이어와 **자동 재학습**을 시연하는 프로젝트입니다.
"고장을 빨리 잡고(SPC), 무슨 고장인지 진단하며(RF), 드리프트가 오면 스스로 다시 학습한다(MLOps)."

---

## 파일 구성

코드는 **5개 MLOps 레이어**(Data / Model / Serving / Monitoring / Automation)에 1:1로 매핑되는
`src/pdm/` 패키지로 구성된다. 레이어↔모듈 매핑·데이터 흐름은 **[docs/architecture.md](docs/architecture.md)** 참고.

| 경로 | 설명 |
|---|---|
| `src/pdm/data/` | **Data** — 적재 · 고장 물리규칙 · 드리프트 주입 · Feature Store |
| `src/pdm/models/` | **Model** — RandomForest 분류기 + 버전 레지스트리(`registry.py`) |
| `src/pdm/serving/` | **Serving** — 전체 시뮬레이션 오케스트레이션(`run_simulation`) |
| `src/pdm/monitoring/` | **Monitoring** — SPC 3σ(Layer 1) + PSI 드리프트 |
| `src/pdm/automation/` | **Automation** — 드리프트 트리거 + 자동 재학습 |
| `src/pdm/config.py` · `metrics.py` · `report.py` | 공통 상수 · 평가지표 · 진단출력 |
| `src/pdm/serving/api.py` | **Serving(온라인)** — FastAPI 실시간 추론 REST API(`/predict`) |
| `src/pdm/tracking/` | **Tracking** — MLflow 실험 추적(옵션·비침습, `mlflow_logger.py`) |
| `app.py` | **실시간 대시보드**(Streamlit) — 4탭: SPC / RF 진단 / 드리프트·재학습 / 비즈니스 가치 |
| `scripts/run_pipeline.py` | **CLI** — 시뮬레이션 실행 + 모델 영속화 + 진단 요약(streamlit 불필요) |
| `pipeline.py` | 하위호환 파사드(`from pdm import *`) — 기존 `import pipeline` 코드용 |
| `tests/` | **pytest 스위트** — 물리규칙·피처·모니터링·자동화·레지스트리·API + 회귀(불변값) |
| `.github/workflows/ci.yml` | **CI** — push/PR마다 ruff 린트 + pytest + Docker 빌드 (Python 3.10/3.11) |
| `Dockerfile` · `docker-compose.yml` | **컨테이너** — 핀 고정 의존성 + 모델 내장, `docker compose up` 으로 대시보드+API |
| `requirements-lock.txt` | Docker용 핀 고정 의존성(정확 수치 재현) |
| `models/` | 모델 레지스트리 산출물 — **생성물(.gitignore 처리)**, `python scripts/run_pipeline.py` 로 생성 |
| `data/ai4i2020.csv` | 데이터셋 (10,000건 · 5센서 · 5고장모드) |
| `docs/architecture.md` | **아키텍처·모듈 참고자료** |
| `scenario_design.html` | 시나리오 설계 문서 (입문자용, 브라우저로 열기) |
| `presentation/` | 발표 자료 모음 — `build_ppt.py`·`capture.py`·`*.pptx`(16장)·`shots/`(캡처) |

---

## 실행 방법

### 가장 간단한 길 — Docker (권장)

파이썬·가상환경 설치 없이 **한 줄로 대시보드 + 추론 API**를 띄웁니다. 의존성이 핀 고정돼
있고 모델이 이미지에 미리 구워져, 어느 PC에서 돌려도 **동일한 결과**가 재현됩니다.

```bash
docker compose up --build
#  → 대시보드   http://localhost:8501
#  → 추론 API   http://localhost:8000/docs
```

> 재현성 메모: 동일 버전이라도 OS/CPU의 BLAS·부동소수점 차이로 RandomForest 지표가
> ±0.5%p 수준에서 미세하게 달라질 수 있습니다(예: macOS 90.4% ↔ Linux 컨테이너 90.2%).
> Docker는 **버전을 고정**해 컨테이너 사용자끼리 같은 값을 보장합니다.

### 직접 설치 (pip)

```bash
# 1) (최초 1회) 가상환경 + 패키지 설치
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                 # 핵심 파이프라인
pip install -e ".[dashboard]"    # 대시보드까지(streamlit, plotly)
# 선택: pip install -e ".[serving]"  (추론 API) / ".[tracking]" (MLflow) / ".[dev]" (테스트·린트)

# 2-a) 핵심 파이프라인 실행 (진단 + 모델 영속화, streamlit 불필요)
python scripts/run_pipeline.py   # 또는: pdm-run

# 2-b) 대시보드 실행
streamlit run app.py
```

### 추가 진입점

```bash
# 테스트 + 린트 (CI와 동일)
pip install -e ".[dev,serving]"
ruff check . && pytest

# 실시간 추론 REST API (먼저 run_pipeline 으로 모델을 만들어 둘 것)
pip install -e ".[serving]"
pdm-serve                        # http://localhost:8000  (문서: /docs)
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
     -d '{"Type":"L","Air temperature [K]":300,"Process temperature [K]":308,
          "Rotational speed [rpm]":1300,"Torque [Nm]":60,"Tool wear [min]":220}'

# MLflow 실험 추적 (옵션·비침습 — 미설치면 자동 skip)
pip install -e ".[tracking]"
python scripts/run_pipeline.py   # run 기록됨
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000 에서 비교
```

실행하면 브라우저가 자동으로 열립니다.

### 대시보드 주소:  http://localhost:8501

> 슬라이더(왼쪽 사이드바)를 드래그해 시점을 **기준기간 → 안정 → 드리프트**로 옮기면
> SPC 이상 감지 → PSI 상승 → 자동 재학습(t*=6,700) → 성능 회복을 직접 볼 수 있습니다.
> '자동 재생' 토글로 흐름을 자동 재생할 수도 있습니다.

---

## 주소 공유에 대한 안내 (꼭 읽어주세요)

`http://localhost:8501` 은 **앱을 실행한 본인 PC에서만** 열립니다.
다른 사람이 같은 주소를 입력하면 자기 PC를 보게 됩니다. → **팀원 각자 위 명령으로 직접 실행**하세요.

함께(라이브로) 보고 싶다면:
- **같은 와이파이**: 실행 시 터미널에 표시되는 `Network URL`(예: `http://10.x.x.x:8501`)을 공유 (방화벽 허용 필요)
- **외부에서 링크로 공유**: Streamlit Community Cloud 배포(영구 URL) 또는 cloudflared/ngrok 터널(임시 URL) 필요

---

## PPT / 스크린샷 다시 만들기 (선택)

발표 관련 파일은 모두 `presentation/` 에 모여 있습니다.

```bash
pip install -e ".[slides]"          # python-pptx, playwright
python presentation/build_ppt.py    # 발표 PPT(16장) 재생성 → presentation/*.pptx
# (대시보드가 8501에서 실행 중일 때) 스크린샷 재캡처:
playwright install chromium
python presentation/capture.py      # → presentation/shots/*.png
```

---

## 핵심 결과 (시뮬레이션 실측)

드리프트 구간(6,000~10,000행)에서:

| 지표 | 재학습 안 함 | 이중 레이어 + 재학습 |
|---|---|---|
| 고장 탐지 recall | 78.8 % | **90.4 %** |
| 정확도 | 92.1 % | **95.9 %** |

→ 베어링 마모로 토크가 상승(+16Nm)하자 SPC·PSI가 드리프트를 감지하고 자동 재학습이 성능을 회복시킴.

> 위 수치는 참조 환경(macOS, scikit-learn 1.8.0) 실측이며, OS/CPU·라이브러리 버전에 따라
> ±0.5%p 미세 변동이 있을 수 있습니다(Linux 컨테이너 ≈ 90.2%). 핵심 명제(자동 재학습이
> 미재학습보다 우수·회복)는 환경과 무관하게 회귀 테스트로 검증됩니다.
