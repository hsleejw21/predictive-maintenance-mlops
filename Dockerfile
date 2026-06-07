# 예지보전 MLOps — 재현 가능한 단일 이미지(대시보드 + 추론 API 공용).
# 핀 고정된 의존성으로 빌드 시 모델을 미리 생성해 이미지에 포함 → 어디서든 동일한
# 정확 수치(t*=6700, drift recall 90.4%)를 재현하고 즉시 추론/시연이 가능하다.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PDM_TRACKING=0

WORKDIR /app

# 1) 핀 고정 의존성 먼저 설치 (소스 변경과 무관하게 레이어 캐시 유지)
COPY requirements-lock.txt ./
RUN pip install -r requirements-lock.txt

# 2) 소스 복사 후 editable 설치(의존성은 1단계에서 이미 고정 설치됨).
#    editable 이라야 패키지의 ROOT_DIR 가 /app 을 가리켜, 빌드 시 생성한 모델을
#    런타임(uvicorn/streamlit)에서도 동일 경로(/app/models)로 찾는다.
COPY . .
RUN pip install --no-deps -e .

# 3) 모델 레지스트리 생성 → 이미지에 포함(추론 즉시 가능 + 수치 재현)
RUN python scripts/run_pipeline.py

EXPOSE 8501 8000

# 기본은 대시보드. compose 에서 서비스별로 command 를 override 한다.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
