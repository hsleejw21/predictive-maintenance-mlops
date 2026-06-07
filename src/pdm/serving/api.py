"""실시간 추론 REST API (Serving 레이어의 온라인 경로).

배치 시뮬레이션(run_simulation)과 달리, 레지스트리에 저장된 모델을 로드해 단건/배치 센서
입력을 즉시 진단한다. 학습과 동일한 변환 함수(make_features)를 그대로 재사용하므로
train-serve skew가 없다 — 이것이 Feature Store를 함수로 분리한 핵심 이유다.

실행:
    pdm-serve                         # 또는: uvicorn pdm.serving.api:app --reload
    curl -X POST localhost:8000/predict -H 'content-type: application/json' \\
         -d '{"Type":"L","Air temperature [K]":300,"Process temperature [K]":310,
              "Rotational speed [rpm]":1400,"Torque [Nm]":45,"Tool wear [min]":120}'
"""
import json
from typing import Literal, Optional, Union

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from pdm.config import MODELS_DIR, SENSORS
from pdm.data import make_features
from pdm.models import registry

app = FastAPI(
    title="Predictive Maintenance — Inference API",
    description="레지스트리 모델로 설비 고장 모드를 실시간 진단(SPC 플래그 포함).",
    version="1.0.0",
)

# 로드한 모델을 버전별로 캐시 (프로세스 생애주기 동안 재사용)
_MODEL_CACHE: dict = {}
_LIMITS_CACHE: list = []   # [limits] — SPC 한계(있으면 1개)


class SensorReading(BaseModel):
    """단일 제품 스냅샷. 원본 CSV 컬럼명을 alias로 받아 make_features와 호환."""
    model_config = ConfigDict(populate_by_name=True)

    type: Literal["L", "M", "H"] = Field(alias="Type")
    air_temperature: float = Field(alias="Air temperature [K]")
    process_temperature: float = Field(alias="Process temperature [K]")
    rotational_speed: float = Field(alias="Rotational speed [rpm]")
    torque: float = Field(alias="Torque [Nm]")
    tool_wear: float = Field(alias="Tool wear [min]")

    def as_row(self) -> dict:
        """make_features가 기대하는 원본 컬럼명 dict로 변환."""
        return {
            "Type": self.type,
            "Air temperature [K]": self.air_temperature,
            "Process temperature [K]": self.process_temperature,
            "Rotational speed [rpm]": self.rotational_speed,
            "Torque [Nm]": self.torque,
            "Tool wear [min]": self.tool_wear,
        }


def _resolve_version(version: Optional[str]) -> str:
    """요청 버전 또는 레지스트리 최신 버전을 확정. 없으면 503."""
    if version is None:
        meta = registry.latest()
        if meta is None:
            raise HTTPException(
                status_code=503,
                detail="저장된 모델이 없습니다. 먼저 `python scripts/run_pipeline.py` 를 실행하세요.",
            )
        return meta["version"]
    return version


def _get_model(version: str):
    if version not in _MODEL_CACHE:
        try:
            _MODEL_CACHE[version] = registry.load_model(version)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"모델 버전 '{version}' 없음")
    return _MODEL_CACHE[version]


def _get_limits():
    """SPC 한계(spc_limits.json). 없으면 None — SPC 플래그는 생략된다."""
    if not _LIMITS_CACHE:
        path = MODELS_DIR / "spc_limits.json"
        _LIMITS_CACHE.append(json.loads(path.read_text(encoding="utf-8"))
                             if path.exists() else None)
    return _LIMITS_CACHE[0]


def _spc_check(row: pd.Series, limits) -> dict:
    """저장된 관리한계로 센서별 OOC 여부 + any_ooc 산출(한계 없으면 빈 dict)."""
    if not limits:
        return {"available": False}
    per = {}
    for c in SENSORS:
        if c in limits:
            lim = limits[c]
            per[c] = bool(row[c] > lim["ucl"] or row[c] < lim["lcl"])
    return {"available": True, "any_ooc": any(per.values()), "by_sensor": per}


@app.get("/health")
def health():
    """서비스 상태 + 서빙 가능한 모델 유무."""
    meta = registry.latest()
    return {
        "status": "ok",
        "model_available": meta is not None,
        "latest_version": meta["version"] if meta else None,
    }


@app.get("/models")
def models():
    """레지스트리에 저장된 모든 버전 메타데이터(학습시점순)."""
    return registry.list_models()


@app.get("/models/{version}")
def model_meta(version: str):
    try:
        return registry.load_meta(version)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"모델 버전 '{version}' 없음")


@app.post("/predict")
def predict(
    payload: Union[SensorReading, list[SensorReading]],
    version: Optional[str] = None,
):
    """단건 또는 배치 센서 입력 → 고장 모드 진단 + 확신도 + 클래스별 확률 + SPC 플래그.

    `version` 쿼리로 특정 모델 버전을 지정할 수 있다(기본: 레지스트리 최신 = 보통 v2).
    """
    readings = payload if isinstance(payload, list) else [payload]
    if not readings:
        raise HTTPException(status_code=422, detail="입력이 비어 있습니다.")

    ver = _resolve_version(version)
    model = _get_model(ver)
    limits = _get_limits()

    df = pd.DataFrame([r.as_row() for r in readings])
    X = make_features(df)
    preds = model.predict(X)
    proba = model.predict_proba(X)
    classes = list(model.classes_)

    out = []
    for i in range(len(readings)):
        probs = {c: float(p) for c, p in zip(classes, proba[i])}
        out.append({
            "predicted_mode": str(preds[i]),
            "confidence": float(max(proba[i])),
            "is_failure": preds[i] != "Normal",
            "probabilities": probs,
            "spc": _spc_check(df.iloc[i], limits),
        })
    return {"model_version": ver, "n": len(out), "predictions": out}


def _cli():
    """`pdm-serve` 진입점 — uvicorn으로 API 기동(기본 0.0.0.0:8000)."""
    import os

    import uvicorn
    uvicorn.run("pdm.serving.api:app",
                host=os.environ.get("PDM_API_HOST", "0.0.0.0"),
                port=int(os.environ.get("PDM_API_PORT", "8000")),
                reload=bool(os.environ.get("PDM_API_RELOAD")))
