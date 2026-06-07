"""추론 API 계약 — /health, /models, /predict(단건·배치) + SPC 플래그.

레지스트리에 저장된 모델(models/)이 있어야 /predict가 동작한다. 없으면 503을 기대.
"""
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from pdm.models import registry  # noqa: E402
from pdm.serving.api import app  # noqa: E402

client = TestClient(app)

SAMPLE = {
    "Type": "L",
    "Air temperature [K]": 300.0,
    "Process temperature [K]": 310.0,
    "Rotational speed [rpm]": 1400.0,
    "Torque [Nm]": 45.0,
    "Tool wear [min]": 120.0,
}

_has_model = registry.latest() is not None
needs_model = pytest.mark.skipif(not _has_model, reason="models/ 에 저장된 모델 필요")


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@needs_model
def test_predict_single():
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 1
    pred = body["predictions"][0]
    assert pred["predicted_mode"] in ["Normal", "TWF", "HDF", "PWF", "OSF", "RNF"]
    assert 0.0 <= pred["confidence"] <= 1.0
    # 확률 합 ≈ 1
    assert abs(sum(pred["probabilities"].values()) - 1.0) < 1e-6


@needs_model
def test_predict_batch():
    r = client.post("/predict", json=[SAMPLE, {**SAMPLE, "Torque [Nm]": 70.0}])
    assert r.status_code == 200
    assert r.json()["n"] == 2


@needs_model
def test_predict_includes_spc_flags():
    pred = client.post("/predict", json=SAMPLE).json()["predictions"][0]
    assert "spc" in pred
    # spc_limits.json 이 있으면 by_sensor 플래그가 포함된다
    if pred["spc"].get("available"):
        assert "any_ooc" in pred["spc"]


@needs_model
def test_models_listing():
    r = client.get("/models")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_predict_unknown_version_404():
    r = client.post("/predict", json=SAMPLE, params={"version": "does-not-exist"})
    assert r.status_code == 404
