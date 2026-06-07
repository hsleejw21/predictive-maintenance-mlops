"""Model 레지스트리 — 저장/로드/조회 라운드트립(임시 디렉터리, 실제 models/ 미오염)."""
from sklearn.ensemble import RandomForestClassifier

from pdm.models import registry


def _tiny_model():
    clf = RandomForestClassifier(n_estimators=3, random_state=0)
    clf.fit([[0], [1], [2], [3]], ["Normal", "HDF", "Normal", "HDF"])
    return clf


def test_save_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path)
    model = _tiny_model()
    meta = registry.save_model(
        model, "v1",
        metrics={"drift_region": {"fail_recall": 0.9}},
        train_range=(0, 3000), feature_names=["f"], scenario={"seed": 42},
    )
    assert meta["version"] == "v1"
    assert meta["n_train"] == 3000
    assert meta["train_range"] == [0, 3000]
    assert meta["feature_importances"] is not None

    loaded = registry.load_model("v1")
    assert list(loaded.predict([[0], [1]])) == list(model.predict([[0], [1]]))
    assert registry.load_meta("v1")["metrics"]["drift_region"]["fail_recall"] == 0.9


def test_list_and_latest_ordered_by_trained_at(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path)
    registry.save_model(_tiny_model(), "v1", train_range=(0, 100))
    registry.save_model(_tiny_model(), "v2", train_range=(0, 200),
                        extra={"trained_at": "2999-01-01T00:00:00+00:00"})
    versions = [m["version"] for m in registry.list_models()]
    assert set(versions) == {"v1", "v2"}
    assert registry.latest()["version"] == "v2"   # 가장 늦은 trained_at


def test_save_json_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path)
    path = registry.save_json_artifact("spc_limits", {"a": 1})
    assert path.exists()
    assert path.name == "spc_limits.json"
