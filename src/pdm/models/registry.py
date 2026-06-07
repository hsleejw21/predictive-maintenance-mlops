"""모델 레지스트리 — 학습한 모델을 버전별로 디스크에 저장/조회(설계문서 7장 Model).

문서의 "v1, v2… 버전을 붙여 보관. 나중에 그때 그 모델을 그대로 재현"을 실제 구현한다.
각 버전마다:
  models/rf_<version>.joblib   — 직렬화된 RandomForest
  models/rf_<version>.json     — 메타데이터(학습시점·학습범위·지표·하이퍼파라미터·특징·시나리오)
또한 SPC 한계와 PSI 기준 분포도 JSON으로 저장해 모니터링 기준까지 재현 가능하게 한다.
"""
import json
from datetime import datetime, timezone

import joblib

from pdm.config import MODELS_DIR


def _ensure_dir():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def model_path(version):
    return MODELS_DIR / f"rf_{version}.joblib"


def meta_path(version):
    return MODELS_DIR / f"rf_{version}.json"


def save_model(model, version, *, metrics=None, train_range=None,
               feature_names=None, scenario=None, extra=None):
    """모델 + 메타데이터를 레지스트리에 저장하고 메타데이터 dict를 반환."""
    _ensure_dir()
    joblib.dump(model, model_path(version))
    meta = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_type": type(model).__name__,
        "params": model.get_params(),
        "classes": list(getattr(model, "classes_", [])),
        "n_train": (train_range[1] - train_range[0]) if train_range else None,
        "train_range": list(train_range) if train_range else None,
        "feature_names": list(feature_names) if feature_names else None,
        "feature_importances": (
            dict(zip(feature_names, [float(v) for v in model.feature_importances_]))
            if feature_names is not None and hasattr(model, "feature_importances_")
            else None
        ),
        "metrics": metrics or {},
        "scenario": scenario or {},
    }
    if extra:
        meta.update(extra)
    meta_path(version).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def load_model(version):
    """버전 이름으로 모델 로드."""
    return joblib.load(model_path(version))


def load_meta(version):
    return json.loads(meta_path(version).read_text(encoding="utf-8"))


def list_models():
    """저장된 모든 버전의 메타데이터를 학습시점 순으로 반환."""
    _ensure_dir()
    metas = [json.loads(p.read_text(encoding="utf-8"))
             for p in sorted(MODELS_DIR.glob("rf_*.json"))]
    return sorted(metas, key=lambda m: m.get("trained_at", ""))


def latest():
    """가장 최근에 학습된 모델의 메타데이터(없으면 None)."""
    metas = list_models()
    return metas[-1] if metas else None


def save_json_artifact(name, obj):
    """SPC 한계·PSI 기준 등 부가 산출물을 models/<name>.json 으로 저장."""
    _ensure_dir()
    path = MODELS_DIR / f"{name}.json"
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
