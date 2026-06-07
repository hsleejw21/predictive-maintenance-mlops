"""Layer 2 — Random Forest 다중분류기(설계문서 5장).

수백 그루의 결정나무 다수결로 6개 클래스(정상+5고장)를 분류한다. 고장이 3.4%뿐인
클래스 불균형은 class_weight="balanced"로 희소 고장에 가중치를 줘 보정한다.
"""
from sklearn.ensemble import RandomForestClassifier

# 하이퍼파라미터를 상수로 노출 → 레지스트리 메타데이터에 그대로 기록(재현성)
RF_PARAMS = dict(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    n_jobs=-1,
)


def train_rf(X, y, seed=42):
    clf = RandomForestClassifier(random_state=seed, **RF_PARAMS)
    clf.fit(X, y)
    return clf
