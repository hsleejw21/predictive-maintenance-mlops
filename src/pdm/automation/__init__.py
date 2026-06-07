"""Automation 레이어 — 드리프트 감지 트리거와 자동 재학습/검증 오케스트레이션."""
from pdm.automation.retrain import detect_trigger, retrain, validate_and_promote

__all__ = ["detect_trigger", "retrain", "validate_and_promote"]
