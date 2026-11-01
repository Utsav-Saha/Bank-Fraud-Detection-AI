"""Trusted in-memory model cache. Only project artifacts are loaded."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)

SUPPORTED_SUPERVISED = ("random_forest", "logistic_regression", "tensorflow_ann")


class ModelNotAvailableError(RuntimeError):
    """Raised when a requested model artifact is missing or failed to load."""


@dataclass
class ModelRegistry:
    models_dir: Path
    feature_columns: list[str] = field(default_factory=list)
    scaler: Any = None
    supervised: dict[str, Any] = field(default_factory=dict)
    isolation_forest: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    load_errors: dict[str, str] = field(default_factory=dict)
    sklearn_version_warnings: list[str] = field(default_factory=list)

    def available_supervised(self) -> list[str]:
        return sorted(self.supervised.keys())

    def require(self, model_name: str) -> Any:
        if model_name == "tensorflow_ann" and model_name not in self.supervised:
            raise ModelNotAvailableError(
                "tensorflow_ann is not available. models/fraud_ann_model.keras was not found. "
                "No substitute model is used. Train and export an ANN with scripts/export_models.py "
                "if you need this endpoint."
            )
        if model_name not in self.supervised:
            raise ModelNotAvailableError(
                f"Model '{model_name}' is not loaded. Available: {self.available_supervised() or 'none'}."
            )
        return self.supervised[model_name]


_REGISTRY: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    if _REGISTRY is None:
        raise RuntimeError("Model registry has not been initialized.")
    return _REGISTRY


def init_registry(models_dir: Path) -> ModelRegistry:
    global _REGISTRY
    _REGISTRY = load_registry(models_dir)
    return _REGISTRY


def load_registry(models_dir: Path) -> ModelRegistry:
    registry = ModelRegistry(models_dir=models_dir)
    feature_path = models_dir / "feature_columns.json"
    metadata_path = models_dir / "model_metadata.json"

    if not feature_path.exists():
        registry.load_errors["feature_columns"] = f"Missing {feature_path}"
        logger.error(registry.load_errors["feature_columns"])
        return registry

    registry.feature_columns = json.loads(feature_path.read_text(encoding="utf-8"))
    if metadata_path.exists():
        registry.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    registry.scaler = _safe_joblib(models_dir / "scaler.pkl", "scaler", registry)
    _load_supervised(models_dir / "random_forest.pkl", "random_forest", registry)
    _load_supervised(models_dir / "logistic_regression.pkl", "logistic_regression", registry)
    _load_keras(models_dir / "fraud_ann_model.keras", registry)
    registry.isolation_forest = _safe_joblib(
        models_dir / "isolation_forest.pkl", "isolation_forest", registry
    )

    _verify_feature_contract(registry)
    return registry


def _safe_joblib(path: Path, key: str, registry: ModelRegistry):
    if not path.exists():
        registry.load_errors[key] = f"Missing artifact: {path.name}"
        logger.warning(registry.load_errors[key])
        return None
    try:
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            obj = joblib.load(path)
            for item in caught:
                message = str(item.message)
                if "InconsistentVersionWarning" in item.category.__name__ or "unpickle estimator" in message:
                    registry.sklearn_version_warnings.append(f"{path.name}: {message}")
        return obj
    except Exception as exc:
        registry.load_errors[key] = f"Failed to load {path.name}: {exc}"
        logger.exception(registry.load_errors[key])
        return None


def _load_supervised(path: Path, name: str, registry: ModelRegistry) -> None:
    model = _safe_joblib(path, name, registry)
    if model is not None:
        registry.supervised[name] = model


def _load_keras(path: Path, registry: ModelRegistry) -> None:
    if not path.exists():
        registry.load_errors["tensorflow_ann"] = (
            "Missing models/fraud_ann_model.keras. ANN serving is disabled until the artifact exists."
        )
        logger.warning(registry.load_errors["tensorflow_ann"])
        return
    try:
        from tensorflow import keras  # type: ignore

        registry.supervised["tensorflow_ann"] = keras.models.load_model(path)
    except Exception as exc:
        registry.load_errors["tensorflow_ann"] = f"Failed to load Keras ANN: {exc}"
        logger.exception(registry.load_errors["tensorflow_ann"])


def _verify_feature_contract(registry: ModelRegistry) -> None:
    expected = registry.feature_columns
    if not expected:
        return
    objects = [("scaler", registry.scaler)]
    objects.extend(registry.supervised.items())
    if registry.isolation_forest is not None:
        objects.append(("isolation_forest", registry.isolation_forest))

    for name, obj in objects:
        if obj is None:
            continue
        n_features = getattr(obj, "n_features_in_", None)
        names = getattr(obj, "feature_names_in_", None)
        if n_features is not None and int(n_features) != len(expected):
            registry.load_errors[f"{name}_features"] = (
                f"{name} expects {n_features} features, schema has {len(expected)}."
            )
        if names is not None and list(names) != expected:
            registry.load_errors[f"{name}_order"] = (
                f"{name} feature order does not match feature_columns.json."
            )
