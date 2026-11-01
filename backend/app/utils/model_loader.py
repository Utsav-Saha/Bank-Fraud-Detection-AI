"""Inspect trusted local model artifacts without loading them into memory.

Full joblib/TensorFlow loading is implemented in Phase 3. This module only
reports filesystem presence so the health endpoint can start before inference
is wired.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

KNOWN_MODELS: tuple[dict[str, str], ...] = (
    {
        "name": "random_forest",
        "filename": "random_forest.pkl",
        "artifact_type": "sklearn_joblib",
        "required": "true",
    },
    {
        "name": "logistic_regression",
        "filename": "logistic_regression.pkl",
        "artifact_type": "sklearn_joblib",
        "required": "true",
    },
    {
        "name": "scaler",
        "filename": "scaler.pkl",
        "artifact_type": "sklearn_joblib",
        "required": "true",
    },
    {
        "name": "isolation_forest",
        "filename": "isolation_forest.pkl",
        "artifact_type": "sklearn_joblib",
        "required": "false",
    },
    {
        "name": "ann",
        "filename": "fraud_ann_model.keras",
        "artifact_type": "keras",
        "required": "false",
    },
    {
        "name": "feature_columns",
        "filename": "feature_columns.json",
        "artifact_type": "json",
        "required": "true",
    },
)


@dataclass(frozen=True)
class ArtifactStatus:
    name: str
    filename: str
    available: bool
    loaded: bool
    artifact_type: str
    required: bool
    notes: str | None
    path: Path

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "filename": self.filename,
            "available": self.available,
            "loaded": self.loaded,
            "artifact_type": self.artifact_type,
            "notes": self.notes,
        }


def inspect_artifacts(models_dir: Path) -> list[ArtifactStatus]:
    results: list[ArtifactStatus] = []
    for spec in KNOWN_MODELS:
        path = models_dir / spec["filename"]
        available = path.is_file()
        notes = None
        if not available and spec["name"] == "ann":
            notes = "TensorFlow ANN artifact is not present. The API will not substitute another model."
        elif not available and spec["required"] == "true":
            notes = "Required artifact is missing from the trusted models directory."
        results.append(
            ArtifactStatus(
                name=spec["name"],
                filename=spec["filename"],
                available=available,
                loaded=False,
                artifact_type=spec["artifact_type"],
                required=spec["required"] == "true",
                notes=notes,
                path=path,
            )
        )
    return results
