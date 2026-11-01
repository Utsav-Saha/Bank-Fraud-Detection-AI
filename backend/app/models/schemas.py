"""Pydantic schemas used for API validation and consistent responses."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class APIError(BaseModel):
    error: str
    message: str
    details: dict[str, Any] | None = None


class ModelAvailability(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    name: str
    filename: str
    available: bool
    loaded: bool = False
    artifact_type: str
    notes: str | None = None


class HealthResponse(BaseModel):
    status: str
    service: str
    environment: str
    database: str
    models_dir: str
    models: list[ModelAvailability]
    missing_required: list[str] = Field(default_factory=list)
