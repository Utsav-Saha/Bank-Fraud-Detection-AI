"""Environment-based Flask configuration."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

load_dotenv(BACKEND_DIR / ".env")
load_dotenv(PROJECT_ROOT / ".env")


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


class Config:
    """Base configuration shared by all environments."""

    SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-change-me")
    DEBUG = False
    TESTING = False

    PROJECT_ROOT = PROJECT_ROOT
    BACKEND_DIR = BACKEND_DIR
    MODELS_DIR = Path(os.getenv("MODELS_DIR", str(PROJECT_ROOT / "models")))
    if not MODELS_DIR.is_absolute():
        MODELS_DIR = PROJECT_ROOT / MODELS_DIR

    METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
    SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "sample" / "sample_transactions.csv"

    _db_url = os.getenv("DATABASE_URL", "")
    if not _db_url:
        sqlite_path = PROJECT_ROOT / "data" / "fraud_detection.db"
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        _db_url = "sqlite:///" + sqlite_path.as_posix()
    SQLALCHEMY_DATABASE_URI = _db_url
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {"pool_pre_ping": True}

    CORS_ORIGINS = [
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if origin.strip()
    ]

    MAX_CONTENT_LENGTH = _int_env("MAX_CONTENT_LENGTH", 10 * 1024 * 1024)
    FRAUD_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.5"))

    JSON_SORT_KEYS = False


class DevelopmentConfig(Config):
    DEBUG = _bool_env("FLASK_DEBUG", True)


class TestingConfig(Config):
    TESTING = True
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    SQLALCHEMY_ENGINE_OPTIONS = {}
    WTF_CSRF_ENABLED = False


class ProductionConfig(Config):
    DEBUG = False
    TESTING = False

    def __init__(self) -> None:
        if self.SECRET_KEY in {"dev-only-change-me", "change-this-to-a-long-random-value"}:
            raise RuntimeError("SECRET_KEY must be set to a strong value in production.")


CONFIG_MAP = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
}


def get_config(name: str | None = None) -> type[Config]:
    env_name = (name or os.getenv("FLASK_ENV") or "development").strip().lower()
    return CONFIG_MAP.get(env_name, DevelopmentConfig)
