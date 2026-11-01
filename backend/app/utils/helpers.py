"""Small shared helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import jsonify


def json_error(message: str, status: int, error: str = "request_error", details: dict[str, Any] | None = None):
    payload = {"error": error, "message": message}
    if details:
        payload["details"] = details
    return jsonify(payload), status


def path_exists(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False
