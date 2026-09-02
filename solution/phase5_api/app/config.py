"""Runtime configuration for the Phase 5 API.

Everything is read from the environment (12-factor); Postgres detail is reused
from ``capstone.db.SETTINGS`` so connection config lives in exactly one place
(``solution/.env``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from capstone import serving as S

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class ApiConfig:
    models_dir: Path
    serving_config_path: Path
    log_predictions: bool
    log_request_payload: bool
    title: str
    version: str

    @classmethod
    def from_env(cls) -> "ApiConfig":
        models_dir = Path(os.environ.get("API_MODELS_DIR", str(S.PHASE3_MODELS_DIR)))
        if not models_dir.is_absolute():
            models_dir = (_REPO_ROOT / models_dir).resolve()
        cfg_path = Path(os.environ.get("API_SERVING_CONFIG", str(S.PHASE5_DIR / "serving_config.json")))
        if not cfg_path.is_absolute():
            cfg_path = (_REPO_ROOT / cfg_path).resolve()
        return cls(
            models_dir=models_dir,
            serving_config_path=cfg_path,
            log_predictions=_env_bool("API_LOG_PREDICTIONS", True),
            log_request_payload=_env_bool("API_LOG_REQUEST_PAYLOAD", True),
            title=os.environ.get("API_TITLE", "Hospital Operations & Revenue Risk Intelligence API"),
            version=S.SERVING_VERSION,
        )


CONFIG = ApiConfig.from_env()
