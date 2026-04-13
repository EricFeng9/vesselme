from __future__ import annotations

import json
from pathlib import Path


DEFAULT_SETTINGS = {
    "right_zoom_sensitivity": 300.0,
}


def _settings_path() -> Path:
    return Path.home() / ".vesselme_config.json"


def load_settings() -> dict:
    path = _settings_path()
    if not path.exists():
        return dict(DEFAULT_SETTINGS)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return dict(DEFAULT_SETTINGS)

    out = dict(DEFAULT_SETTINGS)
    if isinstance(payload, dict):
        out.update(payload)
    return out


def save_settings(settings: dict) -> None:
    path = _settings_path()
    merged = dict(DEFAULT_SETTINGS)
    merged.update(settings)
    path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

