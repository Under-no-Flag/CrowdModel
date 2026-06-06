"""Shared config: env vars > config.json > hardcoded defaults."""

import json
import os
from pathlib import Path

CONFIG_FILE = Path(__file__).resolve().parent.parent / "config.json"
DEFAULTS = {
    "host": "127.0.0.1",
    "port": 8073,
    "default_view_mode": "normal",
    "auto_refresh": "eink-off",
    "refresh_interval": 3,
}


def _load_file_config():
    try:
        return json.loads(CONFIG_FILE.read_text())
    except Exception:
        return {}


def get_all_config():
    """Return merged config: file values over defaults."""
    cfg = dict(DEFAULTS)
    cfg.update(_load_file_config())
    return cfg


def save_config(data):
    """Write config to file. Only saves known keys."""
    current = _load_file_config()
    for key in DEFAULTS:
        if key in data:
            current[key] = data[key]
    CONFIG_FILE.write_text(json.dumps(current, indent=2, ensure_ascii=False) + "\n")


_file_cfg = _load_file_config()


def get_host():
    return os.environ.get("TIKZ_SERVER_HOST") or _file_cfg.get("host") or DEFAULTS["host"]


def get_port():
    val = os.environ.get("TIKZ_SERVER_PORT")
    if val:
        return int(val)
    return _file_cfg.get("port") or DEFAULTS["port"]


def get_default_view_mode():
    val = os.environ.get("TIKZ_DEFAULT_VIEW_MODE")
    if val and val in ("eink", "normal", "eyecare"):
        return val
    return _file_cfg.get("default_view_mode") or DEFAULTS["default_view_mode"]
