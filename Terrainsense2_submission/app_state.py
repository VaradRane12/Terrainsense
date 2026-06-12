"""
app_state.py
────────────
Central app state management to avoid circular imports.
Shared between app.py and camera.py.
"""

import threading

_state_lock = threading.Lock()
_app_state = {
    "running": True,
}


def get_running() -> bool:
    """Check if app inference is currently running."""
    with _state_lock:
        return _app_state["running"]


def set_running(running: bool) -> None:
    """Set app inference state."""
    with _state_lock:
        _app_state["running"] = bool(running)
