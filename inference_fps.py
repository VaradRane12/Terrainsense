"""
inference_fps.py
────────────────
Lightweight rolling FPS tracker used by inference.py's OSD renderer.
Kept in its own file to avoid a circular import between inference.py and camera.py.
"""

import time
from collections import deque

_times: deque = deque(maxlen=10)


def record_frame() -> float:
    """Call once per processed frame. Returns the current rolling FPS."""
    now = time.time()
    _times.append(now)
    return get_avg_fps()


def get_avg_fps() -> float:
    if len(_times) < 2:
        return 0.0
    span = _times[-1] - _times[0]
    return (len(_times) - 1) / span if span > 0 else 0.0
