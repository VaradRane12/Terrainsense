"""
sensor.py
─────────
Runs the sensor subprocess (ToF 8×8 + IMU) in its own daemon thread.
All other modules call get_sensor_snapshot() to read the latest values
without ever blocking on I/O.
"""

import json
import math
import subprocess
import threading
import time

import numpy as np

from config import ENABLE_SENSOR_BRIDGE, SENSOR_STREAM_CMD

# ── Shared state ────────────────────────────────────────────────────────────
_sensor_lock = threading.Lock()
_sensor_state: dict = {
    "enabled":     ENABLE_SENSOR_BRIDGE,
    "front_mm":    None,
    "front_valid_cells": 0,
    "pitch_deg":   0.0,
    "yaw_deg":     0.0,
    "quat":        [1.0, 0.0, 0.0, 0.0],
    "distances":   [0] * 64,
    "status":      [0] * 64,
    "last_update": 0.0,
    "error":       None,
}


# ── Public API ──────────────────────────────────────────────────────────────
def get_sensor_snapshot() -> dict:
    """Return a shallow copy of the latest sensor state (non-blocking)."""
    with _sensor_lock:
        return dict(_sensor_state)


def start_sensor_thread() -> threading.Thread | None:
    """Start the sensor bridge daemon thread. Returns the thread (or None if disabled)."""
    if not ENABLE_SENSOR_BRIDGE:
        with _sensor_lock:
            _sensor_state["error"] = "Sensor bridge disabled"
        print("[INFO] Sensor bridge: disabled")
        return None

    t = threading.Thread(target=_sensor_stream_worker, daemon=True, name="sensor-bridge")
    t.start()
    print(f"[INFO] Sensor bridge started: {SENSOR_STREAM_CMD}")
    return t


# ── Internal helpers ────────────────────────────────────────────────────────
def _extract_front_stats(distances: list, status: list) -> tuple[int | None, int]:
    """Return (median_mm, valid_cells) for the centre 4×4 cells in the 8×8 ToF grid."""
    centre_idxs = [18, 19, 20, 21, 26, 27, 28, 29, 34, 35, 36, 37, 42, 43, 44, 45]
    vals = [
        int(distances[i])
        for i in centre_idxs
        if i < len(distances) and i < len(status)
        and status[i] == 5 and distances[i] > 0
    ]
    return (int(np.median(vals)), len(vals)) if vals else (None, 0)


def _extract_front_mm(distances: list, status: list) -> int | None:
    mm, _ = _extract_front_stats(distances, status)
    return mm


def _quat_to_pitch_deg(quat: list) -> float:
    if not quat or len(quat) != 4:
        return 0.0
    w, x, y, z = (float(v) for v in quat)
    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    return float(math.degrees(math.asin(sinp)))


def _quat_to_yaw_deg(quat: list) -> float:
    if not quat or len(quat) != 4:
        return 0.0
    w, x, y, z = (float(v) for v in quat)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(math.degrees(math.atan2(siny_cosp, cosy_cosp)))


def _sensor_stream_worker() -> None:
    """Daemon thread: reads JSON lines from the sensor subprocess."""
    try:
        proc = subprocess.Popen(
            SENSOR_STREAM_CMD,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
    except Exception as exc:
        with _sensor_lock:
            _sensor_state["error"] = f"Failed to start sensor stream: {exc}"
        return

    if proc.stdout is None:
        with _sensor_lock:
            _sensor_state["error"] = "Sensor stream stdout unavailable"
        return

    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            pkt = json.loads(line)

            distances = list(pkt.get("distances", []))[:64]
            status    = list(pkt.get("status",    []))[:64]
            quat      = list(pkt.get("quat", [1.0, 0.0, 0.0, 0.0]))[:4]

            # Pad to full length
            distances += [0] * (64 - len(distances))
            status    += [0] * (64 - len(status))
            if len(quat) < 4:
                quat = [1.0, 0.0, 0.0, 0.0]

            front_mm, front_valid_cells = _extract_front_stats(distances, status)
            pitch_deg = _quat_to_pitch_deg(quat)
            yaw_deg   = _quat_to_yaw_deg(quat)

            with _sensor_lock:
                _sensor_state["front_mm"]    = front_mm
                _sensor_state["front_valid_cells"] = front_valid_cells
                _sensor_state["pitch_deg"]   = pitch_deg
                _sensor_state["yaw_deg"]     = yaw_deg
                _sensor_state["quat"]        = quat
                _sensor_state["distances"]   = distances
                _sensor_state["status"]      = status
                _sensor_state["last_update"] = time.time()
                _sensor_state["error"]       = None

        except Exception as exc:
            with _sensor_lock:
                _sensor_state["error"] = f"Sensor parse error: {exc}"

    with _sensor_lock:
        _sensor_state["error"] = "Sensor stream stopped"
