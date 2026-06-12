"""
speech.py
─────────
Manages a background thread that feeds text to espeak/espeak-ng.
All other modules call queue_speech(text) — fire-and-forget, never blocks.
"""

import queue
import re
import subprocess
import threading
import time

from config import (
    AUDIO_FORCE_AUX_ONLY,
    AUDIO_QUERY_CMD,
    AUDIO_SWITCH_AUX_CMD,
    AUDIO_SWITCH_BT_CMD,
    ENABLE_VOICE,
    VOICE_INTERVAL_SEC,
    VOICE_RATE,
)

# ── Shared state ────────────────────────────────────────────────────────────
_speech_queue: queue.Queue = queue.Queue(maxsize=6)
_speech_state: dict = {
    "enabled":               ENABLE_VOICE,
    "last_text":             "",
    "last_intent":           "",
    "last_time":             0.0,
    "last_intent_time":      0.0,
    "warned_missing_engine": False,
}

# Injected by recording.py at startup to log audio events — avoids circular import
_audio_event_callback = None


def _detect_audio_mode() -> str:
    try:
        result = subprocess.run(
            AUDIO_QUERY_CMD,
            shell=True,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        sink = (result.stdout or "").strip().lower()
        if "bluez" in sink or "bluetooth" in sink:
            return "bluetooth"
        if "analog" in sink or "headphone" in sink or "alsa" in sink:
            return "aux"
    except Exception:
        pass
    return "unknown"


# ── Public API ──────────────────────────────────────────────────────────────
def queue_speech(text: str | None) -> None:
    """Enqueue a speech utterance. Returns immediately; never blocks."""
    if not text or not _speech_state["enabled"]:
        return

    now = time.time()
    intent = _normalize_intent(text)
    urgent = _is_urgent_intent(intent)

    min_gap = max(0.8, VOICE_INTERVAL_SEC * (0.5 if urgent else 1.0))
    intent_gap = VOICE_INTERVAL_SEC * (1.2 if urgent else 2.0)

    if now - _speech_state["last_time"] < min_gap:
        return
    if intent == _speech_state["last_intent"] and now - _speech_state["last_intent_time"] < intent_gap:
        return
    if text == _speech_state["last_text"] and now - _speech_state["last_time"] < VOICE_INTERVAL_SEC * 2.0:
        return

    _speech_state["last_text"] = text
    _speech_state["last_intent"] = intent
    _speech_state["last_time"] = now
    _speech_state["last_intent_time"] = now

    if _audio_event_callback is not None:
        _audio_event_callback(text, now)

    try:
        _speech_queue.put_nowait(text)
    except queue.Full:
        pass


def set_voice_enabled(enabled: bool) -> dict:
    _speech_state["enabled"] = bool(enabled)
    return {"ok": True, "voice_enabled": _speech_state["enabled"], "interval_sec": VOICE_INTERVAL_SEC}


def get_voice_status() -> dict:
    return {
        "ok": True,
        "voice_enabled": _speech_state["enabled"],
        "interval_sec": VOICE_INTERVAL_SEC,
        "audio_output": _detect_audio_mode(),
    }


def set_audio_output(mode: str) -> dict:
    mode = str(mode).strip().lower()
    if mode not in ("bluetooth", "aux"):
        return {"ok": False, "error": "mode must be 'bluetooth' or 'aux'"}

    if AUDIO_FORCE_AUX_ONLY and mode != "aux":
        return {
            "ok": False,
            "mode": mode,
            "error": "Bluetooth output is disabled. AUX-only mode is enabled.",
        }

    cmd = AUDIO_SWITCH_BT_CMD if mode == "bluetooth" else AUDIO_SWITCH_AUX_CMD
    if not cmd.strip():
        return {"ok": False, "error": f"No command configured for mode: {mode}"}

    res = subprocess.run(cmd, shell=True, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        return {
            "ok": False,
            "mode": mode,
            "error": (res.stderr or res.stdout or "audio switch command failed").strip(),
        }

    return {
        "ok": True,
        "mode": mode,
        "audio_output": _detect_audio_mode(),
    }


def register_audio_event_callback(fn) -> None:
    """Called once by recording.py to hook into speech events for narration."""
    global _audio_event_callback
    _audio_event_callback = fn


def start_speech_thread() -> threading.Thread:
    # Best-effort: force AUX sink at startup so disconnected BT devices do not mute output.
    if AUDIO_FORCE_AUX_ONLY:
        set_audio_output("aux")
    t = threading.Thread(target=_speech_worker, daemon=True, name="speech")
    t.start()
    return t


# ── Worker ──────────────────────────────────────────────────────────────────
def _speech_worker() -> None:
    while True:
        text = _speech_queue.get()
        if text is None:
            _speech_queue.task_done()
            break

        if _speech_state["enabled"]:
            ok = False
            for engine in ("espeak", "espeak-ng"):
                try:
                    subprocess.run(
                        [engine, "-s", VOICE_RATE, text],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    ok = True
                    break
                except FileNotFoundError:
                    continue

            if not ok:
                if not _speech_state["warned_missing_engine"]:
                    print("[WARN] No speech engine found. Install espeak or espeak-ng.")
                    _speech_state["warned_missing_engine"] = True
                _speech_state["enabled"] = False

        _speech_queue.task_done()


def _normalize_intent(text: str) -> str:
    """Normalize dynamic content so distance jitter does not trigger repeated speech."""
    t = text.lower().strip()
    t = re.sub(r"\d+", "<n>", t)
    t = re.sub(r"\s+", " ", t)
    return t


def _is_urgent_intent(intent: str) -> bool:
    return any(k in intent for k in ("stop", "alert", "fall", "obstacle"))
