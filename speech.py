"""
speech.py
─────────
Manages a background thread that feeds text to espeak/espeak-ng.
All other modules call queue_speech(text) — fire-and-forget, never blocks.
"""

import queue
import subprocess
import threading
import time

from config import ENABLE_VOICE, VOICE_INTERVAL_SEC, VOICE_RATE

# ── Shared state ────────────────────────────────────────────────────────────
_speech_queue: queue.Queue = queue.Queue(maxsize=6)
_speech_state: dict = {
    "enabled":               ENABLE_VOICE,
    "last_text":             "",
    "last_time":             0.0,
    "warned_missing_engine": False,
}

# Injected by recording.py at startup to log audio events — avoids circular import
_audio_event_callback = None


# ── Public API ──────────────────────────────────────────────────────────────
def queue_speech(text: str | None) -> None:
    """Enqueue a speech utterance. Returns immediately; never blocks."""
    if not text or not _speech_state["enabled"]:
        return

    now = time.time()
    if now - _speech_state["last_time"] < VOICE_INTERVAL_SEC:
        return
    if text == _speech_state["last_text"] and now - _speech_state["last_time"] < VOICE_INTERVAL_SEC * 2.0:
        return

    _speech_state["last_text"] = text
    _speech_state["last_time"] = now

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
    return {"ok": True, "voice_enabled": _speech_state["enabled"], "interval_sec": VOICE_INTERVAL_SEC}


def register_audio_event_callback(fn) -> None:
    """Called once by recording.py to hook into speech events for narration."""
    global _audio_event_callback
    _audio_event_callback = fn


def start_speech_thread() -> threading.Thread:
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
