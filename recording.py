"""
recording.py
────────────
Manages video recording to disk with optional TTS narration baked in.
Writing is offloaded to a dedicated thread so it never stalls the frame pipeline.

Thread safety
─────────────
  _record_lock  guards _record_state
  _write_queue  is a Queue[np.ndarray | None] consumed by _record_writer_worker
"""

import os
import queue
import shutil
import subprocess
import threading
import time
from datetime import datetime

import cv2

import speech as speech_mod
from config import (
    DEFAULT_RECORD_FPS,
    DEFAULT_RECORD_MINUTES,
    ENABLE_RECORD_AUDIO,
    RECORD_AUDIO_RATE,
    RECORDINGS_DIR,
    VOICE_RATE,
)

# ── Shared state ─────────────────────────────────────────────────────────────
_record_lock = threading.Lock()
_record_state: dict = {
    "active":         False,
    "path":           None,
    "fps":            DEFAULT_RECORD_FPS,
    "started_at":     0.0,
    "stop_at":        0.0,
    "audio_enabled":  ENABLE_RECORD_AUDIO,
    "audio_path":     None,
    "audio_events":   [],   # list of {"t": float, "text": str}
    "audio_error":    None,
}

# Dedicated write queue — frames are put here; worker writes to disk
_write_queue: queue.Queue = queue.Queue(maxsize=60)


# ── Public API ───────────────────────────────────────────────────────────────
def start_recording(minutes: float = DEFAULT_RECORD_MINUTES,
                    fps: float = DEFAULT_RECORD_FPS) -> dict:
    if minutes <= 0:
        raise ValueError("minutes must be > 0")
    if fps <= 0:
        raise ValueError("fps must be > 0")

    with _record_lock:
        if _record_state["active"]:
            return {"ok": False, "error": "Recording already active", "saved_to": _record_state["path"]}

        now = time.time()
        _record_state["active"]       = True
        _record_state["path"]         = _create_recording_path()
        _record_state["fps"]          = float(fps)
        _record_state["started_at"]   = now
        _record_state["stop_at"]      = now + float(minutes) * 60.0
        _record_state["audio_path"]   = None
        _record_state["audio_events"] = []
        _record_state["audio_error"]  = None

        return {
            "ok":            True,
            "recording":     True,
            "minutes":       float(minutes),
            "fps":           float(fps),
            "saved_to":      _record_state["path"],
            "audio_enabled": _record_state["audio_enabled"],
        }


def stop_recording() -> dict:
    with _record_lock:
        return _stop_locked()


def get_record_status() -> dict:
    with _record_lock:
        left = max(0.0, _record_state["stop_at"] - time.time()) if _record_state["active"] else 0.0
        return {
            "ok":            True,
            "recording":     _record_state["active"],
            "saved_to":      _record_state["path"],
            "seconds_left":  round(left, 1),
            "fps":           _record_state["fps"],
            "audio_enabled": _record_state["audio_enabled"],
            "audio_error":   _record_state["audio_error"],
        }


def maybe_enqueue_frame(frame) -> None:
    """Call from the inference thread. Non-blocking — drops frame if queue is full."""
    with _record_lock:
        if not _record_state["active"]:
            return
        if time.time() >= _record_state["stop_at"]:
            _stop_locked()
            return

    try:
            # Include capture timestamp so writer can preserve real-time pacing.
            _write_queue.put_nowait((time.time(), frame.copy()))
    except queue.Full:
        pass  # drop frame rather than stall inference


def register_with_speech() -> None:
    """Hook speech.py so that voice events get timestamped into audio_events."""
    speech_mod.register_audio_event_callback(_on_speech_event)


def start_recording_thread() -> threading.Thread:
    t = threading.Thread(target=_record_writer_worker, daemon=True, name="recording-writer")
    t.start()
    return t


# ── Internal ─────────────────────────────────────────────────────────────────
def _on_speech_event(text: str, now: float) -> None:
    with _record_lock:
        if _record_state["active"]:
            t_rel = max(0.0, now - _record_state["started_at"])
            _record_state["audio_events"].append({"t": t_rel, "text": text})


def _create_recording_path() -> str:
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(RECORDINGS_DIR, f"terrain_{stamp}.mp4")


def _stop_locked() -> dict:
    """Must be called with _record_lock held."""
    # Signal writer thread to flush
    try:
        _write_queue.put_nowait(None)
    except queue.Full:
        pass

    was_active   = _record_state["active"]
    saved_path   = _record_state["path"]
    started_at   = _record_state["started_at"]
    duration_sec = max(0.0, time.time() - started_at) if started_at else 0.0

    final_path = saved_path
    had_audio  = False
    if saved_path:
        audio_path, built = _render_tts_audio_locked(saved_path, duration_sec)
        if built:
            final_path, had_audio = _mux_audio_video_locked(saved_path, audio_path=audio_path)
        else:
            final_path, had_audio = _mux_audio_video_locked(saved_path)

    _record_state["active"]       = False
    _record_state["path"]         = None
    _record_state["started_at"]   = 0.0
    _record_state["stop_at"]      = 0.0
    _record_state["audio_path"]   = None
    _record_state["audio_events"] = []

    return {
        "ok":            True,
        "was_active":    was_active,
        "saved_to":      final_path,
        "audio_recorded": had_audio,
        "audio_error":   _record_state["audio_error"],
        "duration_sec":  round(duration_sec, 2),
    }


def _record_writer_worker() -> None:
    """
    Dedicated thread: owns the cv2.VideoWriter so disk I/O never touches
    the camera/inference thread.
    """
    writer = None
    path = None
    fps = DEFAULT_RECORD_FPS
    frame_wh = None
    video_start_ts = None
    frames_written = 0

    while True:
        item = _write_queue.get()

        if item is None:
            # Flush and close
            if writer is not None:
                writer.release()
                writer = None
                video_start_ts = None
                frames_written = 0
            _write_queue.task_done()
            continue

        ts, frame = item

        # Lazy-create writer on first frame
        if writer is None:
            with _record_lock:
                path = _record_state["path"]
                fps = _record_state["fps"]
            if path:
                h, w = frame.shape[:2]
                frame_wh = (w, h)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(path, fourcc, fps, frame_wh)
                if not writer.isOpened():
                    print(f"[ERROR] Could not open video writer: {path}")
                    writer = None
                else:
                    video_start_ts = float(ts)
                    frames_written = 0

        if writer is not None:
            # Keep output duration aligned with wall clock by pacing writes to timestamps.
            if video_start_ts is None:
                video_start_ts = float(ts)
            elapsed = max(0.0, float(ts) - float(video_start_ts))
            target_frames = int(elapsed * float(fps)) + 1
            if target_frames <= frames_written:
                target_frames = frames_written + 1

            while frames_written < target_frames:
                writer.write(frame)
                frames_written += 1

        _write_queue.task_done()


def _get_tts_engine_path() -> str | None:
    for engine in ("espeak", "espeak-ng"):
        p = shutil.which(engine)
        if p:
            return p
    return None


def _render_tts_audio_locked(video_path: str, duration_sec: float):
    if not _record_state["audio_enabled"]:
        return None, False

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        _record_state["audio_error"] = "ffmpeg not found; video only"
        return None, False

    tts_engine = _get_tts_engine_path()
    if not tts_engine:
        _record_state["audio_error"] = "espeak/espeak-ng not found; video only"
        return None, False

    events = list(_record_state["audio_events"])
    if not events:
        return None, False

    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    stamp     = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    tmp_files = []

    for i, evt in enumerate(events):
        text = str(evt.get("text", "")).strip()
        if not text:
            continue
        wav_i = os.path.join(RECORDINGS_DIR, f".tts_{stamp}_{i}.wav")
        res = subprocess.run(
            [tts_engine, "-s", VOICE_RATE, "-w", wav_i, text],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if res.returncode == 0 and os.path.exists(wav_i) and os.path.getsize(wav_i) > 128:
            delay_ms = max(0, int(float(evt.get("t", 0.0)) * 1000.0))
            tmp_files.append((delay_ms, wav_i))

    if not tmp_files:
        _record_state["audio_error"] = "No narration clips generated"
        return None, False

    base, _   = os.path.splitext(video_path)
    audio_path = f"{base}_tts.wav"

    cmd = [
        ffmpeg, "-y",
        "-f", "lavfi",
        "-t", f"{max(0.5, float(duration_sec)):.3f}",
        "-i", f"anullsrc=r={RECORD_AUDIO_RATE}:cl=mono",
    ]
    for _, wav_i in tmp_files:
        cmd += ["-i", wav_i]

    filters    = []
    mix_inputs = ["[0:a]"]
    for i, (delay_ms, _) in enumerate(tmp_files, start=1):
        label = f"a{i}"
        filters.append(f"[{i}:a]adelay={delay_ms}|{delay_ms}[{label}]")
        mix_inputs.append(f"[{label}]")
    filters.append(f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:normalize=0[aout]")
    cmd += [
        "-filter_complex", ";".join(filters),
        "-map", "[aout]",
        "-ac", "1", "-ar", str(RECORD_AUDIO_RATE),
        audio_path,
    ]

    result = subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for _, wav_i in tmp_files:
        try:
            os.remove(wav_i)
        except OSError:
            pass

    if result.returncode != 0 or not os.path.exists(audio_path) or os.path.getsize(audio_path) < 128:
        _record_state["audio_error"] = "Failed to build narration audio"
        return None, False

    _record_state["audio_path"] = audio_path
    return audio_path, True


def _mux_audio_video_locked(video_path: str, audio_path: str | None = None):
    use_audio = audio_path or _record_state["audio_path"]
    if not _record_state["audio_enabled"] or not use_audio:
        return video_path, False
    if not os.path.exists(use_audio) or os.path.getsize(use_audio) < 128:
        return video_path, False

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return video_path, False

    base, ext   = os.path.splitext(video_path)
    merged_path = f"{base}_av{ext}"
    cmd = [
        ffmpeg, "-y",
        "-i", video_path,
        "-i", use_audio,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        merged_path,
    ]
    result = subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode != 0 or not os.path.exists(merged_path):
        return video_path, False

    os.replace(merged_path, video_path)
    return video_path, True
