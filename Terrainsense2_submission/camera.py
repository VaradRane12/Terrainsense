"""
camera.py
─────────
Thread layout
─────────────
    Thread A  _camera_capture_worker   Picamera2 → raw_frame_queue   (I/O bound)
    Thread B  _inference_worker         raw_frame_queue → model → result_frame_queue
    Flask     generate_frames()         result_frame_queue → MJPEG bytes

Both queues use maxsize=6 for smoother buffering on Pi 4.
If inference falls behind, capture drops frames (never blocks).

Note: When app is paused (not running), frames are still captured and streamed
      but inference is skipped for efficiency.
"""

import queue
import threading
import time

import cv2
import numpy as np

import inference
import inference_fps
import recording
import speech
import app_state

from config import (
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    INFERENCE_DROP_OLD_FRAMES,
    STABILIZATION_ENABLED,
    STABILIZATION_MAX_SHIFT_PX,
    STREAM_JPEG_QUALITY,
)

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None

# ── Frame queues ─────────────────────────────────────────────────────────────
raw_frame_queue:    queue.Queue = queue.Queue(maxsize=6)
result_frame_queue: queue.Queue = queue.Queue(maxsize=6)

# ── Camera singleton ──────────────────────────────────────────────────────────
_camera_lock = threading.Lock()
_camera_state: dict = {"instance": None, "users": 0}

# ── Lifecycle flags ───────────────────────────────────────────────────────────
_running = threading.Event()
_stabilization_enabled = STABILIZATION_ENABLED
_stabilization_lock = threading.Lock()



def set_stabilization_enabled(enabled: bool) -> dict:
    global _stabilization_enabled
    with _stabilization_lock:
        _stabilization_enabled = bool(enabled)
    return {"ok": True, "stabilization_enabled": _stabilization_enabled}


def get_stabilization_status() -> dict:
    with _stabilization_lock:
        return {"ok": True, "stabilization_enabled": _stabilization_enabled}


# ── Public API ────────────────────────────────────────────────────────────────
def start_pipeline() -> None:
    """
    Acquire the camera, then start the capture and inference threads.
    Call once at app startup.
    """
    if Picamera2 is None:
        raise RuntimeError("Picamera2 is not installed. Install it and restart the app.")

    cam = _acquire_camera()
    _running.set()

    threading.Thread(
        target=_camera_capture_worker,
        args=(cam,),
        daemon=True,
        name="camera-capture",
    ).start()

    threading.Thread(
        target=_inference_worker,
        daemon=True,
        name="inference",
    ).start()

    print("[INFO] Camera pipeline started (capture + inference threads)")


def stop_pipeline() -> None:
    """Signal both pipeline threads to stop."""
    _running.clear()


def generate_frames():
    """
    Flask MJPEG generator — pulls annotated frames from result_frame_queue
    and encodes them as JPEG.  Nearly zero CPU in this thread.
    """
    while True:
        frame = result_frame_queue.get()
        _, buffer = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), max(50, min(95, int(STREAM_JPEG_QUALITY)))],
        )
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )


# ── Worker threads ────────────────────────────────────────────────────────────
def _camera_capture_worker(picam2) -> None:
    """
    Thread A: capture frames as fast as the sensor allows.
    Drops frames if inference is behind — never blocks.
    """
    print("[INFO] Capture thread running")
    try:
        while _running.is_set():
            frame = picam2.capture_array()
            if frame is None:
                continue

            # Normalise colour format to BGR
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            try:
                raw_frame_queue.put_nowait(frame)
            except queue.Full:
                pass   # drop — inference is busy, camera should never stall
    finally:
        _release_camera()
        print("[INFO] Capture thread exited")


def _inference_worker() -> None:
    """
    Thread B: pull raw frames, run TFLite, draw OSD, enqueue for streaming.
    Also hands frames to the recording writer (non-blocking).
    When app is paused, frames are still streamed but inference is skipped.
    """
    print("[INFO] Inference thread running")
    prev_gray_small = None
    while _running.is_set() or not raw_frame_queue.empty():
        try:
            frame = raw_frame_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        # Keep only the newest frame to reduce lag and improve effective stream FPS.
        if INFERENCE_DROP_OLD_FRAMES:
            while True:
                try:
                    frame = raw_frame_queue.get_nowait()
                except queue.Empty:
                    break

        frame, prev_gray_small = _stabilize_frame_if_enabled(frame, prev_gray_small)

        # Check if app is running (inference enabled)
        app_running = app_state.get_running()

        if app_running:
            # Run model
            output = inference.run_inference(frame)
            label, color, conf, guidance, speech_text = inference.postprocess(output, frame)

            # Track FPS
            inference_fps.record_frame()

            # Hand annotated frame to recording writer (non-blocking)
            recording.maybe_enqueue_frame(frame)

            # Enqueue speech (non-blocking, rate-limited inside speech.py)
            speech.queue_speech(speech_text)
        else:
            # App paused: stream raw/stabilized frame without inference
            pass

        # Push result to MJPEG stream (inference or raw frame)
        try:
            result_frame_queue.put_nowait(frame)
        except queue.Full:
            pass   # drop — browser is reading slowly, that's fine


def _stabilize_frame_if_enabled(frame, prev_gray_small):
    with _stabilization_lock:
        enabled = _stabilization_enabled
    if not enabled:
        return frame, None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    if prev_gray_small is None:
        return frame, small

    try:
        (dx_s, dy_s), _ = cv2.phaseCorrelate(prev_gray_small.astype("float32"), small.astype("float32"))
        dx = -float(dx_s) * 4.0
        dy = -float(dy_s) * 4.0
        max_shift = float(STABILIZATION_MAX_SHIFT_PX)
        dx = max(-max_shift, min(max_shift, dx))
        dy = max(-max_shift, min(max_shift, dy))
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        stabilized = cv2.warpAffine(frame, m, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR)
        return stabilized, small
    except Exception:
        return frame, small


# ── Camera singleton helpers ──────────────────────────────────────────────────
def _acquire_camera():
    with _camera_lock:
        if _camera_state["instance"] is None:
            print("[INFO] Starting Picamera2...")
            picam2 = Picamera2()
            try:
                cfg = picam2.create_video_configuration(
                    main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"},
                    controls={"FrameRate": float(CAMERA_FPS)},
                )
                picam2.configure(cfg)
                picam2.start()
                time.sleep(0.2)
            except Exception:
                try:    picam2.stop()
                except Exception: pass
                try:    picam2.close()
                except Exception: pass
                raise
            _camera_state["instance"] = picam2
            print("[INFO] Picamera2 started")

        _camera_state["users"] += 1
        return _camera_state["instance"]


def _release_camera() -> None:
    with _camera_lock:
        if _camera_state["users"] > 0:
            _camera_state["users"] -= 1
        if _camera_state["users"] == 0 and _camera_state["instance"] is not None:
            picam2 = _camera_state["instance"]
            _camera_state["instance"] = None
            try:    picam2.stop()
            except Exception: pass
            try:    picam2.close()
            except Exception: pass
            print("[INFO] Picamera2 released")
