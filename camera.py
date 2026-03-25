"""
camera.py
─────────
Thread layout
─────────────
  Thread A  _camera_capture_worker   Picamera2 → raw_frame_queue   (I/O bound)
  Thread B  _inference_worker         raw_frame_queue → model → result_frame_queue
  Flask     generate_frames()         result_frame_queue → MJPEG bytes

Both queues have maxsize=2.  If inference falls behind, the camera
thread drops frames (never blocks).  The MJPEG generator blocks until a
result is ready, ensuring the client always gets the freshest frame.
"""

import queue
import threading
import time

import cv2

import inference
import inference_fps
import recording
import speech

from config import CAMERA_HEIGHT, CAMERA_WIDTH

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None

# ── Frame queues ─────────────────────────────────────────────────────────────
raw_frame_queue:    queue.Queue = queue.Queue(maxsize=2)
result_frame_queue: queue.Queue = queue.Queue(maxsize=2)

# ── Camera singleton ──────────────────────────────────────────────────────────
_camera_lock = threading.Lock()
_camera_state: dict = {"instance": None, "users": 0}

# ── Lifecycle flags ───────────────────────────────────────────────────────────
_running = threading.Event()


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
        _, buffer = cv2.imencode(".jpg", frame)
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
    """
    print("[INFO] Inference thread running")
    while _running.is_set() or not raw_frame_queue.empty():
        try:
            frame = raw_frame_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        # Run model
        output = inference.run_inference(frame)
        label, color, conf, guidance, speech_text = inference.postprocess(output, frame)

        # Track FPS
        inference_fps.record_frame()

        # Hand annotated frame to recording writer (non-blocking)
        recording.maybe_enqueue_frame(frame)

        # Enqueue speech (non-blocking, rate-limited inside speech.py)
        speech.queue_speech(speech_text)

        # Push result to MJPEG stream
        try:
            result_frame_queue.put_nowait(frame)
        except queue.Full:
            pass   # drop — browser is reading slowly, that's fine


# ── Camera singleton helpers ──────────────────────────────────────────────────
def _acquire_camera():
    with _camera_lock:
        if _camera_state["instance"] is None:
            print("[INFO] Starting Picamera2...")
            picam2 = Picamera2()
            try:
                cfg = picam2.create_video_configuration(
                    main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"}
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
