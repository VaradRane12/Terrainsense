"""
camera.py
─────────
Process layout (Pi 4 optimization)
──────────────────────────────────
  Thread A  _camera_capture_worker   Picamera2 → raw_frame_queue   (I/O bound)
  Process B _inference_worker_proc    raw_frame_queue → model → result_frame_queue (CPU intensive)
  Flask     generate_frames()         result_frame_queue → MJPEG bytes

Uses multiprocessing for inference to break Python's GIL and exploit all 4 Pi 4 cores.
Queues have maxsize=6. Camera drops frames if inference backs up (never blocks).
"""

import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process, Queue as MPQueue

import cv2

import inference_fps
import recording
import speech

from config import CAMERA_HEIGHT, CAMERA_WIDTH

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None

# ── Frame queues ─────────────────────────────────────────────────────────────
raw_frame_queue:    MPQueue = MPQueue(maxsize=6)    # Larger buffer for Pi 4
result_frame_queue: queue.Queue = queue.Queue(maxsize=6)

# ── JPEG encoder thread pool (async encoding for better throughput) ––––––––––
_jpeg_encoder = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jpeg")

# ── Camera singleton ──────────────────────────────────────────────────────────
_camera_lock = threading.Lock()
_camera_state: dict = {"instance": None, "users": 0}

# ── Lifecycle flags ───────────────────────────────────────────────────────────
_running = threading.Event()


# ── Public API ────────────────────────────────────────────────────────────────
def start_pipeline() -> None:
    """
    Acquire the camera, then start the capture thread and inference process.
    Call once at app startup.
    
    Process-based inference breaks the GIL and uses all 4 Pi 4 cores.
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

    # Use Process instead of Thread for inference (breaks GIL on Pi 4)
    proc = Process(
        target=_inference_worker_proc,
        daemon=True,
        name="inference-proc",
    )
    proc.start()

    print("[INFO] Camera pipeline started (capture thread + inference process)")
    print("[INFO] Inference running on dedicated process (GIL-free, uses all cores)")


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
    Process worker: pull raw frames, run TFLite, draw OSD, enqueue for streaming.
    Runs in separate process to bypass Python's GIL and use all 4 Pi 4 cores.
    """
    # Pin this process to cores 1-3, leaving core 0 for camera/Flask
    try:
        os.sched_setaffinity(0, {1, 2, 3})
    except AttributeError:
        pass  # Windows/macOS don't support cpu_affinity

    # Import here to avoid loading model in main process
    import inference
    import sensor

    print("[INFO] Inference process running (pinned to cores 1-3)")
    
    while _running.is_set() or not raw_frame_queue.empty():
        try:
            frame = raw_frame_queue.get(timeout=0.5)
        except:
            continue

        # Run model (now uses full CPU without GIL)
        output = inference.run_inference(frame)
        label, color, conf, guidance, speech_text = inference.postprocess(output, frame)

        # Track FPS
        inference_fps.record_frame()

        # Hand to recording writer (non-blocking)
        recording.maybe_enqueue_frame(frame)

        # Enqueue speech (non-blocking, rate-limited)
        speech.queue_speech(speech_text)

        # Push result to MJPEG stream
        try:
            result_frame_queue.put_nowait(frame)
        except queue.Full:
            pass  # drop slow frame


# Wrapper for Process spawn compatibility
def _inference_worker_proc() -> None:
    """Wrapper to run inference in a separate process."""
    _inference_worker()


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
