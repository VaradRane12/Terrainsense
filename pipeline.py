import cv2
import threading
import queue
import time
from collections import deque
from detector import OptimizedYOLOv8TFLite


class AsyncPipeline:
    def __init__(self, model_path, source=0, conf_thresh=0.25, iou_thresh=0.45):
        self.detector = OptimizedYOLOv8TFLite(
            model_path, conf_thresh=conf_thresh, iou_thresh=iou_thresh
        )
        self.source  = source
        self.cap     = None

        # maxsize=1 → always process latest frame, drop stale ones
        self.raw_q    = queue.Queue(maxsize=1)
        self.result_q = queue.Queue(maxsize=1)

        self.running      = False
        self.latest_frame = None
        self.fps_deque    = deque(maxlen=30)
        self.current_fps  = 0.0
        self.detection_count = 0

    # ── Camera setup ──────────────────────────────────────────────────────
    def _open_camera(self):
        cap = cv2.VideoCapture(self.source, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS,          30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # no stale frame buffering
        if not cap.isOpened():
            # Fallback: try default backend
            cap = cv2.VideoCapture(self.source)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    # ── Thread 1: Capture ─────────────────────────────────────────────────
    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            try:
                self.raw_q.put_nowait(frame)
            except queue.Full:
                pass   # drop frame — prefer latency over buffering

    # ── Thread 2: Inference ───────────────────────────────────────────────
    def _inference_loop(self):
        while self.running:
            try:
                frame = self.raw_q.get(timeout=0.5)
            except queue.Empty:
                continue

            h, w = frame.shape[:2]
            inp  = self.detector.preprocess(frame)
            raw  = self.detector.infer(inp)
            boxes, confs, cls_ids = self.detector.postprocess(raw, h, w)

            try:
                self.result_q.put_nowait((frame, boxes, confs, cls_ids))
            except queue.Full:
                pass

    # ── Thread 3: Render ──────────────────────────────────────────────────
    def _render_loop(self):
        prev_time = time.time()
        COLORS = [
            (0, 255, 128), (0, 180, 255), (255, 100, 0),
            (255, 0, 150), (180, 255, 0), (0, 255, 255),
        ]
        while self.running:
            try:
                frame, boxes, confs, cls_ids = self.result_q.get(timeout=0.5)
            except queue.Empty:
                continue

            self.detection_count = len(boxes)

            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                x1, y1, x2, y2 = box
                color = COLORS[int(cls_id) % len(COLORS)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"cls{cls_id}: {conf:.2f}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(frame, (x1, y1 - lh - 6), (x1 + lw, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

            # FPS overlay
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            self.fps_deque.append(fps)
            prev_time = now
            self.current_fps = sum(self.fps_deque) / len(self.fps_deque)

            cv2.putText(frame, f"FPS: {self.current_fps:.1f}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(frame, f"FPS: {self.current_fps:.1f}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 1)

            self.latest_frame = frame

    # ── Public API ────────────────────────────────────────────────────────
    def start(self):
        self.cap     = self._open_camera()
        self.running = True
        for target in (self._capture_loop, self._inference_loop, self._render_loop):
            t = threading.Thread(target=target, daemon=True)
            t.start()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

    def get_jpeg(self, quality=70):
        """Returns latest frame as JPEG bytes, or None."""
        frame = self.latest_frame
        if frame is None:
            return None
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes()
