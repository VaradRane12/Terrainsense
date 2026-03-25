"""
inference.py
────────────
Owns the TFLite interpreter (loaded once at import time) and all detection logic:
  preprocess()   — resize + normalise + quantise
  run_inference() — invoke the model
  postprocess()  — NMS, corridor filter, person fall detection, OSD drawing

All sensor data is read via sensor.get_sensor_snapshot() so this module
never blocks on I/O.
"""

import math
import time

import cv2
import numpy as np

import sensor
from config import (
    CLASS_LABELS,
    CONF_THRESH,
    FALL_ASPECT_THRESHOLD,
    FALL_HEIGHT_DROP_RATIO,
    IGNORE_FAR_MM,
    INPUT_SIZE,
    MODEL_PATH,
    MODEL_DELEGATE,
    MODEL_INPUT_SCALE,
    MODEL_NUM_THREADS,
    NMS_THRESH,
    PERSON_MOVE_PX_PER_SEC,
    TOF_ALERT_MM,
    WALK_CORRIDOR_WIDTH_RATIO,
    YAW_SHIFT_SCALE,
)

# ── Load interpreter ────────────────────────────────────────────────────────
INTERPRETER_BACKEND = None
try:
    from tflite_runtime.interpreter import Interpreter
    INTERPRETER_BACKEND = "tflite_runtime"
except ImportError:
    try:
        from ai_edge_litert.interpreter import Interpreter
        INTERPRETER_BACKEND = "ai_edge_litert"
    except ImportError as exc:
        raise ImportError(
            "No TFLite interpreter found. Install tflite-runtime or ai-edge-litert."
        ) from exc

print("[INFO] Loading TFLite model...")
_interpreter = Interpreter(model_path=MODEL_PATH, num_threads=MODEL_NUM_THREADS)
_interpreter.allocate_tensors()
print(f"[INFO] Interpreter backend : {INTERPRETER_BACKEND}")
print(f"[INFO] Model threads       : {MODEL_NUM_THREADS} (Pi 4 optimization)")
if MODEL_INPUT_SCALE != 1.0:
    print(f"[INFO] Input scale         : {MODEL_INPUT_SCALE:.2f}x (speed-up at cost of accuracy)")
if MODEL_DELEGATE != "auto" and MODEL_DELEGATE != "cpu":
    print(f"[INFO] Delegate            : {MODEL_DELEGATE}")

_input_details  = _interpreter.get_input_details()
_output_details = _interpreter.get_output_details()
_input_dtype    = _input_details[0]["dtype"]
print(f"[INFO] Model input dtype   : {_input_dtype}")
print("[INFO] ✅ TFLite model loaded!")

# ── Per-person motion tracking (module-level, single person) ────────────────
_person_state: dict = {
    "last_center_x": None,
    "last_height":   None,
    "last_time":     0.0,
}


# ── Public API ───────────────────────────────────────────────────────────────
def preprocess(frame: np.ndarray) -> np.ndarray:
    # Apply resolution scaling if configured (speeds up inference on Pi 4)
    target_size = int(INPUT_SIZE * MODEL_INPUT_SCALE)
    img = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    # If scaled smaller than model expects, pad to maintain aspect ratio
    if target_size < INPUT_SIZE:
        pad = (INPUT_SIZE - target_size) // 2
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)        # HWC → CHW
    img = np.expand_dims(img, axis=0)   # add batch dim

    if _input_dtype == np.int8:
        scale, zero_point = _input_details[0]["quantization"]
        img = (img / scale + zero_point).astype(np.int8)
    elif _input_dtype == np.uint8:
        img = (img * 255).astype(np.uint8)

    return img


def run_inference(frame: np.ndarray) -> list:
    input_data = preprocess(frame)
    _interpreter.set_tensor(_input_details[0]["index"], input_data)
    _interpreter.invoke()
    output = _interpreter.get_tensor(_output_details[0]["index"])

    if _output_details[0]["dtype"] == np.int8:
        scale, zero_point = _output_details[0]["quantization"]
        output = (output.astype(np.float32) - zero_point) * scale

    return [output]


def postprocess(output: list, frame: np.ndarray):
    """
    Returns (label, color, conf, guidance_text, speech_text).
    Also draws bounding boxes and OSD text directly onto *frame* (in-place).
    """
    snap        = sensor.get_sensor_snapshot()
    now         = time.time()
    h_f, w_f    = frame.shape[:2]
    corr_x1, corr_x2 = _walking_corridor(w_f, snap)

    preds         = output[0].squeeze(0).T
    class_scores  = _sigmoid(preds[:, 4:])
    confidences   = np.max(class_scores, axis=1)
    class_ids     = np.argmax(class_scores, axis=1)
    mask          = confidences > CONF_THRESH

    if not np.any(mask):
        _handle_no_detection(frame, snap, w_f, h_f, corr_x1, corr_x2)
        if snap.get("front_mm") is not None and snap["front_mm"] < TOF_ALERT_MM:
            return (None, (0, 165, 255), 0.0,
                    f"TOF obstacle {snap['front_mm']} mm",
                    f"Stop. obstacle at {snap['front_mm']} millimeters")
        return None, (0, 255, 0), 0.0, "Path clear", None

    scale_x = w_f / INPUT_SIZE
    scale_y = h_f / INPUT_SIZE

    boxes_xyxy, confs_f, classes_f = [], [], []
    for i, box in enumerate(preds[:, :4][mask]):
        cx, cy, bw, bh = box
        x1 = max(0,   int((cx - bw / 2) * scale_x))
        y1 = max(0,   int((cy - bh / 2) * scale_y))
        x2 = min(w_f, int((cx + bw / 2) * scale_x))
        y2 = min(h_f, int((cy + bh / 2) * scale_y))

        if not _is_in_corridor(x1, x2, corr_x1, corr_x2):
            continue
        obj_mm = _tof_mm_for_bbox(snap, x1, x2, w_f)
        if obj_mm is not None and obj_mm > IGNORE_FAR_MM:
            continue

        boxes_xyxy.append([x1, y1, x2, y2])
        confs_f.append(float(confidences[mask][i]))
        classes_f.append(int(class_ids[mask][i]))

    if not boxes_xyxy:
        _handle_no_detection(frame, snap, w_f, h_f, corr_x1, corr_x2)
        return None, (0, 255, 0), 0.0, "Path clear (corridor)", None

    boxes_nms = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes_xyxy]
    indices   = cv2.dnn.NMSBoxes(boxes_nms, confs_f, CONF_THRESH, NMS_THRESH)
    if len(indices) == 0:
        _handle_no_detection(frame, snap, w_f, h_f, corr_x1, corr_x2)
        if snap.get("front_mm") is not None and snap["front_mm"] < TOF_ALERT_MM:
            return (None, (0, 165, 255), 0.0,
                    f"TOF obstacle {snap['front_mm']} mm",
                    f"Stop. obstacle at {snap['front_mm']} millimeters")
        return None, (0, 255, 0), 0.0, "Path clear", None

    best_i          = max(indices.flatten(), key=lambda i: confs_f[i])
    x1, y1, x2, y2 = boxes_xyxy[best_i]
    conf            = float(confs_f[best_i])
    label, color    = CLASS_LABELS.get(int(classes_f[best_i]), ("UNKNOWN", (255, 255, 255)))

    guidance_text, speech_text = _direction_guidance(label, x1, x2, w_f, snap)

    if label == "PERSON":
        guidance_text, speech_text, color = _handle_person(
            x1, y1, x2, y2, w_f, h_f, snap, now, guidance_text, speech_text, color
        )

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    text          = f"{label}  {conf:.2f}"
    (tw, th), _   = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, text, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    _draw_osd(frame, snap, w_f, h_f, conf, guidance_text, label, color, corr_x1, corr_x2)
    return label, color, conf, guidance_text, speech_text


def draw_tof_grid(frame: np.ndarray, snap: dict) -> None:
    dists = snap.get("distances", [0] * 64)
    stats = snap.get("status",   [0] * 64)
    h, w  = frame.shape[:2]
    cell  = 14
    gap   = 2
    size  = (cell + gap) * 8 + gap
    x0    = w - size - 18
    y0    = 70

    cv2.rectangle(frame, (x0 - 8, y0 - 24), (x0 + size + 8, y0 + size + 8), (20, 20, 20), -1)
    cv2.putText(frame, "ToF 8x8", (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    for r in range(8):
        for c in range(8):
            i   = r * 8 + c
            col = _tof_cell_color(dists[i], stats[i])
            x1  = x0 + gap + c * (cell + gap)
            y1  = y0 + gap + r * (cell + gap)
            x2, y2 = x1 + cell, y1 + cell
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 30, 30), 1)

    # Highlight centre 4×4 region
    cx1 = x0 + gap + 2 * (cell + gap)
    cy1 = y0 + gap + 2 * (cell + gap)
    cx2 = x0 + gap + 6 * (cell + gap) - gap
    cy2 = y0 + gap + 6 * (cell + gap) - gap
    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (255, 255, 255), 1)


# ── Internal helpers ─────────────────────────────────────────────────────────
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _tof_cell_color(d_mm: int, st: int) -> tuple:
    if st != 5 or d_mm <= 0:
        return (45, 45, 45)
    d = max(200, min(3000, int(d_mm)))
    t = (d - 200) / 2800.0
    return (0, int(255 * t), int(255 * (1.0 - t)))


def _tof_mm_for_bbox(snap: dict, x1: int, x2: int, frame_w: int) -> int | None:
    dists = snap.get("distances", [0] * 64)
    stats = snap.get("status",   [0] * 64)
    c0    = max(0, min(7, int((x1 / max(1, frame_w)) * 8)))
    c1    = max(0, min(7, int((x2 / max(1, frame_w)) * 8)))
    if c1 < c0:
        c0, c1 = c1, c0
    vals = [
        int(dists[r * 8 + c])
        for r in range(8)
        for c in range(c0, c1 + 1)
        if r * 8 + c < len(dists) and r * 8 + c < len(stats)
        and stats[r * 8 + c] == 5 and dists[r * 8 + c] > 0
    ]
    if not vals:
        return snap.get("front_mm")
    return int(np.median(vals))


def _tof_nearest_side(snap: dict) -> tuple[str | None, int | None]:
    dists = snap.get("distances", [0] * 64)
    stats = snap.get("status",   [0] * 64)
    bands = {"left": [0, 1, 2], "center": [3, 4], "right": [5, 6, 7]}
    best_side, best_mm = None, None
    for side, cols in bands.items():
        vals = [
            int(dists[r * 8 + c])
            for r in range(8) for c in cols
            if r * 8 + c < len(dists) and r * 8 + c < len(stats)
            and stats[r * 8 + c] == 5 and dists[r * 8 + c] > 0
        ]
        if vals:
            mm = int(np.median(vals))
            if best_mm is None or mm < best_mm:
                best_mm, best_side = mm, side
    return best_side, best_mm


def _walking_corridor(frame_w: int, snap: dict) -> tuple[int, int]:
    yaw_deg = float(snap.get("yaw_deg", 0.0))
    shift   = int(max(-0.25, min(0.25, yaw_deg / 90.0)) * frame_w * YAW_SHIFT_SCALE)
    center  = (frame_w // 2) + shift
    half    = int(max(0.2, min(0.8, WALK_CORRIDOR_WIDTH_RATIO)) * frame_w * 0.5)
    return max(0, center - half), min(frame_w - 1, center + half)


def _is_in_corridor(x1: int, x2: int, cx1: int, cx2: int) -> bool:
    return cx1 <= (0.5 * (x1 + x2)) <= cx2


def _direction_guidance(label: str, x1: int, x2: int, width: int, snap: dict) -> tuple[str, str]:
    center_x = (x1 + x2) / 2.0
    if center_x < width * 0.38:
        side, move = "left", "right"
    elif center_x > width * 0.62:
        side, move = "right", "left"
    else:
        side, move = "center", "stop"

    if move == "stop":
        guidance_text = f"{label} ahead - slow/stop"
        speech_text   = f"Stop. {label.lower()} ahead"
    else:
        guidance_text = f"{label} on {side} - move {move}"
        speech_text   = f"Move {move}. {label.lower()} ahead"

    if snap.get("front_mm") is not None and snap["front_mm"] < TOF_ALERT_MM:
        guidance_text = f"{guidance_text} | {snap['front_mm']} mm"
        if move == "stop":
            speech_text = f"Stop. {label.lower()} at {snap['front_mm']} millimeters"

    return guidance_text, speech_text


def _handle_person(x1, y1, x2, y2, w_f, h_f, snap, now,
                   guidance_text, speech_text, color):
    center_x = 0.5 * (x1 + x2)
    height   = max(1.0, float(y2 - y1))
    width    = max(1.0, float(x2 - x1))

    if center_x < w_f * 0.38:
        person_side = "left"
    elif center_x > w_f * 0.62:
        person_side = "right"
    else:
        person_side = "center"

    motion = "steady"
    prev_x = _person_state["last_center_x"]
    prev_t = _person_state["last_time"]
    if prev_x is not None and prev_t > 0.0:
        dt = max(1e-3, now - prev_t)
        vx = (center_x - prev_x) / dt
        if   vx >  PERSON_MOVE_PX_PER_SEC:  motion = "moving right"
        elif vx < -PERSON_MOVE_PX_PER_SEC:  motion = "moving left"

    aspect   = width / height
    prev_h   = _person_state["last_height"]
    drop     = prev_h is not None and height < float(prev_h) * FALL_HEIGHT_DROP_RATIO
    tilt     = abs(float(snap.get("pitch_deg", 0.0)))
    fall_risk = (aspect > FALL_ASPECT_THRESHOLD and height < h_f * 0.45) or (drop and tilt > 8.0)

    person_mm = _tof_mm_for_bbox(snap, x1, x2, w_f)
    tof_side, _ = _tof_nearest_side(snap)

    _person_state["last_center_x"] = center_x
    _person_state["last_height"]   = height
    _person_state["last_time"]     = now

    if fall_risk:
        return (
            "PERSON fall risk - assist immediately",
            "Alert. person may be falling",
            (0, 0, 255),
        )

    if person_mm is not None:
        guidance_text = f"PERSON {motion} | {person_mm} mm"
        speech_text   = f"Person {motion}. Distance {person_mm} millimeters"
    else:
        guidance_text = f"PERSON {motion}"
        speech_text   = f"Person {motion}"

    if tof_side and tof_side != person_side:
        guidance_text = f"{guidance_text} | ToF nearer {tof_side}"

    return guidance_text, speech_text, color


def _handle_no_detection(frame, snap, w_f, h_f, corr_x1, corr_x2):
    _draw_osd(frame, snap, w_f, h_f, 0.0, "Path clear", "ALL CLEAR", (0, 255, 0), corr_x1, corr_x2)


def _draw_osd(frame, snap, w_f, h_f, conf, guidance, label, color, corr_x1, corr_x2):
    """Render all on-screen display elements onto frame (in-place)."""
    from inference_fps import get_avg_fps   # lazy import to avoid circular
    avg_fps = get_avg_fps()

    display_label = label if label else "ALL CLEAR"

    cv2.rectangle(frame, (0, 0), (w_f, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"  {display_label}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)
    cv2.putText(frame, guidance, (20, h_f - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 255, 200), 2)
    cv2.putText(frame, f"Conf: {conf:.2f}  |  FPS: {avg_fps:.1f}", (20, h_f - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Walking corridor overlay
    cv2.rectangle(frame, (corr_x1, 65), (corr_x2, h_f - 70), (120, 120, 120), 1)
    cv2.putText(frame, "Walk corridor", (corr_x1 + 5, 84),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # ToF / IMU readouts
    if not snap.get("enabled", False):
        tof_text = "ToF front: disabled"
    elif snap.get("front_mm") is not None:
        tof_text = f"ToF front: {snap['front_mm']} mm"
    elif snap.get("error"):
        tof_text = "ToF front: sensor error"
    else:
        tof_text = "ToF front: no valid target"

    cv2.putText(frame, tof_text, (w_f - 265, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(frame, f"Pitch: {snap.get('pitch_deg', 0.0):.1f} deg", (w_f - 265, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(frame, f"Yaw: {snap.get('yaw_deg', 0.0):.1f} deg  Far>{IGNORE_FAR_MM}mm ignored",
                (20, h_f - 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 190, 190), 1)

    draw_tof_grid(frame, snap)
