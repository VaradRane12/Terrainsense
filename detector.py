import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter


class OptimizedYOLOv8TFLite:
    def __init__(self, model_path, conf_thresh=0.25, iou_thresh=0.45, num_threads=4):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        self.interpreter = Interpreter(model_path=model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()

        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        shape = self.input_details[0]['shape']   # [1, H, W, 3]
        self.input_h   = shape[1]
        self.input_w   = shape[2]
        self.input_idx = self.input_details[0]['index']
        self.output_idx= self.output_details[0]['index']

        # Pre-allocated float32 buffer — reused every frame, no GC pressure
        self.preproc_buf = np.empty((1, self.input_h, self.input_w, 3), dtype=np.float32)

    # ── Sigmoid LUT (class-level, built once) ────────────────────────────
    _SIG_X   = np.linspace(-10, 10, 2048)
    _SIG_LUT = 1.0 / (1.0 + np.exp(-_SIG_X))

    @classmethod
    def _fast_sigmoid(cls, x):
        return np.interp(x, cls._SIG_X, cls._SIG_LUT)

    def preprocess(self, frame):
        resized = cv2.resize(frame, (self.input_w, self.input_h),
                             interpolation=cv2.INTER_LINEAR)
        np.copyto(self.preproc_buf[0], resized.astype(np.float32) * (1.0 / 255.0))
        return self.preproc_buf

    def infer(self, preprocessed):
        self.interpreter.set_tensor(self.input_idx, preprocessed)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_idx)

    def postprocess(self, output, orig_h, orig_w):
        preds = output[0]                        # [8, 2100]
        boxes_raw  = preds[:4, :].T              # [2100, 4]
        scores_raw = preds[4:, :].T              # [2100, num_classes]

        # Pre-filter in logit space — sigmoid only on survivors
        logit_thresh = np.log(self.conf_thresh / (1.0 - self.conf_thresh + 1e-9))
        mask = scores_raw.max(axis=1) > logit_thresh
        if not mask.any():
            return [], [], []

        filtered_scores = self._fast_sigmoid(scores_raw[mask])
        filtered_boxes  = boxes_raw[mask]

        conf      = filtered_scores.max(axis=1)
        class_ids = filtered_scores.argmax(axis=1)

        cx, cy, w, h = (filtered_boxes[:, i] for i in range(4))
        sx = orig_w / self.input_w
        sy = orig_h / self.input_h
        x1 = ((cx - w / 2) * sx).astype(np.int32)
        y1 = ((cy - h / 2) * sy).astype(np.int32)
        x2 = ((cx + w / 2) * sx).astype(np.int32)
        y2 = ((cy + h / 2) * sy).astype(np.int32)
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), conf.tolist(),
            self.conf_thresh, self.iou_thresh
        )
        if len(indices) == 0:
            return [], [], []

        idx = indices.flatten()
        return boxes[idx], conf[idx], class_ids[idx]
