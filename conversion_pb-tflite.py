import tensorflow as tf
import numpy as np
import cv2
import os

def representative_dataset():
    IMG_DIR = "terrain_dataset/train/images"
    files = os.listdir(IMG_DIR)

    for i, f in enumerate(files[:300]):
        path = os.path.join(IMG_DIR, f)

        img = cv2.imread(path)
        if img is None:
            continue

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img = cv2.resize(img, (320, 320))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) / 255.0

        # NHWC → NCHW
        img = np.transpose(img, (2, 0, 1))

        img = np.expand_dims(img, axis=0)

        # debug once
        if i == 0:
            print("Sample shape:", img.shape)

        yield [img]


# ─────────────────────────────────────────────

converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open("best_int8.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Conversion complete")