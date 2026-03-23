from ultralytics import YOLO

model = YOLO("best.pt")

model.export(
    format="tflite",
    imgsz=320,        # match your target input size
    int8=True,        # enables INT8 quantization
    data="terrain_dataset/data.yaml"  # calibration dataset — use your own .yaml for best accuracy
)