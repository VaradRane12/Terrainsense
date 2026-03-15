from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="terrain_dataset/data.yaml",
    epochs=200,
    imgsz=640,
    batch=16,

    device="mps",   # Apple Silicon GPU

    mosaic=1.0,
    mixup=0.2,
    degrees=10,
    translate=0.1,
    scale=0.5,
    fliplr=0.5
)