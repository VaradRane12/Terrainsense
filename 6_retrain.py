from ultralytics import YOLO

model = YOLO("best.pt")   # previous trained model

model.train(
    data="terrain_dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=64,
    device="mps",
    workers=8,

    mosaic=1.0,
    mixup=0.2,
    degrees=10,
    translate=0.1,
    scale=0.5,
    fliplr=0.5
)