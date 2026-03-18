from ultralytics import YOLO

model = YOLO("/Users/varad/Documents/Programming /Terrainsense/runs/detect/train9/weights/last.pt")

model.train(
    data="terrain_dataset/data.yaml",
    resume=True,
    epochs=80,
    imgsz=640,
    batch=8,
    device="mps",
    workers=4,

    lr0=0.0005,
    lrf=0.01,
    warmup_epochs=2,

    mosaic=1.0,
    mixup=0.0,
    fliplr=0.5,
    degrees=10,
    translate=0.1,
    scale=0.5,
    close_mosaic=10,

    patience=20,
    save_period=5,
)