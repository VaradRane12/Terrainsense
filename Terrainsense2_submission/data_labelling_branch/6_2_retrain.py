from ultralytics import YOLO

DATA = "terrain_dataset/data.yaml"

# ---------- 1. SCRATCH ----------
model = YOLO("/Users/varad/Documents/Programming /Terrainsense/runs/detect/runs_dual/scratch_1003/weights/best.pt")

model.train(
    resume = True,
    data=DATA,
    epochs=40,
    imgsz=640,
    batch=16,
    device="mps",
    workers=4,
    cache=True,

    lr0=0.0003,
    lrf=0.01,
    warmup_epochs=3,

    mosaic=0.7,
    mixup=0.2,
    fliplr=0.5,
    degrees=5,
    translate=0.05,
    scale=0.3,
    shear=2.0,
    close_mosaic=10,

    cls=0.7,
    box=7.5,
    dfl=1.5,
    weight_decay=0.0005,

    patience=20,

    project="runs_dual",
    name="scratch_100"
)

# ---------- 2. RESUME ----------
# model = YOLO("/Users/varad/Documents/Programming /Terrainsense/runs/detect/train10/weights/best.pt")

# model.train(
#     data=DATA,
#     epochs=100,
#     imgsz=640,
#     batch=16,
#     device="mps",
#     workers=4,
#     cache=True,

#     lr0=0.0003,
#     lrf=0.01,
#     warmup_epochs=2,

#     mosaic=0.5,
#     mixup=0.1,

#     cls=0.7,
#     weight_decay=0.0005,

#     patience=20,

#     project="runs_dual",
#     name="resume_100"
# )