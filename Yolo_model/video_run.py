from ultralytics import YOLO

# load trained model
model = YOLO("../runs/detect/train6/weights/best.pt")

# run detection
model.predict(
    source="walk_test.mp4",
    show=True,
    conf=0.3,
    save=True
)