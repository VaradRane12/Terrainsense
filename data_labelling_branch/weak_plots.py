from ultralytics import YOLO


model = YOLO("/Users/varad/Documents/Programming /Terrainsense/runs/detect/train9/weights/best.pt")
metrics = model.val(data="terrain_dataset/data.yaml", plots=True)