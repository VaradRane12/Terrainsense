from ultralytics import YOLO
import onnx

# Step 1 — Export
print("[1/2] Exporting best.pt to ONNX...")
model = YOLO("final.pt")
model.export(
    format="onnx",
    imgsz=320,
    opset=12,
    simplify=True
)

# Step 2 — Verify
print("[2/2] Verifying ONNX model...")
onnx_model = onnx.load("final.onnx")
onnx.checker.check_model(onnx_model)
print("✅ best.onnx is valid and ready!")