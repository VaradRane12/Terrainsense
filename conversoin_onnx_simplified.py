import onnx
from onnxsim import simplify

model = onnx.load("best.onnx")
model_simp, check = simplify(model)

onnx.save(model_simp, "best_simplified.onnx")