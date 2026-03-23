import onnx
from onnx_tf.backend import prepare

# Load ONNX model
onnx_model = onnx.load("final.onnx")

# Prepare TF representation
tf_rep = prepare(onnx_model)

# Export as TF SavedModel
tf_rep.export_graph("saved_model")