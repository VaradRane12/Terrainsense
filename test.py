# test.py — corrected
from tflite_runtime.interpreter import Interpreter
import numpy as np, time

interp = Interpreter("best_saved_model/best_int8.tflite", num_threads=4)
interp.allocate_tensors()

inp = interp.get_input_details()
out = interp.get_output_details()

print("Input shape:", inp[0]['shape'])
print("Input dtype:", inp[0]['dtype'])   # float32
print("Output shape:", out[0]['shape'])

# ← float32 dummy, values in [0.0, 1.0]
dummy = np.zeros(inp[0]['shape'], dtype=np.float32)

interp.set_tensor(inp[0]['index'], dummy)

times = []
for _ in range(20):
    t = time.time()
    interp.invoke()
    times.append(time.time() - t)

print(f"Avg inference: {np.mean(times)*1000:.1f}ms")
print(f"Max FPS possible: {1/np.mean(times):.1f}")
