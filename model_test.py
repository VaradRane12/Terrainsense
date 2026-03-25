# test.py
from tflite_runtime.interpreter import Interpreter
import numpy as np

interp = Interpreter("best_saved_model/best_int8.tflite", num_threads=4)
interp.allocate_tensors()

inp = interp.get_input_details()
out = interp.get_output_details()

print("Input shape:", inp[0]['shape'])   # should be [1, 320, 320, 3]
print("Input dtype:", inp[0]['dtype'])   # should be int8
print("Output shape:", out[0]['shape'])  # should be [1, 8, 2100]

# Benchmark raw inference speed
dummy = np.zeros(inp[0]['shape'], dtype=np.int8)
interp.set_tensor(inp[0]['index'], dummy)

import time
times = []
for _ in range(20):
    t = time.time()
    interp.invoke()
    times.append(time.time() - t)

print(f"Avg inference: {np.mean(times)*1000:.1f}ms")
print(f"Max FPS possible: {1/np.mean(times):.1f}")
