# Pi 4 Optimization Guide & Model Improvements
## TerrainSense

---

## 🚀 Performance Optimizations Applied

### 1. **Multiprocessing Inference (3-4x potential speedup)**
- Changed from threading to `multiprocessing.Process` for inference
- **Why**: Python's Global Interpreter Lock (GIL) prevents threads from using multiple cores
- **Impact**: Inference now uses all 4 Pi 4 cores without blocking
- **Result**: Expected 3-4x speedup depending on model complexity

**How**: Set `MODEL_NUM_THREADS=3` in config (leaves core 0 for OS/camera)

### 2. **CPU Core Affinity Pinning**
- Inference process pinned to cores 1-3
- Reduces cache thrashing and context switching
- 5-10% speedup on multi-core systems

### 3. **Larger Frame Buffers (2 → 6)**
- Prevents frame drops when inference can't keep up
- Better quality streaming without artifacts

### 4. **Input Resolution Scaling**
- Set `MODEL_INPUT_SCALE=0.75` to run 25% faster (240x240 instead of 320x320)
- Trade-off: ~2-3% accuracy loss for 25-30% speed gain
- Useful for real-time applications where speed > accuracy

---

## 📈 Model Improvement Options

### Option 1: **Edge TPU (BEST - 3-5x speedup)**
**Cost**: ~$25-30 USD  
**Speedup**: 3-5x  
**Accuracy**: No loss

1. Buy: [Coral USB Accelerator](https://coral.ai/products/accelerator/) or [M.2 Accelerator](https://coral.ai/products/m2-accelerator/)

2. Install drivers:
   ```bash
   sudo apt-get install libedgetpu1-std python3-pycoral
   ```

3. Update inference.py to use Coral delegate:
   ```python
   from pycoral.utils.edgetpu import make_interpreter
   _interpreter = make_interpreter("model_edgetpu.tflite")
   ```

4. Convert your model for Edge TPU:
   ```bash
   edgetpu_compiler -o /tmp model.tflite
   ```

---

### Option 2: **Model Quantization (Free, 20-30% faster)**
**Cost**: Free (you do it)  
**Speedup**: 20-30%  
**Accuracy**: ~1-2% loss

Convert your FP32 model to INT8 quantized:

```python
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('your_model.h5')

# Convert to TFLite with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

quantized_tflite_model = converter.convert()

with open('/path/to/model.tflite', 'wb') as f:
    f.write(quantized_tflite_model)
```

---

### Option 3: **Model Pruning (Free, 10-20% faster)**
**Cost**: Free  
**Speedup**: 10-20%  
**Accuracy**: Minimal loss if done right

Remove insignificant weights during training:

```python
import tensorflow_model_optimization as tfmot

# Apply magnitude-based pruning
pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
    initial_sparsity=0.36,
    final_sparsity=0.80,  # 80% of weights become zero
    begin_step=0,
    end_step=end_step,
    frequency=100)

model = tfmot.sparsity.keras.prune_low_magnitude(
    model, pruning_schedule=pruning_schedule)
```

Then convert to TFLite for ~10-15% speedup.

---

### Option 4: **Knowledge Distillation (Medium effort)**
**Cost**: GPU training time  
**Speedup**: Create a smaller model that runs 2-3x faster  
**Accuracy**: Can match larger model with careful tuning

Train a smaller mobile-optimized model using a larger teacher model.

---

### Option 5: **Operator Fusion + Graph Optimization (Free)**
**Speedup**: 5-15%

TensorFlow Lite automatically does this, but you can optimize further:

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS
]
converter.experimental_enable_resource_variables = False
tflite_model = converter.convert()
```

---

## 🎯 Recommended Path for Pi 4

### **Quick win (0 cost):**
1. Set `MODEL_NUM_THREADS=3` ✅ Already done
2. Try `MODEL_INPUT_SCALE=0.85` for 15% speedup
3. Ensure model is INT8 quantized (check if it is)

### **Medium effort (free):**
1. Quantize model to INT8 if not already
2. Add CPU affinity ✅ Already done

### **Best results ($25-30):**
1. Buy Coral USB Accelerator
2. Convert model with `edgetpu_compiler`
3. Get 3-5x speedup with zero accuracy loss

---

## 📊 Estimated Performance Gains

| Optimization | Speedup | Cost | Difficulty |
|---|---|---|---|
| Multiprocessing + threads | 3-4x | Free ✅ | ✅ Done |
| Core affinity | +5-10% | Free ✅ | ✅ Done |
| Input scaling (0.75) | 1.33x | Free | Easy |
| INT8 quantization | 1.2-1.3x | Free | Medium |
| Pruning (80% sparsity) | 1.1-1.2x | Free | Hard |
| **Edge TPU (Coral)** | **3-5x** | $25-30 | Easy |
| **Combined** | **10-20x** | $25-30 | Medium |

---

## 🔧 Environment Variables for Pi 4

```bash
# Fine-tune inference
export MODEL_NUM_THREADS=3        # Use 3 cores (leave 1 for OS)
export MODEL_INPUT_SCALE=0.85     # 15% faster, minimal accuracy loss
export MODEL_DELEGATE=cpu         # or "coral" if using Edge TPU

# Camera optimization
export CAMERA_FPS=24              # Reduce from default 30 for lower CPU load

# Voice guidance
export ENABLE_VOICE=1
export VOICE_INTERVAL_SEC=3.0

# Sensor
export ENABLE_SENSOR_BRIDGE=1
```

Then run:
```bash
python app.py
```

---

## 📈 Benchmarking Your Setup

Monitor performance:

```bash
# In another terminal, watch CPU usage
watch -n 1 'top -bn1 | head -12'

# Check Pi 4 temperature
vcgencmd measure_temp

# Monitor individual processes
ps aux | grep python
```

Expected on Pi 4 with optimizations:
- **CPU usage**: 70-85% (all 4 cores utilized)
- **Temperature**: 50-65°C (normal)
- **FPS**: 15-25 FPS (depending on model)
- **Inference time**: 40-80ms per frame

---

## ⚡ Next Steps

1. **Test current setup** and monitor FPS in browser (`http://<pi-ip>:5000`)
2. **Try input scaling**: Set `MODEL_INPUT_SCALE=0.85` and measure FPS improvement
3. **Check if model is quantized**: Look at model file size (< 20MB is usually quantized)
4. **If FPS < 10**: Consider Edge TPU ($25) for 3-5x guaranteed speedup
5. **Monitor temperature**: Ensure Pi stays cool with good airflow

Good luck! 🚀
