import cv2
import subprocess
import numpy as np
import json

# start ToF sensor
sensor = subprocess.Popen(
    ["python", "sensor_stream.py"],
    stdout=subprocess.PIPE,
    text=True
)

# start camera stream
camera = subprocess.Popen(
    ["rpicam-vid", "-t", "0", "--width", "640", "--height", "480", "--codec", "yuv420", "-o", "-"],
    stdout=subprocess.PIPE
)

distance = 0
w = 640
h = 480
frame_size = w * h * 3 // 2

while True:

    # read sensor data
    try:
        line = sensor.stdout.readline()
        data = json.loads(line)

        distances = data["distances"]
        valid = [d for d in distances if d > 0]

        if valid:
            distance = min(valid)

    except:
        pass

    # read camera frame
    raw = camera.stdout.read(frame_size)

    if len(raw) != frame_size:
        continue

    frame = np.frombuffer(raw, dtype=np.uint8).reshape((h * 3 // 2, w))
    frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)

    # overlay distance
    cv2.putText(
        frame,
        f"Distance: {distance} mm",
        (30,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("TerrainSense Demo", frame)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
