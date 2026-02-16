from picamera2 import Picamera2
import cv2

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (640, 480)}
))
picam2.start()

frame = picam2.capture_array()
print(frame.shape)

cv2.imwrite("test.jpg", frame)
