import cv2
from ultralytics import YOLO

# Load pre-trained YOLOv8 Nano
model = YOLO("yolov8n.pt")

# Path to your video file
video_path = "footpath.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file")
    exit()

print("Running YOLO on video... Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Run detection
    results = model(frame, conf=0.4)

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Video Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()