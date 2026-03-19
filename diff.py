import cv2
from ultralytics import YOLO

# Load YOLO model (lightweight)
model = YOLO("yolov8n.pt")

# Phone camera stream
url = "http://192.168.29.56:8080/video"
cap = cv2.VideoCapture(url)

# Get frame size
ret, frame = cap.read()
height, width, _ = frame.shape

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    # Draw all detections
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            label = model.names[cls]

            # Highlight dogs differently (optional)
            if label == "dog":
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)

            # Draw box + label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

    # Show live feed
    cv2.imshow("YOLO Detection Feed", frame)

    # Record ALL frames (continuous recording)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()