import cv2
import numpy as np

VIDEO_PATH = "walk_test.mp4"  # or use 0 for webcam

cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 🔽 Resize for speed (VERY important for Raspberry Pi)
    frame = cv2.resize(frame, (640, 480))

    h, w, _ = frame.shape

    # Focus on lower region (walking area)
    roi = frame[int(h*0.6):h, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    edge_pixels = np.count_nonzero(edges)
    total_pixels = edges.shape[0] * edges.shape[1]
    edge_density = edge_pixels / total_pixels

    # Classification logic
    if edge_density < 0.01:
        label = "POSSIBLE DROP / STEP"
        color = (0, 0, 255)
    elif edge_density < 0.03:
        label = "OBSTACLE / UNEVEN"
        color = (0, 165, 255)
    else:
        label = "FLAT / SAFE"
        color = (0, 255, 0)

    # Draw ROI box
    cv2.rectangle(frame, (0, int(h*0.6)), (w, h), color, 2)

    cv2.putText(frame, label, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("TerrainSense", frame)
    cv2.imshow("Edges", edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()