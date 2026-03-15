import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
from VAE import CNN_VAE

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = CNN_VAE(latent_dim=1024)
model.load_state_dict(torch.load(
    "vae_latent1024_lr1e-05_bs128_kld1.0_data100.0_20260309_000149_best.pth",
    map_location=device
))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((240, 320)),
    transforms.ToTensor()
])

cap = cv2.VideoCapture("footpath.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        recon, mu, logvar, z, encoded = model(img)

    recon = F.interpolate(recon, size=img.shape[2:], mode="bilinear", align_corners=False)

    error = torch.mean((img - recon) ** 2, dim=1).squeeze().cpu().numpy()

    error = cv2.resize(error, (frame.shape[1], frame.shape[0]))

    error_norm = cv2.normalize(error, None, 0, 255, cv2.NORM_MINMAX)
    error_norm = error_norm.astype(np.uint8)

    # higher threshold
    _, thresh = cv2.threshold(error_norm, 80, 255, cv2.THRESH_BINARY)

    kernel = np.ones((7,7), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame_area = frame.shape[0] * frame.shape[1]

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # ignore very small noise
        if area < 2000:
            continue

        # ignore huge detections
        if area > frame_area * 0.25:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # ignore extremely wide boxes
        if w > frame.shape[1] * 0.6:
            continue

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 3)

    cv2.imshow("Hazard Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 