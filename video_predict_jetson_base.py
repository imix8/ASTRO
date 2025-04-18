import cv2
import torch
from ultralytics import RTDETR

# --- CUDA Check ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA Available: {torch.cuda.is_available()}")

# --- Model Setup ---
model = RTDETR('runs/detect/train/weights/best.pt')
model.to(device)
model.model.half()

# --- Initialize Webcam ---
cap = cv2.VideoCapture("/dev/video0")

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started. Press 'q' to quit.")


while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Optional: Resize to smaller image for speed (320x320)
    frame_resized = cv2.resize(frame, (320, 320))

    # Convert to half precision (fp16)
    frame_resized = frame_resized.astype('float16')

    # Inference
    results = model(frame_resized, imgsz=320, device=device, verbose=False)

    # Annotate (optional – this is slow!)
    annotated = results[0].plot()

    # Display
    cv2.imshow("RTDETR (Fast)", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()