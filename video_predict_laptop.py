import cv2
import time
import torch
from ultralytics import RTDETR

# --- CUDA Check ---
print(f"CUDA Available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Prevent automatic CUDA memory release
##torch.cuda.empty_cache = lambda: None
torch.cuda.set_device(0)

MODEL_NAME = 'runs/detect/train/weights/best.pt'
model = RTDETR(MODEL_NAME)
model.to(torch.device('cuda'))
# model.model.half()

# --- Initialize Webcam ---
print("Starting webcam...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # --- Perform Inference using Ultralytics ---
    results = model(frame, verbose=False) # verbose=False silences Ultralytics console output per frame

    # --- Process and Draw Results ---
    annotated_frame = results[0].plot()

    # --- Display the Resulting Frame ---
    cv2.imshow("RTDETR Object Detection", annotated_frame)
    key = cv2.waitKey(1) & 0xFF # Wait 1ms for a key press
    if key == ord('q'):        # Press 'q' to quit
        print("Exiting...")
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
print("Done.")
