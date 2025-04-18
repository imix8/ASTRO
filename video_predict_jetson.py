import cv2
import time
import torch
from ultralytics import RTDETR

# --- CUDA Check ---
print(f"CUDA Available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Prevent automatic CUDA memory release
torch.cuda.empty_cache = lambda: None

# --- Configuration ---
MODEL_NAME = 'runs/detect/train/weights/best.pt'
BATCH_SIZE = 2
IMAGE_SIZE = 960  # Increase resolution to use more memory

# --- Load the RTDETR model ---
try:
    model = RTDETR(MODEL_NAME)
    model.to(device)
    model.model.float()  # Force float32 instead of FP16
    print(f"Successfully loaded model: {MODEL_NAME}")
except Exception as e:
    print(f"Error loading RTDETR model: {e}")
    print("Ensure you have internet access for the first run to download the model,")
    print("or place the model file in the correct directory if already downloaded.")
    exit()

# --- Initialize Webcam ---
print("Starting webcam...")
cap = cv2.VideoCapture("/dev/video0")  # Corrected path

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started. Press 'q' to quit.")
prev_time = 0

# --- Main Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # --- Simulate batch input ---
    frames = [frame for _ in range(BATCH_SIZE)]

    # --- Perform Inference using Ultralytics ---
    results = model(frames, imgsz=IMAGE_SIZE, device=device, verbose=False)

    # --- Process and Draw Results for First Frame Only ---
    annotated_frame = results[0].plot()

    # --- Calculate and Display FPS ---
    current_time = time.time()
    if prev_time > 0:
        try:
            fps = 1 / (current_time - prev_time)
            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
        except ZeroDivisionError:
            pass
    prev_time = current_time

    # --- Display the Resulting Frame ---
    cv2.imshow("RTDETR Object Detection", annotated_frame)

    # --- Print GPU Memory Usage ---
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"GPU Memory: Allocated = {allocated:.2f} MB | Reserved = {reserved:.2f} MB")

    # --- Exit Condition ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Exiting...")
        break

# --- Cleanup ---
print("Releasing resources...")
cap.release()
cv2.destroyAllWindows()
print("Done.")
