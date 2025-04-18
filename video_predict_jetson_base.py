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

MODEL_NAME = 'runs/detect/train/weights/best.pt'
model = RTDETR(MODEL_NAME)
model.to(device)
model.model.half()  # Enable FP16 acceleration

# --- Initialize Webcam ---
print("Starting webcam...")
# cap = cv2.VideoCapture("/dev/video0")

# ASSIGN CAMERA ADRESS to DEVICE HERE!
pipeline = " ! ".join(["v4l2src device=/dev/video0",
                       "video/x-raw, width=640, height=480, framerate=30/1",
                       "videoconvert",
                       "video/x-raw, format=(string)BGR",
                       "appsink"
                       ])

# --- OpenCV VideoCapture using GStreamer ---
# print("Starting webcam via GStreamer...")
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

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
    # `results[0].plot()` returns the frame with bounding boxes drawn directly
    annotated_frame = results[0].plot()


    # --- Display the Resulting Frame ---
    cv2.imshow("RTDETR Object Detection", annotated_frame)

    # --- Exit Condition ---
    key = cv2.waitKey(1) & 0xFF # Wait 1ms for a key press
    if key == ord('q'):        # Press 'q' to quit
        print("Exiting...")
        break

# --- Cleanup ---
print("Releasing resources...")
cap.release()
cv2.destroyAllWindows()
print("Done.")