import cv2
from ultralytics import RTDETR
import time

# --- Configuration ---
MODEL_NAME = 'runs/detect/train/weights/best.pt'

# --- Load the RTDETR model ---
try:
    model = RTDETR(MODEL_NAME)
    print(f"Successfully loaded model: {MODEL_NAME}")
except Exception as e:
    print(f"Error loading RTDETR model: {e}")
    print("Ensure you have internet access for the first run to download the model,")
    print("or place the model file in the correct directory if already downloaded.")
    exit()

# --- Initialize Webcam ---
print("Starting webcam...")
cap = cv2.VideoCapture("dev/video0") # 0 is usually the default webcam

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

    # --- Perform Inference using Ultralytics ---
    results = model(frame, verbose=False) # verbose=False silences Ultralytics console output per frame

    # --- Process and Draw Results ---
    # `results[0].plot()` returns the frame with bounding boxes drawn directly
    annotated_frame = results[0].plot()

    # --- Calculate and Display FPS ---
    current_time = time.time()
    if prev_time > 0: # Avoid division by zero on the first frame
      try:
          fps = 1 / (current_time - prev_time)
          cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
      except ZeroDivisionError:
          pass # Should not happen after the first frame check, but added for safety
    prev_time = current_time


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