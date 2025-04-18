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
model.model.half()

# --- Initialize Webcam ---
print("Starting webcam...")
cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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


# import cv2
# import time
# import torch
# import numpy as np
# from ultralytics import RTDETR

# # --- CUDA Setup ---
# print(f"CUDA Available: {torch.cuda.is_available()}")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
# torch.cuda.empty_cache = lambda: None  # Prevent cleanup between frames

# # --- Configuration ---
# MODEL_NAME = 'runs/detect/train/weights/best.pt'
# IMAGE_SIZE = 960

# # --- Load Model ---
# model = RTDETR(MODEL_NAME)
# model.to(device)

# # Class name lookup
# class_names = model.names if hasattr(model, 'names') else {}

# # --- Initialize Webcam ---
# print("Starting webcam...")
# # cap = cv2.VideoCapture("/dev/video0")
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()
# print("Webcam started. Press 'q' to quit.")

# tracker = None
# init_once = False
# lost_counter = 0
# max_lost_frames = 30
# confidence = 0
# class_name = "Unknown"

# while True:
#     success, frame = cap.read()
#     if not success:
#         print("Error: Failed to grab frame.")
#         break

#     detections_image = frame.copy()

#     if not init_once:
#         # Run inference with Ultralytics RTDETR
#         results = model(frame, imgsz=IMAGE_SIZE, device=device, verbose=False)
#         boxes = results[0].boxes

#         if boxes is not None and len(boxes) > 0:
#             box = boxes[0]
#             x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
#             w, h = x2 - x1, y2 - y1

#             if w > 5 and h > 5:
#                 print(f"[INFO] Initializing tracker with box: x={x1}, y={y1}, w={w}, h={h}")
#                 tracker = cv2.TrackerCSRT_create()
#                 tracker.init(frame, (x1, y1, w, h))
#                 confidence = float(box.conf[0])
#                 class_id = int(box.cls[0])
#                 class_name = class_names.get(class_id, "Unknown")
#                 init_once = True

#                 # Draw initial detection
#                 cv2.rectangle(detections_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                 label = f"{class_name} {confidence:.2f}"
#                 cv2.putText(detections_image, label, (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
#             else:
#                 print("[WARN] Ignoring detection with invalid size")
#     else:
#         success, box = tracker.update(frame)

#         if success:
#             x, y, w, h = [int(v) for v in box]
#             frame_h, frame_w = frame.shape[:2]

#             if (
#                 0 <= x < frame_w and 0 <= y < frame_h and
#                 x + w <= frame_w and y + h <= frame_h and
#                 w > 10 and h > 10
#             ):
#                 lost_counter = 0
#                 cv2.rectangle(detections_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                 label = f"Tracking: {class_name}, {confidence:.2f}"
#                 cv2.putText(detections_image, label, (x, y - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#             else:
#                 lost_counter += 1
#                 cv2.putText(detections_image, f"Lost ({lost_counter})", (20, 60),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
#         else:
#             lost_counter += 1
#             cv2.putText(detections_image, f"Lost ({lost_counter})", (20, 60),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

#         if lost_counter >= max_lost_frames:
#             print("[INFO] Tracker lost for too long, resetting...")
#             tracker = None
#             init_once = False
#             lost_counter = 0

#     # Show frame
#     cv2.imshow("RTDETR Tracking", detections_image)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         print("Exiting...")
#         break

# cap.release()
# cv2.destroyAllWindows()
# print("Done.")
