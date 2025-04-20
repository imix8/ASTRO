import cv2
import numpy as np
import supervision as sv
from rfdetr import RFDETRBase
import serial
import time

def send_to_arduino(arduino, cmd, last_sent_time, cooldown=0.8):
    if time.time() - last_sent_time > cooldown:
        try:
            # Clear input and output buffers before sending the command
            arduino.reset_input_buffer()
            arduino.reset_output_buffer()

            # print(f"[DEBUG] Time since last sent: {time.time() - last_sent_time}")
            # print(f"[DEBUG] Command to send: {cmd}")
            arduino.write((cmd + '\n').encode('utf-8'))
            arduino.flush()
            print(f"[SEND] Sent command: {cmd}")
            return time.time(), cmd  # Updated timestamp and last command sent
        except serial.SerialTimeoutException as e:
            print(f"[ERROR] Serial timeout occurred while sending command: {e}")
        except serial.SerialException as e:
            print(f"[ERROR] Serial communication error: {e}")
        except Exception as e:
            print(f"[ERROR] Failed to send to Arduino: {e}")
    return last_sent_time


def run_detection_with_tracking():
    # === Load COCO dataset ===
    # print("[DEBUG] Loading dataset...")
    ds = sv.DetectionDataset.from_coco(
        images_directory_path="./dataset/valid",
        annotations_path="./dataset/valid/_annotations.coco.json",
    )
    # print("[DEBUG] Dataset loaded.")

    # === Load RFDETR model ===
    # print("[DEBUG] Loading model...")
    model = RFDETRBase(pretrain_weights="./logs/checkpoint_best_total.pth")
    # print("[DEBUG] Model loaded.")

    cap = cv2.VideoCapture('/dev/video0')
    tracker = None
    init_once = False

    # === Connect to Arduino ===
    try:
        # print("[DEBUG] Attempting to connect to Arduino...")
        arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)  # Set timeout to 1 second
        time.sleep(2)
        # print("[INFO] Serial connection established.")
        last_sent_time = time.time()
    except Exception as e:
        # print(f"[ERROR] Could not connect to Arduino: {e}")
        arduino = None
        last_sent_time = 0

    while True:
        # === Capture frame ===
        success, frame = cap.read()
        if not success:
            # print("[ERROR] Failed to read frame from camera.")
            break

        resolution_wh = (frame.shape[1], frame.shape[0]) if isinstance(frame, np.ndarray) else frame.size
        detections_image = frame.copy()

        if not init_once:
            # === Run model prediction ===
            # print("[DEBUG] Running detection...")
            detections = model.predict(frame, threshold=0.5)
            # print(f"[DEBUG] Detection output: {detections.xyxy}")

            if detections.xyxy.shape[0] > 0:
                x1, y1, x2, y2 = detections.xyxy[0].astype(int)
                w, h = x2 - x1, y2 - y1
                # print(f"[DEBUG] Detected box: ({x1}, {y1}, {x2}, {y2})")

                if w > 5 and h > 5:
                    # print(f"[INFO] Initializing tracker with box: x={x1}, y={y1}, w={w}, h={h}")
                    cv2.rectangle(detections_image, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

                    # === Create and init tracker ===
                    # print("[DEBUG] Creating tracker...")
                    tracker = cv2.TrackerKCF_create()
                    tracker.init(frame, (x1, y1, w, h))
                    # print("[DEBUG] Tracker initialized.")

                    class_id = detections.class_id[0]
                    confidence = detections.confidence[0]
                    class_name = ds.classes[class_id]
                    init_once = True

                    bbox_annotator = sv.BoxAnnotator(thickness=2)
                    label_annotator = sv.LabelAnnotator(
                        text_color=sv.Color.BLACK,
                        text_scale=sv.calculate_optimal_text_scale(resolution_wh),
                        text_thickness=2,
                        smart_position=True
                    )
                    detections_labels = [
                        f"{ds.classes[class_id]} {confidence:.2f}"
                        for class_id, confidence in zip(detections.class_id, detections.confidence)
                    ]
                    detections_image = bbox_annotator.annotate(detections_image, detections)
                    detections_image = label_annotator.annotate(detections_image, detections, detections_labels)
                else:
                    print("[WARN] Ignoring detection with invalid size")
        else:
            # === Update tracker ===
            # print("[DEBUG] Updating tracker...")
            success, box = tracker.update(frame)
            # print(f"[DEBUG] Tracker success: {success}, Box: {box}")

            if 'lost_counter' not in locals():
                lost_counter = 0

            if success:
                x, y, w, h = [int(v) for v in box]
                frame_h, frame_w = frame.shape[:2]

                if (
                    0 <= x < frame_w and 0 <= y < frame_h and
                    x + w <= frame_w and y + h <= frame_h and
                    w > 10 and h > 10
                ):
                    lost_counter = 0
                    cv2.rectangle(detections_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label = f"Tracking: {class_name}, {confidence:.2f}"
                    cv2.putText(detections_image, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    center_x = x + w // 2
                    center_y = y + h // 2
                    region_left = frame_w // 3
                    region_right = 2 * frame_w // 3

                    # === Generate command ===
                    # print("[DEBUG] Generating movement command...")
                    if center_y < 60:
                        cmd = "backward"
                    elif center_x < region_left:
                        cmd = "left"
                    elif center_x > region_right:
                        cmd = "right"
                    elif y + h > frame_h - 50:
                        cmd = "stop_servo"
                    else:
                        cmd = "forward"
                    # print(f"[DEBUG] Command determined: {cmd}")

                    # === Send command ===
                    # Limit to 1 command every 800ms
                    if arduino:
                        last_sent_time = send_to_arduino(arduino, cmd, last_sent_time)
                        time.sleep(0.1)  # Optional: Add small delay after sending the command
                else:
                    lost_counter += 1
                    # print(f"[WARN] Invalid tracker box or out of bounds. Lost counter: {lost_counter}")
                    cv2.putText(detections_image, f"Lost ({lost_counter})", (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                lost_counter += 1
                # print(f"[WARN] Tracker update failed. Lost counter: {lost_counter}")
                cv2.putText(detections_image, f"Lost ({lost_counter})", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # === Reset tracker if lost too long ===
            if lost_counter >= 30:
                # print("[INFO] Tracker lost for too long, resetting...")
                tracker = None
                init_once = False
                lost_counter = 0

        # === Show the result ===
        cv2.imshow("Webcam Tracking", detections_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # print("[INFO] Quit key pressed. Exiting loop.")
            break

    # === Cleanup ===
    # print("[INFO] Releasing resources...")
    cap.release()
    cv2.destroyAllWindows()
    if arduino:
        arduino.close()
    # print("[INFO] Shutdown complete.")

if __name__ == '__main__':
    run_detection_with_tracking()
