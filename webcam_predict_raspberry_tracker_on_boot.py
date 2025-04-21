import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"  # Headless-safe for Qt

import cv2
import numpy as np
import supervision as sv
from rfdetr import RFDETRBase
import serial
import time

def send_to_arduino(arduino, cmd, last_sent_time, cooldown=0.8):
    if time.time() - last_sent_time > cooldown:
        try:
            arduino.reset_input_buffer()
            arduino.reset_output_buffer()
            arduino.write((cmd + '\n').encode('utf-8'))
            arduino.flush()
            print(f"[SEND] Sent command: {cmd}")
            return time.time()
        except serial.SerialTimeoutException as e:
            print(f"[ERROR] Serial timeout occurred while sending command: {e}")
        except serial.SerialException as e:
            print(f"[ERROR] Serial communication error: {e}")
        except Exception as e:
            print(f"[ERROR] Failed to send to Arduino: {e}")
    return last_sent_time


def run_detection_with_tracking():
    print("[INFO] Loading dataset and model...")
    ds = sv.DetectionDataset.from_coco(
        images_directory_path="./dataset/valid",
        annotations_path="./dataset/valid/_annotations.coco.json",
    )

    model = RFDETRBase(pretrain_weights="./logs/checkpoint_best_total.pth")
    print("[INFO] Model loaded successfully.")

    cap = cv2.VideoCapture('/dev/video0')
    if not cap.isOpened():
        print("[ERROR] Unable to access the webcam.")
        return

    tracker = None
    init_once = False

    try:
        arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
        time.sleep(2)
        last_sent_time = time.time()
        print("[INFO] Arduino connected on /dev/ttyACM0.")
    except Exception as e:
        arduino = None
        last_sent_time = 0
        print(f"[WARN] Arduino not connected: {e}")

    frame_counter = 0

    while True:
        success, frame = cap.read()
        if not success:
            print("[ERROR] Failed to read frame from webcam.")
            break
        
        resolution_wh = (frame.shape[1], frame.shape[0]) if isinstance(frame, np.ndarray) else frame.size
        detections_image = frame.copy()

        if not init_once:
            if (frame_counter % 30 == 0):
                detections = model.predict(frame, threshold=0.6)

                if detections.xyxy.shape[0] > 0:
                    x1, y1, x2, y2 = detections.xyxy[0].astype(int)
                    w, h = x2 - x1, y2 - y1

                    if w > 5 and h > 5:
                        print(f"[INFO] Detection found: {ds.classes[detections.class_id[0]]} ({w}x{h})")
                        tracker = cv2.TrackerKCF_create()
                        tracker.init(frame, (x1, y1, w, h))

                        class_id = detections.class_id[0]
                        confidence = detections.confidence[0]
                        class_name = ds.classes[class_id]

                        if arduino:
                            if class_name == "plastic_bottle":
                                last_sent_time = send_to_arduino(arduino, "sorting1", last_sent_time)
                            elif class_name == "aluminum_can":
                                last_sent_time = send_to_arduino(arduino, "sorting2", last_sent_time)
                            elif class_name == "paper_cup":
                                last_sent_time = send_to_arduino(arduino, "sorting1", last_sent_time)
                            elif class_name == "face_mask":
                                last_sent_time = send_to_arduino(arduino, "sorting2", last_sent_time)

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
                frame_counter = (frame_counter + 1) % 30
        else:
            success, box = tracker.update(frame)

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

                    center_x = x + w // 2
                    center_y = y + h // 2
                    region_left = frame_w // 3
                    region_right = 2 * frame_w // 3

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

                    if arduino:
                        last_sent_time = send_to_arduino(arduino, cmd, last_sent_time)
                        time.sleep(0.1)

                    print(f"[TRACKING] Tracking {class_name} ({confidence:.2f}) - Action: {cmd}")
                else:
                    lost_counter += 1
                    print(f"[WARN] Tracker lost target - Lost count: {lost_counter}")
            else:
                lost_counter += 1
                print(f"[WARN] Tracker failed update - Lost count: {lost_counter}")

            if lost_counter >= 15:
                print("[INFO] Lost target for 15 frames — resetting tracker.")
                tracker = None
                init_once = False
                lost_counter = 0
                frame_counter = 1

        # GUI-related lines removed for headless operation
        # cv2.imshow("Webcam Tracking", detections_image)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    if arduino:
        arduino.close()
        print("[INFO] Closed Arduino connection.")

    print("[INFO] Detection system shutting down.")

if __name__ == '__main__':
    run_detection_with_tracking()
