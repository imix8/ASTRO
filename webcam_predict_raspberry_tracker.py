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
            
            # Send the command to the Arduino
            arduino.write((cmd + '\n').encode('utf-8'))
            arduino.flush()
            print(f"[SEND] Sent command: {cmd}")
            
            return time.time()  # updated timestamp
        except serial.SerialTimeoutException as e:
            print(f"[ERROR] Serial timeout occurred while sending command: {e}")
        except serial.SerialException as e:
            print(f"[ERROR] Serial communication error: {e}")
        except Exception as e:
            print(f"[ERROR] Failed to send to Arduino: {e}")
    return last_sent_time


def run_detection_with_tracking():
    # === Load COCO dataset ===
    ds = sv.DetectionDataset.from_coco(
        images_directory_path="./dataset/valid",
        annotations_path="./dataset/valid/_annotations.coco.json",
    )

    # === Load RFDETR model ===
    model = RFDETRBase(pretrain_weights="./logs/checkpoint_best_total.pth")

    cap = cv2.VideoCapture('/dev/video0')
    tracker = None
    init_once = False

    # === Connect to Arduino ===
    try:
        arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
        time.sleep(2)
        last_sent_time = time.time()
    except Exception as e:
        arduino = None
        last_sent_time = 0

    frame_counter = 0

    while True:
        success, frame = cap.read()
        if not success:
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
                        cv2.rectangle(detections_image, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

                        tracker = cv2.TrackerKCF_create()
                        tracker.init(frame, (x1, y1, w, h))

                        class_id = detections.class_id[0]
                        confidence = detections.confidence[0]
                        class_name = ds.classes[class_id]

                        # === NEW: Send sorting command based on detected class ===
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
                # print(f"frame_counter: {frame_counter}")
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
                    cv2.rectangle(detections_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label = f"Tracking: {class_name}, {confidence:.2f}"
                    cv2.putText(detections_image, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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
                else:
                    lost_counter += 1
                    cv2.putText(detections_image, f"Lost ({lost_counter})", (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                lost_counter += 1
                cv2.putText(detections_image, f"Lost ({lost_counter})", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if lost_counter >= 15:
                tracker = None
                init_once = False
                lost_counter = 0
                frame_counter = 1

        cv2.imshow("Webcam Tracking", detections_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if arduino:
        arduino.close()

if __name__ == '__main__':
    run_detection_with_tracking()
