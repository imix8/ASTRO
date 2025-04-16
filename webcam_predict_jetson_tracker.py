import cv2
import numpy as np
import supervision as sv
from rfdetr import RFDETRBase

def run_detection_with_tracking():
    # Load dataset using the COCO annotations.
    ds = sv.DetectionDataset.from_coco(
        images_directory_path="./dataset/valid",
        annotations_path="./dataset/valid/_annotations.coco.json",
    )

    model = RFDETRBase(pretrain_weights="./logs/checkpoint_best_total.pth")

    cap = cv2.VideoCapture("/dev/video0")
    tracker = None
    init_once = False

    while True:
        success, frame = cap.read()
        if not success:
            break

        resolution_wh = (frame.shape[1], frame.shape[0]) if isinstance(frame, np.ndarray) else frame.size
        detections_image = frame.copy()

        if not init_once:
            detections = model.predict(frame, threshold=0.5)
            
            if detections.xyxy.shape[0] > 0:
                x1, y1, x2, y2 = detections.xyxy[0].astype(int)
                w, h = x2 - x1, y2 - y1

                # Validate box size
                if w > 5 and h > 5:
                    print(f"[INFO] Initializing tracker with box: x={x1}, y={y1}, w={w}, h={h}")

                    # Draw initial detection box for visual verification
                    cv2.rectangle(detections_image, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

                    # Initialize CSRT tracker
                    tracker = cv2.legacy.TrackerCSRT_create()
                    tracker.init(frame, (x1, y1, w, h))
                    class_id = detections.class_id[0]
                    confidence = detections.confidence[0]
                    class_name = ds.classes[class_id]
                    init_once = True

                    # Annotate initial detection
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
            success, box = tracker.update(frame)
             # Track consecutive failure frames
            if 'lost_counter' not in locals():
                lost_counter = 0  # First-time init

            if success:
                x, y, w, h = [int(v) for v in box]
                frame_h, frame_w = frame.shape[:2]

                if (
                    0 <= x < frame_w and 0 <= y < frame_h and
                    x + w <= frame_w and y + h <= frame_h and
                    w > 10 and h > 10
                ):
                    lost_counter = 0  # Reset counter on valid update
                    cv2.rectangle(detections_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label = f"Tracking: {class_name}, {confidence:.2f}"
                    cv2.putText(detections_image, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    lost_counter += 1
                    cv2.putText(detections_image, f"Lost ({lost_counter})", (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                lost_counter += 1
                cv2.putText(detections_image, f"Lost ({lost_counter})", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Reset after too many lost frames
            max_lost_frames = 30  # ~1 seconds at 30 FPS
            if lost_counter >= max_lost_frames:
                print("[INFO] Tracker lost for too long, resetting...")
                tracker = None
                init_once = False
                lost_counter = 0

        cv2.imshow("Webcam Tracking", detections_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_detection_with_tracking()