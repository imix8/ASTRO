import cv2
import numpy as np
import supervision as sv
from rfdetr import RFDETRBase

# Load dataset using the COCO annotations.
ds = sv.DetectionDataset.from_coco(
    images_directory_path="./dataset/valid",
    annotations_path="./dataset/valid/_annotations.coco.json",
)

model = RFDETRBase(pretrain_weights="./logs/checkpoint_best_total.pth")

cap = cv2.VideoCapture(0)
while True:
    success, frame = cap.read()
    if not success:
        break

    if isinstance(frame, np.ndarray):
        resolution_wh = (frame.shape[1], frame.shape[0])
    else:
        resolution_wh = frame.size  # PIL Image returns a tuple (width, height)

    detections = model.predict(frame, threshold=0.5)
    
    # Calculate optimal text scale and line thickness based on the image resolution.
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)

    # Create annotators for bounding boxes and labels.
    bbox_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_color=sv.Color.BLACK,
        text_scale=text_scale,
        text_thickness=thickness,
        smart_position=True
    )

    # Generate detection labels.
    detections_labels = [
        f"{ds.classes[class_id]} {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]

    # Annotate the image with bounding boxes and labels.
    detections_image = frame.copy()
    detections_image = bbox_annotator.annotate(detections_image, detections)
    detections_image = label_annotator.annotate(detections_image, detections, detections_labels)

    cv2.imshow("Webcam", detections_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()