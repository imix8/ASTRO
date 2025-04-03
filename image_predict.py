import time
import supervision as sv
import numpy as np
from rfdetr import RFDETRBase
import matplotlib.pyplot as plt
import torch

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Load dataset using the COCO annotations.
ds = sv.DetectionDataset.from_coco(
    images_directory_path="./dataset/valid",
    annotations_path="./dataset/valid/_annotations.coco.json",
)

# Initialize the model.
model = RFDETRBase(pretrain_weights="./logs/checkpoint_best_total.pth")

# List to store elapsed times for each image
elapsed_times = []

# Process each image in the dataset.
for idx, (path, image, annotations) in enumerate(ds):
    # Determine the image resolution.
    # If image is a numpy array, extract width and height from its shape.
    if isinstance(image, np.ndarray):
        resolution_wh = (image.shape[1], image.shape[0])
    else:
        resolution_wh = image.size  # PIL Image returns a tuple (width, height)

    # Start timing before prediction
    start_time = time.perf_counter()

    # Predict detections.
    detections = model.predict(image, threshold=0.5)

    # Stop timing after prediction
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    elapsed_times.append(elapsed_time)
    print(f"Image {idx} took {elapsed_time:.4f} seconds for prediction.")

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
    detections_image = image.copy()
    detections_image = bbox_annotator.annotate(detections_image, detections)
    detections_image = label_annotator.annotate(detections_image, detections, detections_labels)

    # Display the annotated image.
    sv.plot_image(detections_image)

# Plotting the elapsed times for each image after processing all images.
plt.figure(figsize=(10, 6))
plt.plot(range(len(elapsed_times)), elapsed_times, marker='o', linestyle='-')
plt.xlabel('Image Index')
plt.ylabel('Elapsed Time (seconds)')
plt.title('Prediction Time per Image')
plt.grid(True)
plt.show()
