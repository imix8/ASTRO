import os
import time
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

def image_predict():
    # Path to your image directory
    IMAGE_DIR = './datasets/valid/images'
    MODEL_PATH = 'runs/detect/train/weights/best.pt'

    # Load the YOLOv8 model
    model = YOLO(MODEL_PATH)

    # Supported image formats
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    # Find all images in the directory
    image_files = sorted([
        os.path.join(IMAGE_DIR, f)
        for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith(image_extensions)
    ])

    print(f"Found {len(image_files)} images.")

    elapsed_times = []

    # Run inference on each image
    for idx, image_path in enumerate(image_files):
        print(f"Processing: {image_path}")
        start = time.perf_counter()
        results = model(image_path, verbose=False)[0]  # First batch element
        end = time.perf_counter()

        elapsed = end - start
        elapsed_times.append(elapsed)
        print(f"Inference time: {elapsed:.4f} seconds")

        # Annotate and show image
        annotated = results.plot()
        cv2.imshow("YOLOv8 Inference", annotated)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    # Close window
    cv2.destroyAllWindows()

    # Plotting inference time per image
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(elapsed_times)), elapsed_times, marker='o')
    plt.xlabel("Image Index")
    plt.ylabel("Inference Time (seconds)")
    plt.title("YOLOv8 Inference Time per Image")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    image_predict()
