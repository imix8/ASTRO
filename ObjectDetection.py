import cv2
import numpy as np
import random
from tflite_runtime.interpreter import Interpreter, load_delegate

# Load the class names from list.txt
with open("list.txt", "r") as f:
    class_list = [line.strip() for line in f if line.strip()]

# Generate random detection colors for each class
detection_colors = []
for _ in range(len(class_list)):
    detection_colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

# Load the compiled TensorFlow Lite model with the Edge TPU delegate.
# Make sure the model file is the one produced by the edgetpu_compiler (e.g., "yolov8n_edgetpu.tflite")
model_path = "yolov8n_edgetpu.tflite"

# Run for standard performance: sudo apt-get install libedgetpu1-std
# Run for maximum performance: sudo apt-get install libedgetpu1-max
interpreter = Interpreter(
    model_path=model_path,
    experimental_delegates=[load_delegate('libedgetpu.so.1')]
)
interpreter.allocate_tensors()

# Retrieve model input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Determine the expected input shape (height, width)
input_shape = input_details[0]['shape'][1:3]

# When trying to detect objects in the video
# cap = cv2.VideoCapture("Video1.mp4")

# When trying to detect objects through my camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Preprocess the frame: resize, convert to RGB, and normalize.
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[0]))
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(rgb_frame, axis=0).astype(np.float32) / 255.0

    # Set the input tensor and run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve the detection results.
    # This example assumes that the output tensor has a shape similar to [1, num_detections, 6]
    # where each detection is in the form: [x1, y1, x2, y2, score, class]
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    # Process each detection
    for detection in output_data:
        # Assumes the output tensor is shaped like [1, num_detections, 6] where each detection
        # is in the form of [x1, y1, x2, y2, score, class].
        x1, y1, x2, y2, score, class_id = detection
        if score < 0.45:
            continue

        # Scale the detection coordinates to the original frame dimensions
        h_ratio = frame.shape[0] / input_shape[0]
        w_ratio = frame.shape[1] / input_shape[1]
        x1 = int(x1 * w_ratio)
        y1 = int(y1 * h_ratio)
        x2 = int(x2 * w_ratio)
        y2 = int(y2 * h_ratio)

        class_id = int(class_id)
        color = detection_colors[class_id % len(detection_colors)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        label = f"{class_list[class_id]} {score:.3f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('ObjectDetection', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()