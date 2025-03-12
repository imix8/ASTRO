import ultralytics
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
if __name__ == '__main__':
    # Train the model
    model.train(data = 'config.yaml', epochs = 150, patience = 30, batch = 32, imgsz = 416)

    # Export the model to TensorFlow Lite format (ensure that the model is fully trained)
    # The exported file (e.g., "yolov8n.tflite") must then be compiled for the Edge TPU.
    model.export(format='tflite', device='cpu', half=False)
    
    # After export, compile the TFLite model with the Edge TPU compiler in your terminal:
    # edgetpu_compiler yolov8n.tflite