import ultralytics
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
if __name__ == '__main__':
    model.train(data = 'config.yaml', epochs = 150, patience = 30, batch = 32, imgsz = 416)