from ultralytics import RTDETR

model = RTDETR("rtdetr-l.pt")

results = model.train(data="datasets/data.yaml", epochs=50, imgsz=640)
