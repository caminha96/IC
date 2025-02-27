from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="datasets/data/data.yaml", epochs=30, imgsz=640)