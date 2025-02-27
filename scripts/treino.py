from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="datasets/data/data.yaml", epochs=150, imgsz=640, batch=16)