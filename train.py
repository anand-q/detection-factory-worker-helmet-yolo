from ultralytics import YOLO


model = YOLO(model="./models/yolov8n.pt")
model.train(data='data.yaml', epochs = 50, imgsz = 640, batch = 16)