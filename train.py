from ultralytics import YOLO


model = YOLO(model="./models/yolo11n.pt")
model.train(data='data.yaml', epochs = 50, imgsz = 640, batch = 16)