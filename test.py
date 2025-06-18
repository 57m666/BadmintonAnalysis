from ultralytics import YOLO

model = YOLO('src/models/weights/best.pt')

model.predict('videos/test1.mp4', show=True)