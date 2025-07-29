import torch 
from ultralytics import YOLO
import matplotlib.pyplot as plt 
import cv2 as cv

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = YOLO('yolov8n.pt')
cars_counted = []
counter = 0
def rescaleFrame(frame, scale =1):
    width = 640
    height = 360
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
capture = cv.VideoCapture("Videos/Traffic.mp4")



while True: 
    isTrue, frame = capture.read()
    
    frame_resized = rescaleFrame(frame, scale=0.75)
    outputs = model.track(frame_resized, persist=True)
    cv.line(frame_resized, (0,200), (600,200))
    for o in outputs: 
        for box in o.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            if class_name == 'car':
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 =  int(x1), int (y1), int(x2), int (y2)
                cv.rectangle(frame_resized, (x1, y1), (x2,y2), color = (205,255,0), thickness = 2)
            
    
    
    cv.imshow('Cars', frame_resized)
    if cv.waitKey(1) & 0xFF==ord('d'):
        break



capture.release()
cv.destroyAllWindows()




