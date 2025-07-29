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

counter = 0
line_y = 200
while True: 
    isTrue, frame = capture.read()
    
    frame_resized = rescaleFrame(frame, scale=0.75)
    outputs = model.track(frame_resized, persist=True)
    cv.line(frame_resized, (0,line_y), (640,line_y), color = (0, 0 , 255))
    
    for o in outputs: 
        if o.boxes.id is not None:
            for box in o.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                id = int(box.id[0])
                
                
                if class_name == 'car':
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 =  int(x1), int (y1), int(x2), int (y2)
                    cv.rectangle(frame_resized, (x1, y1), (x2,y2), color = (205,255,0), thickness = 2)
                    box_center = int((y1 + y2) / 2)
                    
                    
                    if line_y - 10 < box_center < line_y + 10 and id not in cars_counted:
                        counter += 1
                        cars_counted.append(id)
    
    
    cv.putText(frame_resized, f"Counter: {counter}", (20, 30), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(205,255,0), thickness=2)

    cv.imshow('Cars', frame_resized)
    if cv.waitKey(1) & 0xFF==ord('d'):
        break



capture.release()
cv.destroyAllWindows()




