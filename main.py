import torch 
from ultralytics import YOLO
import matplotlib.pyplot as plt 
import cv2 as cv

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")

model = YOLO('yolov8n.pt')
model.to(device)
cars_counted = []
counter = 0
def rescaleFrame(frame, scale =1):
    width = 1280
    height = 720
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
capture = cv.VideoCapture("Videos/Traffic.mp4")

counter = 0
line_y = 200

output_width = 1280
output_height = 720
fps = int(capture.get(cv.CAP_PROP_FPS))
out = cv.VideoWriter("Videos/traffic_demo.mp4", cv.VideoWriter_fourcc(*'mp4v'), fps, (output_width, output_height))



while True: 
    isTrue, frame = capture.read()
    if not isTrue:
        break
    frame_resized = rescaleFrame(frame, scale=0.75)
    outputs = model.track(frame_resized, persist=True)
    cv.line(frame_resized, (0,line_y), (1280,line_y), color = (0, 0 , 255), thickness = 3)
    
    for o in outputs: 
        if o.boxes.id is not None:
            for box in o.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                id = int(box.id[0])
                confidence = float(box.conf[0])
                
                if class_name == 'car' and confidence > 0.2:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 =  int(x1), int (y1), int(x2), int (y2)
                    cv.rectangle(frame_resized, (x1, y1), (x2,y2), color = (205,255,0), thickness = 2)
                    box_center = int((y1 + y2) / 2)
                    
                    
                    if line_y - 10 < box_center < line_y + 10 and id not in cars_counted:
                        counter += 1
                        cars_counted.append(id)
    
    
    cv.putText(frame_resized, f"Counter: {counter}", (20, 30), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(205,255,0), thickness=2)
    out.write(frame_resized)
    cv.imshow('Cars', frame_resized)
    if cv.waitKey(1) & 0xFF==ord('d'):
        break



capture.release()
out.release()
cv.destroyAllWindows()




