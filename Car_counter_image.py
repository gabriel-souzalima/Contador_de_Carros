import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2 as cv
model = YOLO('yolov8n.pt')
outputs  = model("Images/two_cars.png")
image = cv.imread("Images/two_cars.png")
posX = 350
counter = 0
for o in outputs: 
    for box in o.boxes:
        counter += 1
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 =  int(x1), int (y1), int(x2), int (y2)
        
      
        if x1 < posX:
            posX = x1
        print (posX, x1)

        print (f"the coordinates are : \n x1 = {x1} \n y1 = {y1} \n x2 = {x2} \n y2 = {y2}") 
        cv.rectangle(image, (x1, y1), (x2,y2), color = (205,255,0), thickness = 5)

cv.putText(image, f"Counter: {counter}", (posX, y1 - 50), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(205,255,0), thickness=2)

cv.imshow("Porsche 718 Cayman GTS 4.0", image)
cv.waitKey(0)

