

import cv2
from ultralytics import YOLO

#vfrom gpiozero import PWMLED
from time import sleep

# Load YOLO model
model = YOLO("runs/detect/train3/weights/last.pt")

# Initialize video capture (camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)

# Variable to store the tracker
tracker = None
init_once = False

frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
margin_x = int(frame_w * 0.1)
margin_y = int(frame_h * 0.1)
            
c_left = frame_w // 2 - margin_x
c_right = frame_w // 2 + margin_x
c_top = frame_h // 2 - margin_y
c_bottom = frame_w // 2 + margin_y

while True: 
    ret, frame = cap.read()
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    if not init_once:
        
        #results = model(frame)[0]
        detect_params = model.predict(source=[frame], conf=0.45, save=False)

        DP = detect_params[0].cpu().numpy()
        print(DP)
        
        if len(DP) > 0:
            
            #box = results.boxes[0].xyxy[0].cpu().numpy().astype(int)
            #x1, y1, x2, y2 = box
            #w, h = x2 - x1, y2 - y1
            
            i = 0
            boxes = detect_params[0].boxes
            box = boxes[i]
            clsID = box.cls.cpu().numpy().astype(int)[0]
            conf = box.conf.cpu().numpy().astype(int)[0]
            bb = box.xyxy.cpu().numpy().astype(int)[0]  # Format [xmin, ymin, xmax, ymax]

            # Convert [xmin, ymin, xmax, ymax] to [x, y, width, height]
            x1, y1, w, h = int(bb[0]), int(bb[1]), int(bb[2] - bb[0]), int(bb[3] - bb[1])
            
            
            #tracker = cv2.legacy.TrackerMOSSE_create()
            tracker = cv2.TrackerKCF_create()

            tracker.init(frame, (x1, y1, w, h))
            init_once = True
    
    else:
        
        success, box = tracker.update(frame)
        
        if success:
            x, y, w, h = [int(v) for v in box]
            cx, cy = x + w // 2, y + h // 2 #box center
            
                       
            cv2.rectangle(frame, (c_left, c_top), (c_right, c_bottom), (255, 0, 0), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            if c_left < cx < c_right and c_top < cy < c_bottom:
                cv2.putText(frame, "In Center", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                #led = 0
                #led = 0
            else:
                cv2.putText(frame, "Tracking", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                #led = 1
                #led = 1
    
            
        else:
            cv2.putText(frame, "Lost", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            #led = 0
            #led = 0
            
    cv2.imshow("Tracking", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()

        
     
