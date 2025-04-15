import cv2
from ultralytics import YOLO
import os
import sys

MODEL_NAME = 'runs/detect/train/weights/best.pt'

def focusing(val):
    value = (val << 4) & 0x3ff0
    data1 = (value >> 8) & 0x3f
    data2 = value & 0xf0
    os.system("i2cset -y 6 0x0c %d %d" % (data1,data2))

def laplacian(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.mean(cv2.Laplacian(gray, cv2.CV_16U))[0]


# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 60fps
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen
def gstreamer_pipeline(capture_width=1280, capture_height=720, display_width=1280,
                       display_height=720, imgrate=60, flip_method=0):
    return (
        'nvarguscamerasrc ! '
        'video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, '
        'format=(string)NV12, imgrate=(fraction)%d/1 ! '
        'nvvidconv flip-method=%d ! '
        'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
        'videoconvert ! '
        'video/x-raw, format=(string)BGR ! appsink'
        % (capture_width, capture_height, imgrate,
           flip_method, display_width, display_height)
    )

model = YOLO(MODEL_NAME)
print(model.names)
webcamera = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
if not webcamera.isOpened():
    print("Cant open camera")
    sys.exit()

# Autofocus routine
max_index = 10
max_value = 0.0
last_value = 0.0
dec_count = 0
focal_distance = 10
focus_finished = False

cv2.namedWindow('CSI Camera', cv2.WINDOW_AUTOSIZE)

while cv2.getWindowProperty('CSI Camera', 0) >= 0:
    print("Reading from camera...")
    ret, img = webcamera.read()
    if not ret:
        break
    
    if dec_count < 6 and focal_distance < 1000:
        #Adjust focus
        focusing(focal_distance)
        #Take image and calculate image clarity
        val = laplacian(img)
        #Find the maximum image clarity
        if val > max_value:
            max_index = focal_distance
            max_value = val
            
        #If the image clarity starts to decrease
        if val < last_value:
            dec_count += 1
        else:
            dec_count = 0
        #Image clarity is reduced by six consecutive frames
        if dec_count < 6:
            last_value = val
            #Increase the focal distance
            focal_distance += 10
    elif not focus_finished:
        #Adjust focus to the best
        focusing(max_index)
        focus_finished = True

    results = model.track(img, classes=0, conf=0.8, imgsz=480)
    cv2.putText(img, f"Total: {len(results[0].boxes)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Live Camera", results[0].plot())

    if cv2.waitKey(1) == ord('q'):
        break

webcamera.release()
cv2.destroyAllWindows()
