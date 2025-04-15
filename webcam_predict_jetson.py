import cv2
import numpy as np
import supervision as sv
from rfdetr import RFDETRBase
import os

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


def run_detection():
    # Load dataset using the COCO annotations.
    ds = sv.DetectionDataset.from_coco(
        images_directory_path="./dataset/valid",
        annotations_path="./dataset/valid/_annotations.coco.json",
    )

    # Load model
    model = RFDETRBase(pretrain_weights="./logs/checkpoint_best_total.pth")

    # Autofocus setup
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Unable to open camera.")
        return
    
    # Autofocus routine
    max_index = 10
    max_value = 0.0
    last_value = 0.0
    dec_count = 0
    focal_distance = 10
    focus_finished = False

    cv2.namedWindow('CSI Camera', cv2.WINDOW_AUTOSIZE)

    while cv2.getWindowProperty('CSI Camera', 0) >= 0:
        ret, img = cap.read()
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

        resolution_wh = (img.shape[1], img.shape[0])
        detections = model.predict(img, threshold=0.5)
        
        # Calculate optimal text scale and line thickness based on the image resolution.
        text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)
        thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)

        # Create annotators for bounding boxes and labels.
        bbox_annotator = sv.BoxAnnotator(thickness=thickness)
        label_annotator = sv.LabelAnnotator(
            text_color=sv.Color.BLACK,
            text_scale=text_scale,
            text_thickness=thickness,
            smart_position=True
        )

        # Generate detection labels.
        labels = [
            f"{ds.classes[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        # Annotate the image with bounding boxes and labels.
        annotated = bbox_annotator.annotate(img.copy(), detections)
        annotated = label_annotator.annotate(annotated, detections, labels)

        cv2.imshow("CSI Camera", annotated)

        keyCode = cv2.waitKey(16) & 0xff
        # Stop the program on the ESC key
        if keyCode == 27:
            break
        elif keyCode == 10:
            max_index = 10
            max_value = 0.0
            last_value = 0.0
            dec_count = 0
            focal_distance = 10
            focus_finished = False

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_detection()