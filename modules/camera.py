import cv2
import pickle
import numpy as np

# ===CAMERA SECTION===
# Calibration Parameters
cameraMatrix, dist = pickle.load(open("resources/calibration.pkl", "rb"))
height = 480
width = 640

# Camera Matrix
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (width, height), 1, (width, height))
x, y, w, h = roi 

mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (width, height), 5)

def undistort_frame(frame): # Undistorts
    undistorted_frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    if roi != (0, 0, 0, 0):  
        undistorted_frame = undistorted_frame[y:y+h, x:x+w] # Crop
    return undistorted_frame

def preprocess(frame, target_size = (640,640)): # Preprocessing for YOLO model
    frame = cv2.resize(frame, target_size)
    frame = frame.astype(np.float32)
    frame = np.expand_dims(frame, axis=0)
    return frame

def draw_detection(image, bbox, confidence, color, scale_x, scale_y):
    ymin, xmin, ymax, xmax = bbox
    xmin, xmax = int(xmin * scale_x), int(xmax * scale_x)
    ymin, ymax = int(ymin * scale_y), int(ymax * scale_y)
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)  # Draw bounding box

    label = f"Iguana: {confidence:.2f}"    
    # Background for text
    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.rectangle(image, (xmin, ymin - text_height - 5), (xmin + text_width, ymin), color, -1)

    # Confidence text
    cv2.putText(image, label, (xmin, ymin - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return center_x, center_y
