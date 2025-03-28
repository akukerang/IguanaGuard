import cv2
from multiprocessing import Process
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
    InputVStreamParams, OutputVStreamParams, InputVStreams, OutputVStreams, FormatType) 
import numpy as np
import pickle

# ===CAMERA SECTION===
FRAME_RATE = 10

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FPS, FRAME_RATE)

# Calibration Parameters
cameraMatrix, dist = pickle.load(open("resources/calibration.pkl", "rb"))

# Frame
ret, frame = camera.read()
if not ret:
    raise RuntimeError("Failed to read from camera")
height, width = frame.shape[:2]

# Camera Matrix
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (width, height), 1, (width, height))
x, y, w, h = roi 

mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (width, height), 5)

def undistort_frame(frame): # Undistorts
    return cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

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

#=== Hailo Setup ===
target = VDevice()
CONFIDENCE_THRESHOLD = 0.65
# Loading compiled HEFs to device:

hef_path = 'models/unique_yolov8n.hef'
hef = HEF(hef_path)

# Configure network groups
configure_params = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
network_groups = target.configure(hef, configure_params)
network_group = network_groups[0]
network_group_params = network_group.create_params()

# Create input and output virtual streams params
input_vstreams_params = InputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
output_vstreams_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

# Define dataset params
input_vstream_info = hef.get_input_vstream_infos()[0]
output_vstream_info = hef.get_output_vstream_infos()[0]
image_height, image_width, channels = input_vstream_info.shape

dim = 640
while camera.isOpened():
    ret, frame = camera.read()
    if not ret:
        break

    undistorted_frame = undistort_frame(frame) # Remove fisheye effect

    if roi != (0, 0, 0, 0):  
        undistorted_frame = undistorted_frame[y:y+h, x:x+w] # Crop
    
    preprocessed = preprocess(undistorted_frame)

    with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
        with network_group.activate(network_group_params):
            infer_results = infer_pipeline.infer(preprocessed) # Inference
            oh, ow, _ = undistorted_frame.shape
            rh, rw = oh / dim, ow / dim
            detections = np.array(infer_results[list(infer_results.keys())[0]][0])

            if detections[0].shape[0] > 0: # If there is more than 0 detections
                flattened_detections = detections[0].reshape(-1, 5)
                sorted_detections = flattened_detections[flattened_detections[:, 4].argsort()[::-1]] # Sort by confidnce
                highest_conf = sorted_detections[0] # Highest confidence detection
                if highest_conf[4] > CONFIDENCE_THRESHOLD:  # Confidence check
                    bbox = highest_conf[0:4] * dim
                    center_x, center_y = draw_detection(undistorted_frame, bbox, highest_conf[4], (0,255,0), rw, rh)

    cv2.imshow("IguanaGuard", undistorted_frame) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


camera.release()
