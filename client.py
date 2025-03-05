import cv2
from multiprocessing import Process
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
    InputVStreamParams, OutputVStreamParams, InputVStreams, OutputVStreams, FormatType) 
import numpy as np
import pickle
import RPi.GPIO as GPIO
import time
import threading


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

# === GPIO Section ===
# Initial servo angles
x_angle = 90
y_angle = 90
servo_lock = threading.Lock()  # Lock to ensure only one servo update happens at a time

# ===GPIO SECTION===
GPIO.setmode(GPIO.BCM)

# Pin NUmbers
X_axis_servo = 14
Y_axis_servo = 15
# SoundPin = 21
GPIO.setup(X_axis_servo, GPIO.OUT)
GPIO.setup(Y_axis_servo, GPIO.OUT)
# GPIO.setup(SoundPin, GPIO.OUT)

# Set up PWM for servos
X_servo = GPIO.PWM(X_axis_servo, 50)  # 50 Hz
Y_servo = GPIO.PWM(Y_axis_servo, 50)  # 50 Hz

X_servo.start(0)
Y_servo.start(0)

# Servo Functions

def set_servo_angle(servo, angle):
    duty_cycle = (angle / 18.0) + 2.5
    servo.ChangeDutyCycle(duty_cycle)
    time.sleep(0.3)  
    servo.ChangeDutyCycle(0)

def coords_to_angles(x, y): 
    # Normalize Coordinates
    norm_x = (x / 609) * 2 - 1
    norm_y = (y / 427) * 2 - 1

    horizontal_fov = 54.42
    vertical_fov = 42.12

    # Convert to angle
    x_angle = -norm_x * (horizontal_fov / 2)  
    y_angle = -norm_y * (vertical_fov / 2)  

    # Offset Angle
    # x_angle = int(125+x_angle)  # with ultrasound
    x_angle = int(95+x_angle)
    y_angle = int(105+y_angle)  

    x_angle = max(0, min(180, x_angle))
    y_angle = max(0, min(180, y_angle))
    return x_angle, y_angle

def move_servos(x, y):
    with servo_lock:
        angle_x, angle_y = coords_to_angles(x, y)
        set_servo_angle(X_servo, angle_x)
        set_servo_angle(Y_servo, angle_y)


previous_x, previous_y = None, None
dim = 640
last_move_time = time.time()
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
                    if previous_x is None or abs(center_x - previous_x) > 25 or abs(center_y - previous_y) > 25: # Checks for enough movement
                        move_servos(center_x, center_y)
                        # threading.Thread(target=move_servos_thread, args=(center_x, center_y)).start()
                        previous_x, previous_y = center_x, center_y

    cv2.imshow("IguanaGuard", undistorted_frame) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


camera.release()
