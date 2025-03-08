import cv2
from multiprocessing import Process
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
    InputVStreamParams, OutputVStreamParams, InputVStreams, OutputVStreams, FormatType) 
import numpy as np
import pickle
import RPi.GPIO as GPIO
import time
from flask import Flask, Response, render_template, request
import os
import psutil


# ===CAMERA SECTION===
# Calibration Parameters
cameraMatrix, dist = pickle.load(open("resources/calibration.pkl", "rb"))
height = 480
width = 640

# Camera Matrix
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (width, height), 1, (width, height))
x, y, w, h = roi 

mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (width, height), 5)

# Camera Functions
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
CONFIDENCE_THRESHOLD = 0.50
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

GPIO.setmode(GPIO.BCM)

# Pin Numbers
X_AXIS_SERVO = 14
Y_AXIS_SERVO = 15
LASER_PIN = 18
# SOUND_PIN = 23
# RELAY_PIN = 4

GPIO.setup(X_AXIS_SERVO, GPIO.OUT)
GPIO.setup(Y_AXIS_SERVO, GPIO.OUT)
GPIO.setup(LASER_PIN, GPIO.OUT)
# GPIO.setup(SOUND_PIN, GPIO.OUT)
# GPIO.setup(RELAY_PIN, GPIO.OUT)

# Set up PWM for servos
X_servo = GPIO.PWM(X_AXIS_SERVO, 50)  # 50 Hz
Y_servo = GPIO.PWM(Y_AXIS_SERVO, 50)  # 50 Hz

X_servo.start(0)
Y_servo.start(0)
GPIO.output(LASER_PIN, GPIO.LOW)  # Laser init off 
# GPIO.output(SOUND_PIN, GPIO.LOW)  # Sound init off 
# GPIO.output(RELAY_PIN, GPIO.LOW)  # Liquid init off 



# Servo Functions
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

def set_servo_angle(servo, angle):
    duty_cycle = (angle / 18.0) + 2.5
    servo.ChangeDutyCycle(duty_cycle)
    time.sleep(0.3)  
    servo.ChangeDutyCycle(0)

def move_servos(x, y):
    angle_x, angle_y = coords_to_angles(x, y)
    set_servo_angle(X_servo, angle_x)
    set_servo_angle(Y_servo, angle_y)

# ===Flask Section===
app = Flask(__name__)

## Flask Functions
def get_cpu_usage(): # Get CPU usage 
    cpu_usage = psutil.cpu_percent()
    return cpu_usage

def get_cpu_temperature(): # Gets CPU Temp
    try:
        temp = None
        if os.path.exists("/sys/class/thermal/thermal_zone0/temp"):
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                temp = float(f.read()) / 1000 
        return temp
    except Exception as e:
        return None

DEBOUNCE_FRAMES = 25 # around half a second
def generate_frames():
    camera = cv2.VideoCapture(0)
    previous_x, previous_y = None, None
    dim = 640
    last_move_time = time.time()
    detection_count = 0 # Frame counters for debounce 
    no_detection_count = 0
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
                infer_results = infer_pipeline.infer(preprocessed)  
                oh, ow, _ = undistorted_frame.shape
                rh, rw = oh / dim, ow / dim
                detections = np.array(infer_results[list(infer_results.keys())[0]][0])
                detected = False
                if detections[0].shape[0] > 0: # Iguana detected in frame
                    flattened_detections = detections[0].reshape(-1, 5)
                    sorted_detections = flattened_detections[flattened_detections[:, 4].argsort()[::-1]] # Gets box with highest confidence
                    highest_conf = sorted_detections[0]
                    if highest_conf[4] > CONFIDENCE_THRESHOLD:
                        detected = True 
                        bbox = highest_conf[0:4] * dim
                        center_x, center_y = draw_detection(undistorted_frame, bbox, highest_conf[4], (0,255,0), rw, rh) # Draws bounding boxes

        # Debounce 
        if detected:
            detection_count += 1
            no_detection_count = 0  # Reset no-detection count
        else:
            no_detection_count += 1
            detection_count = 0  # Reset detection count

        # Turn deterrant on if enough detected frames
        if detection_count >= DEBOUNCE_FRAMES:
            
            if previous_x is None or abs(center_x - previous_x) > 25 or abs(center_y - previous_y) > 25: # Move if previous X and y, vary enough to current
                move_servos(center_x, center_y)
                previous_x, previous_y = center_x, center_y
            GPIO.output(LASER_PIN, GPIO.HIGH) 
            detection_count = DEBOUNCE_FRAMES  

            # TODO: Escalation Model Here





        # Turn Deterrants off if enough non-detected frames
        if no_detection_count >= DEBOUNCE_FRAMES:
            GPIO.output(LASER_PIN, GPIO.LOW)
            # GPIO.output(SOUND_PIN, GPIO.LOW)  
            # GPIO.output(RELAY_PIN, GPIO.LOW)  
            no_detection_count = DEBOUNCE_FRAMES  

        ret, buffer = cv2.imencode('.jpg', undistorted_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    camera.release()


@app.route('/')
def index():
    return render_template('index.html', temperature=get_cpu_temperature(), cpu_usage=get_cpu_usage())


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def cleanup(): # Cleans up after app ends
    GPIO.cleanup()

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=False) 
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        cleanup()
