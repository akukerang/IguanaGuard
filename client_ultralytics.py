import cv2
from ultralytics import YOLO
import numpy as np
import pickle
import RPi.GPIO as GPIO
import time
import threading




# ===GPIO SECTION===

# Set up GPIO
GPIO.setmode(GPIO.BCM)
X_axis_servo = 23
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




# ===CAMERA SECTION===
FRAME_RATE = 10
CONFIDENCE_LEVEL = 0.25

model = YOLO("best.onnx")
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FPS, FRAME_RATE)

# Calibration Parameters
cameraMatrix, dist = pickle.load(open("calibration.pkl", "rb"))

# Frame
ret, frame = camera.read()
if not ret:
    raise RuntimeError("Failed to read from camera")
height, width = frame.shape[:2]

# Camera Matrix
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (width, height), 1, (width, height))
x, y, w, h = roi 

mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (width, height), 5)

def undistort_frame(frame):
    return cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

def set_servo_angle(servo, angle):
    duty_cycle = (angle / 18.0) + 2.5
    servo.ChangeDutyCycle(duty_cycle)
    time.sleep(0.2)  
    servo.ChangeDutyCycle(0)


# ===Servo Functions===
# 609x427 Frame size after calibration

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
    x_angle = int(118+x_angle)
    y_angle = int(84+y_angle)  

    x_angle = max(0, min(180, x_angle))
    y_angle = max(0, min(180, y_angle))
    return x_angle, y_angle

def move_servos(x, y):
    angle_x, angle_y = coords_to_angles(x, y)
    set_servo_angle(X_servo, angle_x)
    set_servo_angle(Y_servo, angle_y)


# Main camera loop
previous_x, previous_y = None, None

while camera.isOpened():
    ret, frame = camera.read()
    if not ret:
        break

    undistorted_frame = undistort_frame(frame) # Remove fisheye effect

    if roi != (0, 0, 0, 0):  
        undistorted_frame = undistorted_frame[y:y+h, x:x+w] # Crop

    results = model(undistorted_frame, conf=CONFIDENCE_LEVEL, verbose=False) # Model Prediction
    detections = results[0].boxes.data.cpu().numpy()

    if len(detections) > 0:
        # Sort detections by confidence (column index 4) in descending order
        detections = sorted(detections, key=lambda x: x[4], reverse=True)
        x1, y1, x2, y2, conf, cls = detections[0]  # Get highest confidence detection

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        label = f"{model.names[int(cls)]}: {conf:.2f}"
        
        cv2.rectangle(undistorted_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(undistorted_frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # GPIO.output(SoundPin, GPIO.HIGH)
        with angle_lock:  # Sync angle updates
            if previous_x is None or abs(center_x - previous_x) > 10 or abs(center_y - previous_y) > 10:  # Ignore small changes
                move_servos(center_x, center_y)
                previous_x, previous_y = center_x, center_y

    cv2.imshow("IguanaGuard", undistorted_frame) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # else: 
    #     GPIO.output(SoundPin, GPIO.LOW)


camera.release()
cv2.destroyAllWindows()
X_servo.stop()
Y_servo.stop()
GPIO.cleanup()


