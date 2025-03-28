from flask import Flask, Response, render_template, request
import cv2
import time
import RPi.GPIO as GPIO
from ultralytics import YOLO
import atexit
import pickle
import numpy as np
import psutil



app = Flask(__name__)

# === Global Vars ===
CONFIDENCE_THRESHOLD = 0.4
FRAME_RATE = 60

# Set up GPIO
GPIO.setmode(GPIO.BCM)
X_axis_servo = 15
Y_axis_servo = 23
GPIO.setup(X_axis_servo, GPIO.OUT)
GPIO.setup(Y_axis_servo, GPIO.OUT)

# Set up PWM for servos
X_servo = GPIO.PWM(X_axis_servo, 50)  # 50 Hz
Y_servo = GPIO.PWM(Y_axis_servo, 50)  # 50 Hz

X_servo.start(0)
Y_servo.start(0)



# Calibration Parameters
cameraMatrix, dist = pickle.load(open("calibration.pkl", "rb"))

# Camera Start
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FPS, FRAME_RATE)

# Frame Info
ret, frame = camera.read()
if not ret:
    raise RuntimeError("Failed to read from camera")

height, width = frame.shape[:2]

# Camera Matrix
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (width, height), 1, (width, height))
x, y, w, h = roi  # Unpack cropping values

# Model
model = YOLO('best.onnx')

mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (width, height), 5)

def undistort_frame(frame):
    return cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

import psutil

def get_cpu_usage():
    # Get CPU usage percentage
    cpu_usage = psutil.cpu_percent(interval=1)
    return cpu_usage

import os

def get_cpu_temperature():
    try:
        # Read temperature from the system's file (Raspberry Pi or Linux-based systems)
        temp = None
        if os.path.exists("/sys/class/thermal/thermal_zone0/temp"):
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                temp = float(f.read()) / 1000  # Convert from millidegree Celsius to Celsius
        return temp
    except Exception as e:
        return None




def generate_frames():
    while True:
        success, frame = camera.read()

        if not success:
            break

        # Undistort the frame
        # undistorted_frame = cv2.undistort(frame, cameraMatrix, dist, None, newCameraMatrix)
        undistorted_frame = undistort_frame(frame)

        # Crop (optional, if valid ROI)
        if roi != (0, 0, 0, 0):  
            undistorted_frame = undistorted_frame[y:y+h, x:x+w]

        # Prediction
        results = model(undistorted_frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

        # Get largest confidence box
        max_conf = -float('inf')  # Initialize with a very low value
        main_box = None

        # Flatten the results and only iterate over boxes that meet the threshold
        for r in results:
            for box in r.boxes:
                confidence = float(box.conf[0])
                if confidence > max_conf and confidence >= CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    max_conf = confidence
                    main_box = (x1, y1, x2, y2)



        if main_box:
            x1, y1, x2, y2 = main_box
            target_x = (x1 + x2) / 2
            target_y = (y1 + y2) / 2
            label = f"{confidence:.2f}"
            cv2.rectangle(undistorted_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(undistorted_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Encode in JPEG format
        # ret, buffer = cv2.imencode('.jpg', undistorted_frame)
        ret, buffer = cv2.imencode('.jpg', undistorted_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        frame = buffer.tobytes()

        # Yield the frame in byte format
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# === Flask Routes ===
@app.route('/')
def index():
    return render_template('index.html', temperature=get_cpu_temperature(), cpu_usage=get_cpu_usage())

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_confidence', methods=['POST'])
def set_confidence():
    global CONFIDENCE_THRESHOLD
    CONFIDENCE_THRESHOLD = float(request.form['confidence'])
    return '', 204  # Success

# === Cleanup GPIO on Exit ===
def cleanup():
    GPIO.cleanup()

atexit.register(cleanup)

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)  # Run without debug mode
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        cleanup()

