import cv2
from multiprocessing import Process
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
    InputVStreamParams, OutputVStreamParams, InputVStreams, OutputVStreams, FormatType) 
import numpy as np
import RPi.GPIO as GPIO
import time
from flask import Flask, Response, render_template, request
import os
import psutil
import modules.camera as cam
from modules.escalation import EscalationManager
from modules.servos import ServoController

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

DEBOUNCE_FRAMES = 26 # around half a second
def generate_frames():
    camera = cv2.VideoCapture(0)
    previous_x, previous_y = None, None
    dim = 640
    last_move_time = time.time()
    detection_count = 0 # Frame counters for debounce 
    no_detection_count = 0
    escalation_manager = EscalationManager()
    servo_controller = ServoController()
    escalation_manager.start()
    movement_detected = False  # Detection State
    while camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            break
        undistorted_frame = cam.undistort_frame(frame) # Remove fisheye effect        
        preprocessed = cam.preprocess(undistorted_frame)

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
                        center_x, center_y = cam.draw_detection(undistorted_frame, bbox, highest_conf[4], (0,255,0), rw, rh) # Draws bounding boxes

        if detected:
            if not movement_detected:  # Only trigger once per detection sequence
                escalation_manager.detect_movement()
                movement_detected = True  # Mark as triggered
            
            detection_count = min(detection_count + 1, DEBOUNCE_FRAMES)
            no_detection_count = 0

            if previous_x is None or abs(center_x - previous_x) > 25 or abs(center_y - previous_y) > 25:
                # servo.move_servos(center_x, center_y)
                servo_controller.move_servos(center_x, center_y)
                previous_x, previous_y = center_x, center_y

        else:
            no_detection_count += 1
            detection_count = max(detection_count - 1, 0)

        # Reset deterrents if no detection for long enough
        if no_detection_count >= DEBOUNCE_FRAMES:
            if movement_detected:  # Only reset if previously triggered
                escalation_manager.reset()
                movement_detected = False  # Reset flag
            no_detection_count = DEBOUNCE_FRAMES  # Cap counter

        ret, buffer = cv2.imencode('.jpg', undistorted_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()
    escalation_manager.stop()
    servo_controller.cleanup()

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
