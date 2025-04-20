import cv2
import time
from flask import Flask, Response, render_template, request
import os
import psutil
from modules.escalation import EscalationManager
from modules.servos import ServoController
from modules.detection import Detection_Model

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

DEBOUNCE_FRAMES = 80 # around half a second
state = "idle"
detected_time = "0"
temperature = get_cpu_temperature()
cpu_usage = get_cpu_usage()

CONFIDENCE_THRESHOLD = 0.5
model  = Detection_Model(confidence=CONFIDENCE_THRESHOLD)

def generate_frames():    
    global state, detected_time, temperature, cpu_usage, model
    camera = cv2.VideoCapture(0)
    previous_x, previous_y = None, None
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
        model.run(frame)

        state = escalation_manager.get_status()
        detected_time = escalation_manager.get_elapsed_time()
        temperature = get_cpu_temperature()
        cpu_usage = get_cpu_usage()

        if model.detected:
            if not movement_detected:  # Only trigger once per detection sequence
                escalation_manager.detect_movement()
                movement_detected = True  # Mark as triggered
            
            detection_count = min(detection_count + 1, DEBOUNCE_FRAMES)
            no_detection_count = 0

            if previous_x is None or abs(model.center_x - previous_x) > 25 or abs(model.center_y - previous_y) > 25:
                # servo.move_servos(center_x, center_y)
                servo_controller.move_servos(model.center_x, model.center_y)
                previous_x, previous_y = model.center_x, model.center_y

        else:
            no_detection_count += 1
            detection_count = max(detection_count - 1, 0)

        # Reset deterrents if no detection for long enough
        if no_detection_count >= DEBOUNCE_FRAMES:
            if movement_detected:  # Only reset if previously triggered
                escalation_manager.reset()
                movement_detected = False  # Reset flag
            no_detection_count = DEBOUNCE_FRAMES  # Cap counter

        ret, buffer = cv2.imencode('.jpg', model.undistorted_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()
    escalation_manager.stop()
    servo_controller.cleanup()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status')
def status():
    return {
        'state': state,
        'detected_time': detected_time,
        'temperature': temperature,
        'cpu_usage': cpu_usage
    }

@app.route('/set_confidence', methods=['POST'])
def set_confidence():
    global CONFIDENCE_THRESHOLD, model
    try:
        new_confidence = float(request.form['confidence'])
        CONFIDENCE_THRESHOLD = new_confidence
        model.update_confidence(CONFIDENCE_THRESHOLD)
        return {'status': 'success', 'confidence': CONFIDENCE_THRESHOLD}
    except ValueError:
        return {'status': 'error', 'message': 'Invalid confidence value'}, 400

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=False) 
    except Exception as e:
        print(f"Error occurred: {e}")