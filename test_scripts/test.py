import cv2
import pickle
from flask import Flask, render_template, Response

# Flask setup
app = Flask(__name__)



# Calibration Parameters
cameraMatrix, dist = pickle.load(open("calibration.pkl", "rb"))

# Camera Start
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FPS, 60)  # Set frame rate if needed

# Frame Info
ret, frame = camera.read()
if not ret:
    raise RuntimeError("Failed to read from camera")

height, width = frame.shape[:2]

# Camera Matrix
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (width, height), 1, (width, height))
x, y, w, h = roi  # Unpack cropping values

# Initialize undistortion map
mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (width, height), 5)

def undistort_frame(frame):
    return cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

# Generate frames for streaming
def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Undistort the frame
        undistorted_frame = undistort_frame(frame)

        # Crop the frame (optional, if valid ROI)
        if roi != (0, 0, 0, 0):  
            undistorted_frame = undistorted_frame[y:y+h, x:x+w]

        # Convert the frame to grayscale for simplicity (no predictions)
        gray_frame = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)

        # Capture the mouse coordinates when hovering over the frame
        ret, buffer = cv2.imencode('.jpg', undistorted_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        frame = buffer.tobytes()

        # Yield the frame in byte format for the Flask response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Main route to render HTML page
@app.route('/')
def index():
    return render_template('index2.html')

# Video route for streaming frames
@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)  # Run without debug mode



def laser():
    laser_pin = 18

    