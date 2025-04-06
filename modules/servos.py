import time
import RPi.GPIO as GPIO
import threading

class ServoController:
    def __init__(self):
        # Initial servo angles
        self.x_angle = 90
        self.y_angle = 90

        GPIO.setmode(GPIO.BCM)

        # Pin Numbers
        self.X_AXIS_SERVO = 14
        self.Y_AXIS_SERVO = 15

        GPIO.setup(self.X_AXIS_SERVO, GPIO.OUT)
        GPIO.setup(self.Y_AXIS_SERVO, GPIO.OUT)

        # Set up PWM for servos
        self.X_servo = GPIO.PWM(self.X_AXIS_SERVO, 50)  # 50 Hz
        self.Y_servo = GPIO.PWM(self.Y_AXIS_SERVO, 50)  # 50 Hz

        self.X_servo.start(0)
        self.Y_servo.start(0)

        self.servo_lock = threading.Lock()
        self.movement_thread = None

    def coords_to_angles(self, x, y): 
        # Normalize Coordinates
        norm_x = (x / 609) * 2 - 1
        norm_y = (y / 427) * 2 - 1

        horizontal_fov = 54.42
        vertical_fov = 42.12

        # Convert to angle
        x_angle = -norm_x * (horizontal_fov / 2)  
        y_angle = -norm_y * (vertical_fov / 2)  

        # Offset Angle
        x_angle = int(100+x_angle)
        y_angle = int(80+y_angle)  

        x_angle = max(0, min(180, x_angle))
        y_angle = max(0, min(180, y_angle))
        return x_angle, y_angle

    def set_servo_angle(self, servo, angle):
        duty_cycle = (angle / 18.0) + 2.5
        servo.ChangeDutyCycle(duty_cycle)
        time.sleep(0.3)  
        servo.ChangeDutyCycle(0)

    def move_servos(self, x, y):
        with self.servo_lock:
            angle_x, angle_y = self.coords_to_angles(x, y)
            self.set_servo_angle(self.X_servo, angle_x)
            self.set_servo_angle(self.Y_servo, angle_y)

    def cleanup(self):
        self.X_servo.stop()
        self.Y_servo.stop()
        GPIO.cleanup()

