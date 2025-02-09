import RPi.GPIO as GPIO
import time
import threading


# Set up GPIO
GPIO.setmode(GPIO.BCM)
X_axis_servo = 23
Y_axis_servo = 15
GPIO.setup(X_axis_servo, GPIO.OUT)
GPIO.setup(Y_axis_servo, GPIO.OUT)

# Set up PWM for servos
X_servo = GPIO.PWM(X_axis_servo, 50)  # 50 Hz
Y_servo = GPIO.PWM(Y_axis_servo, 50)  # 50 Hz

X_servo.start(0)
Y_servo.start(0)

def set_servo_angle(servo, angle):

    duty_cycle = (angle / 18.0) + 2.5  # Convert angle to duty cycle
    servo.ChangeDutyCycle(duty_cycle)
    time.sleep(0.3)  # Allow time to move
    servo.ChangeDutyCycle(0)  # Stop sending signal

def control_servo_in_thread(servo, angle):
    """
    Wrapper function to control the servo in a new thread.
    """
    # Create and start a new thread for setting the servo angle
    thread = threading.Thread(target=set_servo_angle, args=(servo, angle))
    thread.start()


try:
    while True:
        x_angle = float(input("Enter X servo angle (0-180): "))
        y_angle = float(input("Enter Y servo angle (0-180): "))

        control_servo_in_thread(X_servo, x_angle) # MIn 90
        control_servo_in_thread(Y_servo, y_angle) # min 90
        time.sleep(1)

except KeyboardInterrupt:
    print("\nExiting...")
    X_servo.stop()
    Y_servo.stop()
    GPIO.cleanup()


def set_servo_angle(servo, angle):
    duty_cycle = (angle / 18.0) + 2.5  # Convert angle to duty cycle
    servo.ChangeDutyCycle(duty_cycle)
    time.sleep(1)  # Allow time to move
    servo.ChangeDutyCycle(0)  # Stop sending signal


# Top Left (135, 106)
# Top Right (88, 110)
# Bottom Left (154, 77)
# Bottom Right (128, 70)

# Center (116,88)