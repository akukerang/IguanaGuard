import RPi.GPIO as GPIO
import time

LASER_PIN = 18  # Choose any GPIO pin

GPIO.setmode(GPIO.BCM)
GPIO.setup(LASER_PIN, GPIO.OUT)

try:
    while True:
        GPIO.output(LASER_PIN, GPIO.HIGH)  # Laser ON
        time.sleep(1)
        GPIO.output(LASER_PIN, GPIO.LOW)   # Laser OFF
        time.sleep(1)

except KeyboardInterrupt:
    GPIO.cleanup()  # Cleanup GPIO on exit