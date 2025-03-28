import RPi.GPIO as GPIO

class Deterrence:
    def __init__(self):
        self.laser_active = False
        self.buzzer_active = False
        self.spray_active = False
        self.LASER_PIN = 18
        self.SOUND_PIN = 23
        self.RELAY_PIN = 4
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.LASER_PIN, GPIO.OUT)
        GPIO.setup(self.SOUND_PIN, GPIO.OUT)
        GPIO.setup(self.RELAY_PIN, GPIO.OUT)
        GPIO.output(self.LASER_PIN, GPIO.LOW)  # Laser init off 
        GPIO.output(self.SOUND_PIN, GPIO.LOW)  # Sound init off 
        GPIO.output(self.RELAY_PIN, GPIO.LOW)  # Liquid init off 

    def activate_laser(self):
        self.laser_active = True
        GPIO.output(self.LASER_PIN, GPIO.HIGH) 

    def deactivate_laser(self):
        self.laser_active = False
        GPIO.output(self.LASER_PIN, GPIO.LOW) 

    def activate_buzzer(self):
        self.buzzer_active = True
        GPIO.output(self.SOUND_PIN, GPIO.HIGH) 

    def deactivate_buzzer(self):
        self.buzzer_active = False
        GPIO.output(self.SOUND_PIN, GPIO.LOW) 

    def activate_spray(self):
        self.spray_active = True
        GPIO.output(self.RELAY_PIN, GPIO.HIGH)

    def deactivate_spray(self):
        self.spray_active = False
        GPIO.output(self.RELAY_PIN, GPIO.LOW)

    def deactivate_all(self):
        self.deactivate_laser()
        self.deactivate_buzzer()
        self.deactivate_spray()