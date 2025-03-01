import RPi.GPIO as GPIO
import time


RELAY_PIN = 2
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT)

def toggle_relay(state):
    """Toggle the relay on or off based on the state."""
    GPIO.output(RELAY_PIN, GPIO.LOW if state else GPIO.HIGH)  # Invert logic
    print(f"Relay {'ON' if state else 'OFF'}")

try:
    while True:
        command = input("Enter 'on' to activate, 'off' to deactivate, or 'exit' to quit: ").strip().lower()
        
        if command == "on":
            toggle_relay(True)   # Turns ON (LOW for active-low relay)
        elif command == "off":
            toggle_relay(False)  # Turns OFF (HIGH for active-low relay)
        elif command == "exit":
            break
        else:
            print("Invalid input. Type 'on', 'off', or 'exit'.")

except KeyboardInterrupt:
    print("\nExiting program...")

finally:
    GPIO.cleanup()
    print("GPIO cleaned up.")