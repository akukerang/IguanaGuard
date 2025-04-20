import time
import threading
from enum import Enum, auto
from modules.deterrence import Deterrence

class State(Enum):
    IDLE = auto()
    LASER = auto()
    BUZZER = auto()
    SPRAY = auto()
    MAX_ESCALATION = auto()

class EscalationManager:
    def __init__(self, laser_time=2, buzzer_time=60, spray_time=300, reset_time=600):
        self.deterrence = Deterrence()
        self.current_state = State.IDLE
        self.last_detection_time = None
        self.running = False
        self.time_thread = threading.Thread(target=self.update_time, daemon=True)
        self.time_lock = threading.Lock()
        self.elapsed_time = 0
        self.monitor_thread = threading.Thread(target=self._monitor_escalation, daemon=True)
        self.laser_time = laser_time
        self.buzzer_time = buzzer_time
        self.spray_time = spray_time
        self.reset_time = reset_time

    def update_laser_time(self, laser_time):
        """Update the laser time threshold."""
        self.laser_time = laser_time

    def update_buzzer_time(self, buzzer_time):  
        """Update the buzzer time threshold."""
        self.buzzer_time = buzzer_time

    def update_spray_time(self, spray_time):
        """Update the spray time threshold."""
        self.spray_time = spray_time

    def update_reset_time(self, reset_time):
        """Update the reset time threshold."""
        self.reset_time = reset_time


    def update_time(self):
        """Background thread to track elapsed time since last detection."""
        while self.running:
            with self.time_lock:
                if self.last_detection_time:
                    self.elapsed_time = time.time() - self.last_detection_time
                else:
                    self.elapsed_time = 0  # Reset time if no detection
            time.sleep(1)  # Update every second

    def start(self):
        """Start monitoring and time tracking."""
        if not self.running:
            self.running = True
            self.time_thread.start()
            self.monitor_thread.start()

    def stop(self):
        """Stop all monitoring and reset system."""
        self.running = False
        self.current_state = State.IDLE
        self.time_thread.join()
        self.monitor_thread.join()
        self.reset()

    def detect_movement(self):
        """Called when movement is detected."""
        with self.time_lock:
            self.last_detection_time = time.time()
        print("Movement detected, resetting timer.")

    def stop_monitoring(self): 
        """Stops escalation monitoring."""
        self.running = False

    def _monitor_escalation(self):
        """Monitors movement and escalates deterrence over time."""
        while self.running:            
            with self.time_lock:
                elapsed = self.elapsed_time            
            if elapsed >= self.reset_time: # reset 
                self.reset()
            elif elapsed >= self.spray_time:
                if self.current_state != State.SPRAY:
                    self.current_state = State.SPRAY
                    self.deterrence.activate_spray()
                    print("Escalation: SPRAY ON")

            elif elapsed >= self.buzzer_time:
                if self.current_state != State.BUZZER:
                    self.current_state = State.BUZZER
                    self.deterrence.activate_buzzer()
                    print("Escalation: BUZZER ON")

            elif elapsed >= self.laser_time:
                if self.current_state != State.LASER:
                    self.current_state = State.LASER
                    self.deterrence.activate_laser()
                    print("Escalation: LASER ON")
            else:
                if self.current_state != State.IDLE:
                    self.current_state = State.IDLE
                    self.deterrence.deactivate_all()
                    print("Escalation: All deterrents OFF")

           

            time.sleep(1)  # Check escalation every second

    def reset(self):
        """Resets deterrence system if no movement is detected for a while."""
        self.current_state = State.IDLE
        self.last_detection_time = None
        self.deterrence.deactivate_all()
        print("System reset: All deterrents OFF.")

    def get_status(self):
        """Returns the current status of the escalation system."""
        return self.current_state.name

    def get_elapsed_time(self):
        """Returns the elapsed time since the last detection."""
        with self.time_lock:
            return self.elapsed_time