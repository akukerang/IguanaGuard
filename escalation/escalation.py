import time
import threading
from enum import Enum, auto
from deterrence import Deterrence

class State(Enum):
    IDLE = auto()
    LASER = auto()
    BUZZER = auto() # Might switch pos with laser
    SPRAY = auto()
    MAX_ESCALATION = auto()

class EscalationManager:
    def __init__(self):
        self.deterrence = Deterrence()
        self.current_state = State.IDLE
        self.last_detection_time = None
        self.running = False
        self.time_thread = threading.Thread(target=self.update_time, daemon=True)
        self.time_lock = threading.Lock()
        self.elapsed_time = 0

    def update_time(self):
        """Function for background thread that tracks elapsed time since last detection."""
        while self.running:
            with self.time_lock:
                if self.last_detection_time:
                    self.elapsed_time = time.time() - self.last_detection_time
            time.sleep(1)  # Update time every second

    def start(self):
        """Start thread to run time and start escalation"""
        if not self.running:
            self.running = True
            self.time_thread.start()
    
    def stop(self):
        """Join time and main thread and stop system"""
        self.running = False
        self.time_thread.join()

    def detect_movement(self):
        """Called when movement is detected."""
        pass
            
    def stop_monitoring(self):
        """Stops the escalation loop."""
        pass

    def _monitor_escalation(self):
        """Monitors movement and escalates deterrence over time."""
        pass

    def reset(self):
        """Resets deterrence if no movement is detected for a while."""
        pass
