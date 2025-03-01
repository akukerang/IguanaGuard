class Deterrence:
  def __init__(self):
        self.laser_active = False
        self.buzzer_active = False
        self.spray_active = False
  
  def activate_laser(self):
      self.laser_active = True
      print("Laser activated.")

  def deactivate_laser(self):
      self.laser_active = False
      print("Laser deactivated.")

  def activate_buzzer(self):
      self.buzzer_active = True
      print("Buzzer activated.")

  def deactivate_buzzer(self):
      self.buzzer_active = False
      print("Buzzer deactivated.")

  def activate_spray(self):
      self.spray_active = True
      print("Spray activated.")

  def deactivate_spray(self):
      self.spray_active = False
      print("Spray deactivated.")