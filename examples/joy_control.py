import sys
import os
import json
import numpy as np
import pygame
import matplotlib.pyplot as plt

# Pygame initialization for joystick handling
pygame.init()
pygame.joystick.init()

# Attempt to initialize the first joystick (PS5 controller)
if pygame.joystick.get_count() > 0:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
else:
    print("No joystick detected!")
    sys.exit()

# Your existing code for setting up the environment and drone parameters
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'envs'))
sys.path.append(env_path)

from base_drone_flying import BaseDroneFlying

base = BaseDroneFlying()
current_state = base.current_state

def cap_action_value(value, min_value, max_value):
    return max(min_value, min(value, max_value))

# Main loop modified for joystick input
while True:
    pygame.event.pump()  # Process event queue
    vertical_input = -joystick.get_axis(1)  # Y-axis of the left joystick
    torque_roll_input = joystick.get_axis(2)  # X-axis of the right joystick for roll control
    torque_pitch_input = joystick.get_axis(3)  # Y-axis of the right joystick for pitch control
    # Assuming vertical_input directly controls vertical movement, modify as needed
    # Here, we directly use vertical_input to influence the first element of the action array
    # This requires adjusting the scale of joystick input to your application's needs
    action = np.array([vertical_input, torque_roll_input*0.01, torque_pitch_input*0.01, 0])  # Modify this based on how your drone interprets actions
    
    base.render()
    print(action)
    current_state, _, _, _, _ = base.step(action)
    
