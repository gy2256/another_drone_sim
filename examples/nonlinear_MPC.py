import sys
import os
import json
import numpy as np
import nmpc
import matplotlib.pyplot as plt


env_path = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "envs")
)
controller_path = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "control")
)
config_path = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config")
)
sys.path.append(env_path)
sys.path.append(controller_path)

from base_drone_flying import BaseDroneFlying





def load_config(config_file):
    try:
        with open(config_file) as json_file:
            return json.load(json_file)
    except FileNotFoundError:
        print(f"Error: Config file {config_file} not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Config file {config_file} cannot be decoded.")
        sys.exit(1)



def main():
    drone_params = os.path.join(config_path, "drone_params.json")
    drone_config = load_config(drone_params)



    base = BaseDroneFlying()
    nmpc_controller = nmpc.construct_mpc(optimizer="NMPC_Crazyflie",
                                         horizon=5,
                                         dt=drone_config["dt"])

    current_state = base.current_state

    # setpoint: phi, theta, psi, p, q, r, x_dot, y_dot, z_dot, x, y, z
    setpoint_t = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 3.0, 2.5]).reshape(-1, 1)
    repeated_setpoint = np.tile(setpoint_t, (1, 800))

    for _ in range(800):
        base.render()
        action = nmpc_controller(current_state, repeated_setpoint)
        print(action)
        current_state, _, _, _, _ = base.step(action)

    plt.show()


if __name__ == "__main__":
    main()
