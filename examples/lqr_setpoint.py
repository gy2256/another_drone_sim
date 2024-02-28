import sys
import os
import json
import numpy as np

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
from LQR_controller import LQRController


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


def create_matrices(drone_config, LQR_config):

    # Extract parameters from drone_config
    g = drone_config["g"]  # gravitational acceleration
    m = drone_config["mass"]  # mass of the drone
    Ixx = drone_config["inertiaMatrix"]["Ixx"]  # moment of inertia around x-axis
    Iyy = drone_config["inertiaMatrix"]["Iyy"]  # moment of inertia around y-axis
    Izz = drone_config["inertiaMatrix"]["Izz"]  # moment of inertia around z-axis

    # Construct the Q and R matrices from LQR_config
    Q = np.diag(LQR_config["Q"])
    R = np.diag(LQR_config["R"])

    # Linear dynamics Matrix, https://sal.aalto.fi/publications/pdf-files/eluu11_public.pdf
    A = np.array(
        [
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # phi
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # theta
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # psi
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # p
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # q
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # r
            [0, g, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # x_dot
            [-g, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # y_dot
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # z_dot
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # x
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # y
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # z
        ]
    )

    B = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1 / Ixx, 0, 0],
            [0, 0, 1 / Iyy, 0],
            [0, 0, 0, 1 / Izz],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1 / m, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )

    return A, B, Q, R


def main():
    drone_params = os.path.join(config_path, "drone_params.json")
    LQR_params = os.path.join(config_path, "LQR_params.json")

    drone_config = load_config(drone_params)
    LQR_config = load_config(LQR_params)

    A, B, Q, R = create_matrices(drone_config, LQR_config)
    base = BaseDroneFlying()
    LQR = LQRController(A, B, Q, R)

    current_state = base.current_state

    # setpoint: phi, theta, psi, p, q, r, x_dot, y_dot, z_dot, x, y, z
    setpoint = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 3.0, 2.5])

    for _ in range(500):
        base.render()
        action = LQR.get_control(current_state, setpoint)
        print(action)
        current_state, _, _, _, _ = base.step(action)

    plt.show()


if __name__ == "__main__":
    main()
