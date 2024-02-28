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
sys.path.append(env_path)
sys.path.append(controller_path)

from base_drone_flying import BaseDroneFlying
from LQR_controller import LQRController



if __name__ == "__main__":
    try:
        if os.path.exists(os.path.join(env_path, "drone_params.json")):
            with open(os.path.join(env_path, "drone_params.json")) as json_file:
                config = json.load(json_file)
                print("Drone parameters loaded for controller")
        else:
            raise FileNotFoundError
    except Exception as e:
        print(e)
        exit(1)

    # Linear dynamics Matrix
    g: float = config["g"]
    m: float = config["mass"]
    Ixx: float = config["inertiaMatrix"]["Ixx"]
    Iyy: float = config["inertiaMatrix"]["Iyy"]
    Izz: float = config["inertiaMatrix"]["Izz"]

    # Linear Dynamics
    A: np.ndarray = np.array(
        [
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], # phi
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], # theta
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # psi
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # p
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # q
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # r
            [0, g, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # x_dot
            [-g, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # y_dot
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # z_dot
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # x
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # y
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], # z
        ]
    )

    B: np.ndarray  = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1/Ixx, 0, 0],
            [0, 0, 1/Iyy, 0],
            [0, 0, 0, 1/Izz],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1/m, 0, 0, 0],
            [0, 0 , 0, 0],
            [0, 0, 0 , 0],
            [0, 0, 0, 0],
        ]
    )
    #Q = np.diag([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]) # for drone_params2.json
    #R = np.diag([5.0 , 5.0, 5.0, 5.0]) # for drone_params2.json
    Q = np.diag([10e2, 10e2, 10e2, 10., 10., 10., 10e2, 10e2, 10e3, 10e3, 10e3, 10e4]) # for drone_parames.json
    R = np.diag([10e3 , 10e7, 10e7, 10e7]) # for drone_parames.json
    
    base = BaseDroneFlying()

    LQR = LQRController(A, B, Q, R)

    current_state = base.current_state

    # setpoint: phi, theta, psi, p, q, r, x_dot, y_dot, z_dot, x, y, z
    setpoint = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0, 2.0, 2.5, 2.5])

    for _ in range(500):
        base.render()
        action = LQR.get_control(current_state, setpoint)
        current_state, _, _, _, _ = base.step(action)
    
    plt.show()
