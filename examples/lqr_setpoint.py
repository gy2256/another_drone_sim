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

    base = BaseDroneFlying()

    for _ in range(300):
        base.render()
        action = np.array([3.0, 0.0, 0.0, -0.00001])
        base.step(action)

    plt.show()
