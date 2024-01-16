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

    # dynamics, https://pdf.sciencedirectassets.com/270704/1-s2.0-S1110016822X00033/1-s2.0-S1110016821007900/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEPj%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIHyHjTDRA3uXplxRGkisSsKOyVV0Zf%2Boc8pzP9U9Mb2GAiEAxxHFMLID8n5FlSoVrQFl%2Faylyfvw8KQ6QArUw5xzDCIqvAUIof%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDOIFxFSgqKRjqzPBjyqQBXMcFD2464vaAEwgOhqA7sytZF9FM71iDvGOFlB4qRCZYUywbl74eOUoxpNXXNQ0coz2r5%2Fm8ToKvGnq9Y%2FJo9bwKCI0KUM255gyQqbnCHtJikx3DLwQlTE9O%2BSgRWMdFZ6vhVj4yO7icJGdvVYTkXAcn4kka0qAmXfrLbh1NmKyGYjLpRbge9Pwsd2m4kbtyD81lVb%2FwdNjhkqCLUeXsEb2o%2BvRM1DjYpl3HRKJDmQ2VIfQWu3NJCZahJGLGa24FUsmrV3CzAWBfZjHQg%2B7rs0wSV92N%2FD8PTAHpmmrciViUOeh%2FNOSX1MUiaVx%2BrF5A5l1zpZ9u2BeEMi%2BCib3JpkYcijk%2BZq6WBRPDHAFnlunV18RMSLTLc9HUwZh8opQK8%2Bq%2BO15uAAL%2Fih5SyyeIeXi0GZ%2FEsvrYJd84C5mmG9rqhzcLYOusxsU3TMakftpbSXL2MG%2Blxid6CkUYNzYu8YMdrVwpSvYsSFQ5Q73Mp1Wt6Ih2Oe3HYi2eX68WRGFZ8086BcWvbP72DNmL%2B5t7BNzfAt1IiqeGZ%2BgqhMxwzqLC%2BSljgE7SBdbiKl%2FBfgrdw4%2Bl8gBcL4REM8yVdeHAxqdRai1XqS0UoUbGiZO5vEbKJrc%2BAKE5BVZT7xYeYH7cBuEKUaFhrp1Ol1jCNV7Qv4vmagmMdb2HpvaiXH%2FfazbjB7OKLvqfhKXV%2F9yFwugw8LC0YW6mvqaWsBLdOxwUS2TFzUq%2FHO7D6VPwcAe84cllcRLw%2FCouR96h47qqOrwYOuDCGX4iJ4WqsTjQX8kJm511OIJjsxgFDNFciSwilS%2FK%2B%2FKbpC%2FNdfT6YFZE4uTstBoOYrlGeucL1TJfBYRnIGpmgHnOIhg2a4IpD%2FLXMRQMLTsmK0GOrEBlcvkL41LJw%2BkbCTsbpwsyNyyXm83ccjIqJsM41dDOuLocURJAI7oD24sv0A7fGS4h1uiTn0v7eOOexQrvR0bOOZdqYslKO%2BveXVd8Y36ES1%2Bt0QmVqeo1y07OD62CXr%2BusS%2B08U%2BC63G2bzhRERrSngIJxI7DR6R1O2WNHZ1fPmaF8MmOtjC24S38lwTa9bEvKBNb68ofceJw3wFiWNOBvmJlsaz%2Fektrw4i%2FkFHs0ge&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240116T080744Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYUOYJNWQX%2F20240116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=17028de1dc96dc1625308bfa9d4ab2353f01c49568847bd91edc7a9830b92b07&hash=7e8a8ba3724ddd9bd0cf36d89a79b185df9a6b850549c9c873e7e5a56103181d&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1110016821007900&tid=spdf-538e0408-b224-4075-8e07-95fc26c5bbfb&sid=0a2c1f022fe619407e4a6b9285c8d8bb4028gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=1d045d5155560604065751&rr=8464ddd36a7d52b2&cc=gb
    A: np.ndarray = np.array(
        [
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, -g, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, g, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    B: np.ndarray = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [-1 / m, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1 / Ixx, 0, 0],
            [0, 0, 1 / Iyy, 0],
            [0, 0, 0, 1 / Izz],
        ]
    )

    base = BaseDroneFlying()

    for _ in range(300):
        base.render()
        action = np.array([3.0, 0.0, 0.0, -0.00001])
        base.step(action)

    plt.show()
