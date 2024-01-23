import numpy as np
import gymnasium as gym
import os

env_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import json
import os.path

import torch


class BaseDroneFlying(gym.Env):
    def __init__(self):
        super(BaseDroneFlying, self).__init__()
        try:
            if os.path.exists(os.path.join(env_path, "drone_params.json")):
                with open(os.path.join(env_path, "drone_params.json")) as json_file:
                    self.config = json.load(json_file)

                print("Drone parameters loaded for environment")
            else:
                raise FileNotFoundError
        except Exception as e:
            print(e)
            print("Error in reading config file")
            exit(1)

        self.state_space_shape = (12,)
        self.action_space_shape = (4,)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=self.state_space_shape, dtype=np.float32
        )

        self.dt = self.config["dt"]
        self.g = self.config["g"]
        self.mass = self.config["mass"]
        self.Ixx = self.config["inertiaMatrix"]["Ixx"]
        self.Iyy = self.config["inertiaMatrix"]["Iyy"]
        self.Izz = self.config["inertiaMatrix"]["Izz"]
        self.max_thrust = self.config["ControlLimits"]["max_thrust"]
        self.max_torque = self.config["ControlLimits"]["max_torque"]
        self.max_roll = self.config["StateLimits"]["max_roll"]
        self.max_pitch = self.config["StateLimits"]["max_pitch"]
        self.max_yaw = self.config["StateLimits"]["max_yaw"]
        self.max_roll_rate = self.config["StateLimits"]["max_roll_rate"]
        self.max_pitch_rate = self.config["StateLimits"]["max_pitch_rate"]
        self.max_yaw_rate = self.config["StateLimits"]["max_yaw_rate"]
        self.stable_thrust = self.mass * self.g

        self.action_space = gym.spaces.Box(
            low=np.array(
                [-self.max_thrust, -self.max_torque, -self.max_torque, -self.max_torque]
            ),
            high=np.array(
                [self.max_thrust, self.max_torque, self.max_torque, self.max_torque]
            ),
            dtype=np.float32,
        )

        self.current_state = np.zeros(self.state_space_shape)
        self.current_state[2] = 0.5  # set height to be 1 meter

        # Initialize the matplotlib 3D plot
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")

        # Set the labels and title
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("Drone Movement")

    def RK4(self, dynamics, s, a, dt):
        k1 = dynamics(s, a)
        k2 = dynamics(s + dt / 2 * k1, a)
        k3 = dynamics(s + dt / 2 * k2, a)
        k4 = dynamics(s + dt * k3, a)
        return s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def step(self, action):
        done = False

        # Construct the time derivative of the state
        def dynamics(s, a):
            phi, theta, psi, p, q, r, x_dot, y_dot, z_dot, x, y, z = s
            u1, u2, u3, u4 = a  # Thrust, torque_x, torque_y, torque_z

            # Precompute some common trigonometric values
            c_phi = np.cos(phi)
            s_phi = np.sin(phi)
            c_theta = np.cos(theta)
            s_theta = np.sin(theta)
            c_psi = np.cos(psi)
            s_psi = np.sin(psi)
            t_theta = np.tan(theta)

            # Position dynamics
            x_ddot = (c_phi * s_theta * c_psi + s_phi * s_psi) * u1 / self.mass
            y_ddot = (c_phi * s_theta * s_psi - s_phi * c_psi) * u1 / self.mass
            z_ddot = (c_phi * c_theta) * u1 / self.mass - self.g

            # Angular velocity dynamics
            p_dot = (u2 - (self.Iyy - self.Izz) * q * r) / self.Ixx
            q_dot = (u3 - (self.Izz - self.Ixx) * p * r) / self.Iyy
            r_dot = (u4 - (self.Ixx - self.Iyy) * p * q) / self.Izz

            state_dot = np.array([p, q, r, p_dot, q_dot, r_dot, x_ddot, y_ddot, z_ddot, x_dot, y_dot, z_dot])

            return state_dot

        # Integrate the state using Runge-Kutta 4th order method
        action[0] += self.stable_thrust # Add thrust to ensure hovering
        next_state = self.RK4(dynamics, self.current_state, action, self.dt)
        self.current_state = next_state

        # return observation, reward, terminated, truncated, info
        return next_state, 0, done, False, {}
        

    def rotation_matrix(self, phi, theta, psi):
        """Create a rotation matrix for given roll (phi), pitch (theta), yaw (psi)."""
        R_x = np.array(
            [[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]]
        )

        R_y = np.array(
            [
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)],
            ]
        )

        R_z = np.array(
            [[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]]
        )

        return np.dot(R_z, np.dot(R_y, R_x))

    def render(self):
        # Clear the previous plot
        self.ax.clear()

        # Get the current state of the drone
        # current_state = torch.zeros(12)  # Replace with actual state
        # current_state = self.observation_space.sample()

        # Extract drone's position from the state
        phi, theta, psi, p, q, r, x_dot, y_dot, z_dot, x, y, z = self.current_state

        # Plot the drone's position
        self.ax.scatter(x, y, z, c="r", marker="o", s=10)

        # Define the length of the drone arms
        arm_length = 0.3

        arms = np.array(
            [
                [arm_length, 0, 0],
                [-arm_length, 0, 0],
                [0, arm_length, 0],
                [0, -arm_length, 0],
            ]
        )

        # Calculate the rotation matrix
        R = self.rotation_matrix(phi, theta, psi)
        # Apply the rotation and translation
        arms_rotated = np.dot(arms, R.T) + np.array([x, y, z])

        # Draw the arms
        for i in range(0, 4, 2):
            self.ax.plot(
                [arms_rotated[i][0], arms_rotated[i + 1][0]],
                [arms_rotated[i][1], arms_rotated[i + 1][1]],
                [arms_rotated[i][2], arms_rotated[i + 1][2]],
                c="b",
            )

        # Set the plot limits
        self.ax.set_xlim([-5, 5])
        self.ax.set_ylim([-5, 5])
        self.ax.set_zlim([0, 5])

        # Draw the plot
        plt.draw()
        plt.pause(0.001)  # Pause to update the plot

    def reset(self):
        self.current_state = self.observation_space.sample()
        # return observation, info

    def close(self):
        pass
