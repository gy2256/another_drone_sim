import scipy
import numpy as np

class LQRController:
    def __init__(self, A, B, Q, R):
        # Solve Ricatti equation
        X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))
        self.K = np.asarray(np.matrix(scipy.linalg.inv(R) @ B.T @ X))


    def get_control(self, state, state_setpoint):
        error = state - state_setpoint
        U = -np.dot(self.K, error.T)
        return U
