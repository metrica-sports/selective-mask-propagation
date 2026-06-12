"""Minimal Kalman filter matching filterpy.kalman.KalmanFilter interface.

Only implements the subset used by SORT: F, H, R, P, Q matrices,
x state vector, predict() and update(z).
"""

import numpy as np


class KalmanFilter:
    def __init__(self, dim_x, dim_z):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.x = np.zeros((dim_x, 1))
        self.P = np.eye(dim_x)
        self.Q = np.eye(dim_x)
        self.F = np.eye(dim_x)
        self.H = np.zeros((dim_z, dim_x))
        self.R = np.eye(dim_z)
        self._I = np.eye(dim_x)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        if z is None:
            return
        z = np.asarray(z)
        if z.ndim == 1:
            z = z.reshape(-1, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (self._I - K @ self.H) @ self.P
