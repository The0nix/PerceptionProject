import numpy as np

import jacobian


class EKF:
    def __init__(self, x_init, P_init):
        self.x_dim = x_init.shape[0]
        self.xs = [x_init]
        self.us = []
        self.P = P_init

        self.xs_xyz = [x_init[:3]]

    def _add_x(self, x):
        self.xs.append(x)
        self.xs_xyz.append(x[:3])

    def _update_x(self, x):
        self.xs[-1] = x
        self.xs_xyz[-1] = x[:3]

    def predict(self, u, delta_t):
        cur_x = self.xs[-1]
        new_x = jacobian.f(x=cur_x, u=u, w=0, delta_t=delta_t)
        self._add_x(new_x)
        self.us.append(u)
        self.P = F @ self.P @ F.T + W @ Q @ W.T

    def update(self, z):
        cur_x = self.xs[-1]
        P_inv = np.linalg.inv(self.P)
        K = P_inv @ H.T @ np.linalg.inv(H @ P_inv @ H.T + V @ R @ V.T)
        new_x = cur_x + K @ (z - utils.h(cur_x, 0))
        self._update_x(new_x)
        self.P = (self.eye(self.x_dim) - K @ H) @ self.P
