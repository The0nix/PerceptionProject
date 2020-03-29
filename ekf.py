import numpy as np
import equations



class EKF:
    def __init__(self, x_init, P_init, Q):
        self.x_dim = x_init.shape[0]
        self.xs = [x_init]
        self.us = []
        self.P = P_init

        self.xs_xyz = [x_init[:3]]
        self.Q = Q

    def _add_x(self, x):
        self.xs.append(x)
        self.xs_xyz.append(x[:3])

    def _update_x(self, x):
        self.xs[-1] = x
        self.xs_xyz[-1] = x[:3]

    def predict(self, u, delta_t):
        cur_x = self.xs[-1]
        w = np.zeros(u.shape[0])
        new_x = equations.f(x=cur_x, u=u, w=w, delta_t=delta_t)
        self._add_x(new_x)
        self.us.append(u)
        F = equations.make_F(cur_x, u, w, delta_t)
        W = equations.make_W(cur_x, u, w, delta_t)
        self.P = F @ self.P @ F.T + W @ self.Q @ W.T

    def update(self, z):
        raise NotImplementedError
        # cur_x = self.xs[-1]
        # P_inv = np.linalg.inv(self.P)
        # K = P_inv @ H.T @ np.linalg.inv(H @ P_inv @ H.T + V @ R @ V.T)
        # new_x = cur_x + K @ (z - utils.h(cur_x, 0))
        # self._update_x(new_x)
        # self.P = (self.eye(self.x_dim) - K @ H) @ self.P
