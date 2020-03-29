import numpy as np
import equations
# import equations_np as equations



class EKF:
    def __init__(self, x_init, P_init, Q, R):
        self.x_dim = x_init.shape[0]
        self.xs = [x_init.ravel()]
        self.us = []
        self.P = P_init

        self.xs_xyz = [x_init.ravel()[3:6]]
        self.Q = Q
        self.R = R

    def _add_x(self, x):
        self.xs.append(x)
        self.xs_xyz.append(x[3:6])

    def _update_x(self, x):
        self.xs[-1] = x
        self.xs_xyz[-1] = x[3:6]

    def predict(self, u, delta_t):
        cur_x = self.xs[-1]
        w = np.zeros(u.shape[0])
        new_x = equations.f(x=cur_x, u=u, w=w, delta_t=delta_t).ravel()
        self._add_x(new_x)
        self.us.append(u)
        F = equations.make_F(cur_x, u, w, delta_t)
        W = equations.make_W(cur_x, u, w, delta_t)
        self.P = F @ self.P @ F.T + W @ self.Q @ W.T

    def update(self, z):
        cur_x = self.xs[-1]
        v = np.zeros(6)
        P_inv = np.linalg.inv(self.P)
        H = equations.make_H(cur_x, v)
        V = equations.make_V(cur_x, v)
        K = P_inv @ H.T @ np.linalg.inv(H @ P_inv @ H.T + V @ self.R @ V.T)
        new_x = cur_x.ravel() + K @ (z - equations.h(cur_x, np.zeros(6)))
        self._update_x(new_x)
        self.P = (np.eye(self.x_dim) - K @ H) @ self.P
