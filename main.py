#!/usr/bin/env python
import argparse

import numpy as np

import utils
import ekf


parser = argparse.ArgumentParser(description='EKF localization by video')
parser.add_argument('-a', '--animate', action='store_true', help='Show animation of trajectory')
parser.add_argument('-c', '--config', type=str, help='Config path', default='config.yml')
parser.add_argument('-o', '--output', type=str, help='File to output trajectory to')


if __name__ == '__main__':
    args = parser.parse_args()
    animate = args.animate
    config_path = args.config
    output = args.output

    config = utils.read_config(config_path)

    accelerometer = np.loadtxt(config['accelerometer'], delimiter=',')
    gyroscope = np.loadtxt(config['gyroscope'], delimiter=',')
    markers = np.loadtxt(config['markers'], delimiter=',')
    timestamps = accelerometer[:,0]

    if animate:
        plotter = utils.Plotter()

    state_dim = 15
    x_init = np.zeros(state_dim)  # TODO: Make proper initialization
    P_init = 1e-1 * np.eye(state_dim)
    model = ekf.EKF(x_init=x_init, P_init=P_init)


    ##### Main loop #####
    marker_ix = 0
    prev_ts = timestamps[0] - (timestamps[1] - timestamps[0])
    for imu_ix, ts in enumerate(timestamps):
        u = np.hstack([gyroscope[imu_ix,1:], accelerometer[imu_ix,1:]])
        delta_t = ts - prev_ts
        model.predict(u, delta_t)
        if markers[marker_ix,0] <= ts:
            z = markers[marker_ix,1:]
            model.update(z)
            marker_ix += 1
        prev_ts = ts
        if animate:
            plotter.plot_trajectory(model.xs_xyz)

    if output:
        np.savetxt(output, np.vstack(model.xs_xyz))
