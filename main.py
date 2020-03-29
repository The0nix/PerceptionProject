#!/usr/bin/env python
import argparse

import numpy as np
from tqdm import tqdm

import utils
import ekf


parser = argparse.ArgumentParser(description='EKF localization by video')
parser.add_argument('-a', '--animate', action='store_true', help='Show animation of trajectory')
parser.add_argument('-i', '--input', type=str, help='Input dataset path', default='data/dataset_real.npz')
parser.add_argument('-o', '--output', type=str, help='File to output trajectory to')
parser.add_argument('--no-update', action='store_true', help='Disable update step')


if __name__ == '__main__':
    args = parser.parse_args()
    animate = args.animate
    dataset_path = args.input
    output = args.output
    no_update = args.no_update

    dataset = np.load(dataset_path)
    accelerometer = dataset['acc']
    gyroscope = dataset['gyro']
    timestamps = accelerometer[:,0]
    visual_rots = dataset['visual_rot']
    visual_tvecs = dataset['visual_tvec']
    visual_ts = dataset['visual_time']

    if animate:
        plotter = utils.Plotter()

    state_dim = 15
    x_init = np.zeros(state_dim)  # TODO: Make proper initialization
    P_init = 1e-1 * np.eye(state_dim)
    model = ekf.EKF(x_init=x_init, P_init=P_init, Q=0.01 * np.eye(6))


    ##### Main loop #####
    visual_ix = 0
    prev_ts = timestamps[0] - (timestamps[1] - timestamps[0])
    for imu_ix, ts in enumerate(tqdm(timestamps)):
        u = np.hstack([gyroscope[imu_ix,1:], accelerometer[imu_ix,1:]])
        delta_t = ts - prev_ts
        model.predict(u, delta_t)
        if not no_update and visual_ts[visual_ix] <= ts:
            z = visual_ix[visual_ix,1:]
            model.update(z)
            visual_ix += 1
        prev_ts = ts
        if animate:
            plotter.plot_trajectory(model.xs_xyz)

    if output:
        np.savetxt(output, np.vstack(model.xs_xyz))
