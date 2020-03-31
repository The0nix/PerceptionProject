import numpy as np
import cv2, PIL, os
from cv2 import aruco
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import time
from scipy import interpolate


datadir = "frames/"
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
board = aruco.CharucoBoard_create(7, 5, 0.039, 0.0315, aruco_dict)
timestamps = []
images = []

file1 = open('frames.txt', 'r') 
Lines = file1.readlines() 
  
count = 0
# Strips the newline character 
for line in Lines: 
    images.append(datadir+line.strip())
    timestamps.append(line.strip()[:-4])

print(images)


# camera_matrix = mtx
camera_matrix = np.array([
    1121.1522216796875,
    0,
    649.2800903320312,
    0,
    1121.1522216796875,
    431.84375,
    0,
    0,
    1
]).reshape((3, 3))

dist_coeffs = np.array([
    0.20940789580345154,
    -0.494929701089859,
    0,
    0,
    0
])


retval = False

R = []
t = []
timestamps_ = []
ret = 0

for name, timestamp in zip(images, timestamps):
    # Read frame from /data/frames
    # convert frame to grayscale
    frame = cv2.imread(name)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict)

    # if enough markers were detected
    # then process the board
    if ids is not None:
        ret, ch_corners, ch_ids = aruco.interpolateCornersCharuco(corners, ids, gray, board)

    # if there are enough corners to get a reasonable result
    if ret > 5:
        aruco.drawDetectedCornersCharuco(frame, ch_corners, ch_ids, (0, 0, 255))
        rvec = np.zeros((1, 3))
        tvec = np.zeros((1, 3))
        rot = np.zeros((3, 3))
        retval, rvec, tvec = aruco.estimatePoseCharucoBoard(ch_corners, ch_ids, board, camera_matrix, dist_coeffs, rvec,
                                                            tvec)
    # if a pose could be estimated
    if retval:
        cv2.Rodrigues(rvec, rot)
        R.append(rot.T)
        t.append(-tvec)
        timestamps_.append(int(timestamp))
        frame = aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.032)

    # imshow and waitKey are required for the window
    # to open on a mac.
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(len(R), "R")
print(len(t), "t")
print(len(timestamps_), "time")
cv2.destroyAllWindows()

fig = plt.figure()
ax = fig.gca(projection='3d')
t = np.squeeze(np.array(t))
print(t.shape)
ax.plot(t[:,0], t[:,1], t[:,2])
plt.show()


np.savez("visual_data", R=np.array(R), t=np.array(t), time=np.array(timestamps_))

sensorsAccelerometer = np.array(pd.read_csv("sensorsAccelerometer.csv", header = None))
sensorsGyroscope = np.array(pd.read_csv("sensorsGyroscope.csv", header = None))
sensorsLinearAcceleration = np.array(pd.read_csv("sensorsLinearAcceleration.csv", header = None))

sensorsGyroscope = np.delete(sensorsGyroscope, np.array([1,2,6]), 1)[24:,:]
sensorsAccelerometer = np.delete(sensorsAccelerometer, np.array([1,2,6]), 1)[24:,:]
sensorsLinearAcceleration = np.delete(sensorsLinearAcceleration, np.array([1,2,6]), 1)#[4:,:]

sensorsGyroscope = sensorsGyroscope[:-18,:]
sensorsAccelerometer = sensorsAccelerometer[:-18,:]

a = np.load("visual_data.npz")

rot_vis = a["R"][3:-2,:,:]
tvec_vis = a["t"][3:-2,:,:]
time_vis = a["time"][3:-2]

t0 = time_vis[0]

time_vis = (time_vis - t0)/1e9

sensorsGyroscope[:,0] = (sensorsGyroscope[:,0] - t0)/1e9
sensorsAccelerometer[:,0] = (sensorsAccelerometer[:,0] - t0)/1e9
sensorsLinearAcceleration[:,0] = (sensorsLinearAcceleration[:,0] - t0)/1e9

linear_t_old = sensorsLinearAcceleration[:,0]
linear_t_new = sensorsAccelerometer[:,0]

f1 = interpolate.interp1d(linear_t_old, sensorsLinearAcceleration[:,1])
f2 = interpolate.interp1d(linear_t_old, sensorsLinearAcceleration[:,2])
f3 = interpolate.interp1d(linear_t_old, sensorsLinearAcceleration[:,3])

sensorsLinearAcceleration_x = f1(linear_t_new)
sensorsLinearAcceleration_y = f2(linear_t_new)
sensorsLinearAcceleration_z = f3(linear_t_new)

sensorsLinearAcceleration_interpolated = np.vstack((linear_t_new, sensorsLinearAcceleration_x, sensorsLinearAcceleration_y, sensorsLinearAcceleration_z))

sensorsLinearAcceleration_interpolated = sensorsLinearAcceleration_interpolated.T

plt.figure(figsize=(10,10))
plt.plot(sensorsLinearAcceleration[:,0],sensorsLinearAcceleration[:,3])
plt.plot(sensorsLinearAcceleration_interpolated[:,0], sensorsLinearAcceleration_interpolated[:,3], alpha = 0.5)
plt.show()

a1 = np.ones(time_vis.shape[0])
b1 = np.ones(sensorsLinearAcceleration_interpolated.shape[0])
c1 = np.ones(sensorsAccelerometer.shape[0])

plt.figure(figsize=(10,10))
plt.scatter(time_vis, a1)
plt.scatter(sensorsLinearAcceleration_interpolated[:,0], b1)
plt.scatter(sensorsAccelerometer[:,0], c1, alpha = 0.5)
plt.show()

np.savez("dataset_real", gyro = sensorsGyroscope, acc = sensorsAccelerometer, acc_linear = sensorsLinearAcceleration_interpolated, visual_rot = rot_vis, visual_tvec = tvec_vis, visual_time = time_vis)
