import numpy as np


def C_b_v(angles):
    """

    :param angles: Euler angles, np.ndarray, shape: (3,1)
    :return: transition matrix from b-frame to v-frame, np.ndarray, shape: (3,3)
    """
    phi, theta, psi = angles.flatten()

    result = np.zeros(shape=(3, 3))
    #first row
    result[0, 0] = np.cos(psi) * np.cos(theta)
    result[0, 1] = np.cos(psi) * np.sin(theta) * np.sin(phi) - np.sin(psi) * np.cos(phi)
    result[0, 2] = np.cos(psi) * np.sin(theta) * np.cos(phi) + np.sin(psi) * np.sin(phi)

    # second row
    result[1, 0] = np.sin(psi) * np.cos(theta)
    result[1, 1] = np.sin(psi) * np.sin(theta) * np.sin(phi) + np.cos(psi) * np.cos(phi)
    result[1, 2] = np.sin(psi) * np.sin(theta) * np.cos(phi) - np.cos(psi) * np.sin(phi)

    # third row
    result[2, 0] = -np.sin(theta)
    result[2, 1] = np.cos(theta) * np.sin(phi)
    result[2, 2] = np.cos(theta) * np.cos(phi)

    return result


def f_euler_update(x, u, w, delta_t):
    """

    :param x: state vector, np.ndarray, shape: (15,1)
    :param u: measurements vector, np.ndarray, shape: (6,1)
    :param w: noise vector, np.ndarray, shape: (6,1)
    :param delta_t: time step, scalar
    :return: deltas of eulaer angles, np.ndarray, shape: (3,1)
    """

    omega_x, omega_y, omega_z = u.flatten()[:3]
    phi, theta, psi = x.flatten()[:3]

    result = np.ones(shape=3)
    result[0] = (omega_y * np.sin(phi) + omega_z * np.cos(phi)) * np.tan(theta) + omega_x
    result[1] = omega_y * np.cos(phi) - omega_z * np.sin(phi)
    result[2] = (omega_y * np.sin(phi) + omega_z * np.cos(phi)) * (1./np.cos(theta))

    return result.reshape(-1, 1) * delta_t

def omega_unbiased(omega, bias, noise):
    return omega - bias - noise


def acc_unbiased(acc, bias, noise):
    return acc - bias - noise


def f(x, u, w, delta_t, g_v=np.array([0, 0, 9.81])):
    """

    :param x: state vector, np.ndarray, shape: (15,1)
    :param u: measurements vector, np.ndarray, shape: (6,1)
    :param w: noise vector, np.ndarray, shape: (6,1)
    :param delta_t: time step, scalar
    :param g_v: acceleration of gravity, np.ndarray: shape: (3,)
    :return: state vector at the next time step, np.ndarray, shape: (15,1)
    """

    result = np.zeros(shape=15)
    angles = x.flatten()[:3]
    pose_coordinates = x.flatten()[3:6] # x,y,z
    velocity = x.flatten()[6:9] # v_x, v_y, v_z
    bias_omega = x.flatten()[9:12] # bias in gyroscope
    bias_acc = x.flatten()[12:] # bias in accelerometer

    omega_imu = u.flatten()[:3] # measurements from gyroscope
    acc_imu = u.flatten()[3:] # measurements from accelerometer

    noise_omega = w.flatten()[:3] # omega noise
    noise_acc = w.flatten()[3:] # acceleration noise

    u_unbiased = np.hstack((omega_unbiased(omega=omega_imu, bias=bias_omega, noise=noise_omega),
                            acc_unbiased(acc=acc_imu, bias=bias_acc, noise=noise_acc)))

    trans_matrix = C_b_v(angles)

    result[:3] = angles + f_euler_update(x=x, u=u_unbiased, w=w, delta_t=delta_t).flatten()
    result[3:6] = pose_coordinates + velocity * delta_t + \
                  0.5 * delta_t**2 * (trans_matrix @ u_unbiased[3:] + g_v)
    result[6:9] = velocity + delta_t * (trans_matrix @ u_unbiased[3:] + g_v)
    result[9:12] = bias_omega
    result[12:15] = bias_acc

    return result.reshape(-1, 1)
