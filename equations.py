# import numpy as np
import jax.numpy as np
from jax import jacfwd
from jax.ops import index_update


def C_b_v(angles):
    """

    :param angles: Euler angles, np.ndarray, shape: (3,1)
    :return: transition matrix from b-frame to v-frame, np.ndarray, shape: (3,3)
    """
    phi, theta, psi = angles.flatten()

    result = np.zeros(shape=(3, 3))
    #first row
    result = index_update(result, (0, 0), np.cos(psi) * np.cos(theta))
    result = index_update(result, (0, 1), np.cos(psi) * np.sin(theta) * np.sin(phi) - np.sin(psi) * np.cos(phi))
    result = index_update(result, (0, 2), np.cos(psi) * np.sin(theta) * np.cos(phi) + np.sin(psi) * np.sin(phi))

    # second row
    result = index_update(result, (1, 0), np.sin(psi) * np.cos(theta))
    result = index_update(result, (1, 1), np.sin(psi) * np.sin(theta) * np.sin(phi) + np.cos(psi) * np.cos(phi))
    result = index_update(result, (1, 2), np.sin(psi) * np.sin(theta) * np.cos(phi) - np.cos(psi) * np.sin(phi))

    # third row
    result = index_update(result, (2, 0), -np.sin(theta))
    result = index_update(result, (2, 1), np.cos(theta) * np.sin(phi))
    result = index_update(result, (2, 2), np.cos(theta) * np.cos(phi))

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

    result = np.zeros(shape=3)
    result = index_update(result, 0, (omega_y * np.sin(phi) + omega_z * np.cos(phi)) * np.tan(theta) + omega_x)
    result = index_update(result, 1, omega_y * np.cos(phi) - omega_z * np.sin(phi))
    result = index_update(result, 2, (omega_y * np.sin(phi) + omega_z * np.cos(phi)) * (1./np.cos(theta)))

    return result.reshape(-1, 1) * delta_t


def omega_unbiased(omega, bias, noise):
    return omega - bias - noise


def acc_unbiased(acc, bias, noise):
    return acc - bias - noise


def f(x, u, w, delta_t, g_v=None):
    """

    :param x: state vector, np.ndarray, shape: (15,1)
    :param u: measurements vector, np.ndarray, shape: (6,1)
    :param w: noise vector, np.ndarray, shape: (6,1)
    :param delta_t: time step, scalar
    :param g_v: acceleration of gravity, np.ndarray: shape: (3,)
    :return: state vector at the next time step, np.ndarray, shape: (15,1)
    """
    if g_v is None:
        g_v = np.array([0, -9.81, 0])

    result = np.zeros(shape=15)
    angles = x.flatten()[:3]
    pose_coordinates = x.flatten()[3:6] # x,y,z
    velocity = x.flatten()[6:9] # v_x, v_y, v_z

    omega_imu = u.flatten()[:3] # measurements from gyroscope
    acc_imu = u.flatten()[3:] # measurements from accelerometer

    noise_omega = w.flatten()[:3] # omega noise
    noise_acc = w.flatten()[3:] # acceleration noise

    bias_omega = x.flatten()[9:12] # bias in gyroscope
    bias_acc = x.flatten()[12:] # bias in accelerometer

    u_unbiased = np.hstack((omega_unbiased(omega=omega_imu, bias=bias_omega, noise=noise_omega),
                            acc_unbiased(acc=acc_imu, bias=bias_acc, noise=noise_acc)))

    trans_matrix = C_b_v(angles)

    result = index_update(result, slice(0, 3), angles + f_euler_update(x=x, u=u_unbiased, w=w, delta_t=delta_t).flatten())
    result = index_update(result, slice(3, 6), pose_coordinates + velocity * delta_t + \
                  0.5 * delta_t**2 * (trans_matrix @ u_unbiased[3:] + g_v))
    result = index_update(result, slice(6, 9), velocity + delta_t * (trans_matrix @ u_unbiased[3:] + g_v))
    result = index_update(result, slice(9, 12), bias_omega)
    result = index_update(result, slice(12, 15), bias_acc)

    return result.reshape(-1, 1)


def jac_f_euler_angles(x, u, delta_t):
    """

    :param x: state vector, np.ndarray, shape: (15,1)
    :param u: measurements vector, np.ndarray, shape: (6,1)
    :param delta_t: time step, scalar
    :return: the derivetive of f_euler_update function by angles.
                np.ndarray, shape: (3, 3)
    """

    phi, theta, psi = x.flatten()[:3]
    omega_x, omega_y, omega_z = u.flatten()[:3]

    result = np.zeros(shape=(3,3))

    # first row
    result = index_update(result, (0, 0), (omega_y * np.cos(phi) - omega_z * np.sin(phi)) * np.tan(theta))
    result = index_update(result, (0, 1), (omega_y * np.sin(phi) + omega_z * np.cos(phi)) * (1./np.cos(theta))**2)

    # second row
    result = index_update(result, (1, 0),  -omega_y * np.sin(phi) - omega_z * np.cos(phi))

    # third row
    result = index_update(result, (2, 0), (omega_y * np.cos(phi) - omega_z * np.sin(phi))*(1./np.cos(theta)))
    result = index_update(result, (2, 1), (omega_y * np.sin(phi) + omega_z * np.cos(phi))*(np.sin(theta)/(np.cos(theta)**2)))

    return result * delta_t

def h(x, v):
    result = x[:6].reshape(-1) + v.reshape(-1)
    return result


def c_b_v_angles(angles, acc):
    C = C_b_v(angles)
    return C @ acc

def jac_c_b_v_angles(angles, acc): # uff...
    """

    :param angles: Euler angles, np.ndarray, shape: (3,1)
    :param acc: accelerations, np.ndarray, shape: (3, 1)
    :return: the derivetive of C_b_v @ acc function by angles.
                np.ndarray, shape: (3, 3)
    """

    phi, theta, psi = angles.flatten()
    a_x, a_y, a_z = acc.flatten()

    result = np.zeros(shape=(3,3))

    # first row
    result = index_update(result, (0, 0), a_y * (np.cos(psi) * np.sin(theta) * np.cos(phi) + np.sin(psi) * np.sin(phi)) + \
                   a_z * (-np.cos(psi) * np.sin(theta) * np.sin(phi) + np.sin(psi) * np.cos(phi)))
    result = index_update(result, (0, 1), a_x * (-np.cos(psi) * np.sin(theta)) + \
                   a_y * (np.cos(psi) * np.cos(theta) * np.sin(phi)) + \
                   a_z * (np.cos(psi) * np.cos(theta) * np.cos(phi)))
    result = index_update(result, (0, 2), a_x * (-np.sin(psi) * np.cos(theta)) + \
                   a_y * (-np.sin(psi) * np.sin(theta) * np.sin(phi) - np.cos(psi) * np.cos(phi)) + \
                   a_z * (-np.sin(psi) * np.sin(theta) * np.cos(phi) + np.cos(psi) * np.sin(phi)))

    # second row
    result = index_update(result, (1, 0), a_y * (np.sin(psi) * np.sin(theta) * np.cos(phi) - np.cos(psi) * np.sin(phi)) + \
                   a_z * (-np.sin(psi) * np.sin(theta) * np.sin(phi) - np.cos(psi) * np.cos(phi)))
    result = index_update(result, (1, 1), a_x * (-np.sin(psi) * np.sin(theta)) + \
                   a_y * (np.sin(psi) * np.cos(theta) * np.sin(phi)) + \
                   a_z * (np.sin(psi) * np.cos(theta) * np.cos(phi)))
    result = index_update(result, (1, 2), a_x * (np.cos(psi) * np.cos(theta)) + \
                   a_y * (np.cos(psi) * np.sin(theta) * np.sin(phi) - np.sin(psi) * np.cos(phi)) + \
                   a_z * (np.cos(psi) * np.sin(theta) * np.cos(phi) + np.sin(psi) * np.sin(phi)))

    result = index_update(result, (2, 0), a_y * (np.cos(theta) * np.cos(psi)) + \
                   a_z * (-np.cos(theta) * np.sin(phi)))
    result = index_update(result, (2, 1), a_x * (-np.cos(theta)) + \
                   a_y * (-np.sin(theta) * np.sin(phi)) + \
                   a_z * (-np.sin(theta) * np.cos(phi)))
    result = index_update(result, (2, 2), 0)

    return result


def jac_f_x(x, u, w, delta_t):
    """

    :param x: state vector, np.ndarray, shape: (15,1)
    :param u: measurements vector, np.ndarray, shape: (6,1)
    :param w: noise vector, np.ndarray, shape: (6,1)
    :param delta_t: time step, scalar
    :return: jacobian of transition function with respect to state
                np.ndarray, shape: (15, 15)
    """
    angles = x.flatten()[:3]

    omega_imu = u.flatten()[:3]  # measurements from gyroscope
    acc_imu = u.flatten()[3:]  # measurements from accelerometer

    noise_omega = w.flatten()[:3]  # omega noise
    noise_acc = w.flatten()[3:]  # acceleration noise

    bias_omega = x.flatten()[9:12]  # bias in gyroscope
    bias_acc = x.flatten()[12:]  # bias in accelerometer

    u_unbiased = np.hstack((omega_unbiased(omega=omega_imu, bias=bias_omega, noise=noise_omega),
                            acc_unbiased(acc=acc_imu, bias=bias_acc, noise=noise_acc)))

    result = np.zeros(shape=(15, 15))

    result = index_update(result, [slice(0, 3), slice(0,3)], jac_f_euler_angles(x=x, u=u_unbiased, delta_t=delta_t))

    result = index_update(result, [slice(3, 6), slice(0,3)], 0.5 * delta_t**2 * jac_c_b_v_angles(angles=angles, acc=u_unbiased.flatten()[3:]))
    result = index_update(result, [slice(3, 6), slice(3,6)], np.identity(3))
    result = index_update(result, [slice(3, 6), slice(6,9)], delta_t * np.identity(3))

    result = index_update(result, [slice(6, 9), slice(0,3)], delta_t * jac_c_b_v_angles(angles=angles, acc=u_unbiased.flatten()[3:]))
    result = index_update(result, [slice(6, 9), slice(6,9)], np.identity(3))

    result = index_update(result, [slice(9, 12), slice(9,12)], np.identity(3))
    result = index_update(result, [slice(12, 15), slice(12,15)], np.identity(3))

    return result

_make_F = jacfwd(f, argnums=0)
def make_F(x, u, w, delta_t):
    return _make_F(x, u, w, delta_t).reshape(x.shape[0], x.shape[0])

_make_W = jacfwd(f, argnums=2)
def make_W(x, u, w, delta_t):
    return _make_W(x, u, w, delta_t).reshape(x.shape[0], w.shape[0])

_make_H = jacfwd(h, argnums=0)
def make_H(x, v):
    return _make_H(x, v).reshape(6, x.shape[0])

_make_V = jacfwd(h, argnums=1)
def make_V(x, v):
    return _make_V(x, v).reshape(6, 6)
