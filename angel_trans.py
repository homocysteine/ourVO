from scipy.spatial.transform import Rotation as R
import numpy as np
from math import cos, sin, pi, atan2, asin


def quaternion2euler(quaternion):
    r = R.from_quat(quaternion)
    euler = r.as_euler('xyz', degrees=True)
    return euler


def euler2quaternion(euler):
    r = R.from_euler('xyz', euler, degrees=True)
    quaternion = r.as_quat()
    return quaternion


def euler2rotation(euler):
    r = R.from_euler('xyz', euler, degrees=True)
    rotation_matrix = r.as_matrix()
    return rotation_matrix


def rotation2euler(rotation):
    r = R.from_matrix(rotation)
    euler = r.as_euler('xyz', degrees=True)
    return euler

# a = np.array([45, 45, 45])
# b = euler2rotation(a)
# print(b)
# print(rotation2euler(b))
