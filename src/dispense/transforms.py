"""
***************************************************************************
Acknowledgements:
Some of the code is adapted from:
    Modern Robotics: Mechanics, Planning, and Control - Code Library
    URL: https://github.com/NxRLab/ModernRobotics
***************************************************************************
"""

import numpy as np
from geometry_msgs.msg import Twist


def VecToso3(omg):
    """Converts a 3-vector to an so(3) representation
    :param omg: A 3-vector
    :return: The skew symmetric representation of omg
    Example Input:
        omg = np.array([1, 2, 3])
    Output:
        np.array([[ 0, -3,  2],
                  [ 3,  0, -1],
                  [-2,  1,  0]])
    """
    return np.array([[0, -omg[2], omg[1]], [omg[2], 0, -omg[0]], [-omg[1], omg[0], 0]])


def TransToRp(T):
    """Converts a homogeneous transformation matrix into a rotation matrix
    and position vector
    :param T: A homogeneous transformation matrix
    :return R: The corresponding rotation matrix,
    :return p: The corresponding position vector.
    Example Input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        (np.array([[1, 0,  0],
                   [0, 0, -1],
                   [0, 1,  0]]),
         np.array([0, 0, 3]))
    """
    T = np.array(T)
    return T[0:3, 0:3], T[0:3, 3]


def Adjoint(T):
    """
    CHANGED!!!!!

    Computes the adjoint representation of a homogeneous transformation
    matrix
    :param T: A homogeneous transformation matrix
    :return: The 6x6 adjoint representation [AdT] of T
    Example Input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        np.array([[1, 0,  0, 0, 0,  0],
                  [0, 0, -1, 0, 0,  0],
                  [0, 1,  0, 0, 0,  0],
                  [0, 0,  3, 1, 0,  0],
                  [3, 0,  0, 0, 0, -1],
                  [0, 0,  0, 0, 1,  0]])
    """
    R, p = TransToRp(T)
    return np.r_[np.c_[R, np.dot(VecToso3(p), R)], np.c_[np.zeros((3, 3)), R]]


def twist2numpy(twist):
    return np.array(
        [
            twist.linear.x,
            twist.linear.y,
            twist.linear.z,
            twist.angular.x,
            twist.angular.y,
            twist.angular.z,
        ],
        dtype=np.float,
    )


def TransformTwist(twist, T):
    """
    Converts a twist using the provided transformation
    """
    return np.matmul(Adjoint(T), twist)


def numpy2twist(np_array):
    twist = Twist()
    twist.linear.x, twist.linear.y, twist.linear.z = np_array[:3]
    twist.angular.x, twist.angular.y, twist.angular.z = np_array[3:]
    return twist


def quaternion2numpy(quat):
    return np.array([quat.x, quat.y, quat.z, quat.w], dtype=np.float)
