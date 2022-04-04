"""
***************************************************************************
Acknowledgements:
Some of the code is adapted from:
    Modern Robotics: Mechanics, Planning, and Control - Code Library
    URL: https://github.com/NxRLab/ModernRobotics
***************************************************************************
"""
import math
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


def TransInv(T):
    """Inverts a homogeneous transformation matrix
    :param T: A homogeneous transformation matrix
    :return: The inverse of T
    Uses the structure of transformation matrices to avoid taking a matrix
    inverse, for efficiency.
    Example input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        np.array([[1,  0, 0,  0],
                  [0,  0, 1, -3],
                  [0, -1, 0,  0],
                  [0,  0, 0,  1]])
    """
    R, p = TransToRp(T)
    Rt = np.array(R).T
    return np.r_[np.c_[Rt, -np.dot(Rt, p)], [[0, 0, 0, 1]]]


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
    """
    Convert a Twist message into a numpy array
    """
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
    """
    Converts a numpy array into a Twist message
    """
    twist = Twist()
    twist.linear.x, twist.linear.y, twist.linear.z = np_array[:3]
    twist.angular.x, twist.angular.y, twist.angular.z = np_array[3:]
    return twist


def quaternion2numpy(quat):
    """
    Converts a quaternion message into a numpy vector 
    """
    return np.array([quat.x, quat.y, quat.z, quat.w], dtype=np.float)


def position2numpy(pos):
    """
    Converts a Vector3/Point message into a numpy vector 
    """
    return np.array([pos.x, pos.y, pos.z], dtype=np.float)


# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0


def pose2matrix(pose):
    """
    Return homogeneous rotation matrix from a quaternion and translation
    """
    q = quaternion2numpy(pose.orientation)
    t = position2numpy(pose.position)
    nq = np.dot(q, q)
    if nq < _EPS:
        matrix = np.identity(4)
    else:
        q *= math.sqrt(2.0 / nq)
        q = np.outer(q, q)
        matrix = np.array(
            [
                [1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3], 0],
                [q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3], 0],
                [q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1], 0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float,
        )
    matrix[3, :3] = t
    return matrix
