#!/usr/bin/env python3
import os
import yaml
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Union, List
from collections import deque

import rospy
from geometry_msgs.msg import Pose
from tf.transformations import (
    quaternion_matrix,
    translation_matrix,
    rotation_from_matrix,
)

from motion.commander import RobotMoveGroup
from sensor_interface.msg import Weight

import dispense.transforms as T
import dispense.primitives as prim


T_STEP = 0.01
MAX_ROT_ACC = np.pi / 4
MIN_ROT_ACC = -2 * MAX_ROT_ACC
MAX_ROT_VEL = np.pi / 32
MIN_ROT_VEL = -4 * MAX_ROT_VEL

ANGLE_LIMIT = (1 / 2) * np.pi
DERIVATIVE_WINDOW = 0.1  # has to greater than equal to T_STEP

# offsets from wrist_link_3
CONTAINER_OFFSET = np.array([0.040, 0.030, 0.250], dtype=np.float)


def get_transform(curr_pose: Pose, container_offset: Union[List, np.ndarray]):
    """
    Get the transform from the robot base frame to the container
    pouring edge/corner

    Inputs:
        curr_pose        - Current pose of the robot
        container_offset - Offset of the robot pouring edge/corner
    """
    curr_quat = T.quaternion2numpy(curr_pose.orientation)
    quat_transform = quaternion_matrix(curr_quat)
    container_transform = translation_matrix(container_offset)
    total_tranform = np.matmul(quat_transform, container_transform)
    return total_tranform


def get_rotation(start_T, curr_T):
    start_T_inv = T.TransInv(start_T)
    disp = np.matmul(start_T_inv, curr_T)
    angle, axis, _ = rotation_from_matrix(disp)
    return angle, axis


class Dispenser:
    def __init__(self, robot_mg):
        self.wt_subscriber = rospy.Subscriber(
            "/cooking_pot/weighing_scale", Weight, callback=self._weight_callback
        )
        self.rate = rospy.Rate(1 / T_STEP)
        self.robot_mg = robot_mg
        self._data = None

    def _weight_callback(self, data):
        self._data = data

    def get_weight(self):
        if self._data is None:
            rospy.logerr("No values received from the publisher")
            raise
        return self._data.weight

    def dispense_ingredient(
        self, ingredient_params, target_wt, err_tolerance=None, log_data=True
    ):
        # allow weighing scale measurement to be read
        rospy.sleep(0.2)

        # set ingredient-specific params
        err_tolerance = (
            ingredient_params["tolerance"] if err_tolerance is None else err_tolerance
        )
        self.ctrl_params = ingredient_params["controller"]

        # set run-specific params
        self.log_data = log_data
        self.rate.sleep()
        self.start_wt = self.get_weight()
        self.vel = self.last_vel = 0
        self.start_joint = self.robot_mg.get_current_joints()

        # setup logger
        if self.log_data:
            np.set_printoptions(precision=4, suppress=True)
            log_dir = "logs"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            log_file = "log_{0}.txt".format(datetime.now().strftime("%b-%d--%H-%M-%S"))
            self.out_file = open(log_dir + "/" + log_file, "w")
            self.out_file.write(
                "{0:10} {1:10} {2:10}\n".format(
                    "error", "velocity", "acceleration", "iter time"
                )
            )

        # Dispense ingredient
        rospy.loginfo("Dispensing started...")
        if self.ctrl_params["type"] == "pd":
            success = self.run_pd_control(target_wt, err_tolerance)

        # Move robot to start position
        self.robot_mg.go_to_joint_state(
            self.start_joint,
            cartesian_path=True,
            velocity_scaling=0.4,
            acc_scaling=0.4,
        )

        dispensed_wt = self.get_weight() - self.start_wt
        if (dispensed_wt - target_wt) > err_tolerance:
            rospy.logerr(
                f"Dispensed amount exceeded the tolerance...\nRequested Qty: {target_wt:0.2f}g \t Dispensed Qty: {dispensed_wt:0.2f}g"
            )
            success = False
        if success:
            rospy.loginfo(
                f"Ingredient dispensed successfuly...\nRequested Qty: {target_wt:0.2f}g \t Dispensed Qty: {dispensed_wt:0.2f}g"
            )

        if self.log_data:
            self.out_file.close()

    def run_pd_control(self, target_wt, err_tolerance):
        assert DERIVATIVE_WINDOW >= T_STEP
        start_T = T.pose2matrix(self.robot_mg.get_current_pose())
        if self.ctrl_params["shaking"]:
            shake_generator = prim.SinusoidalTrajectory(
                time_interval=T_STEP, **self.ctrl_params["shake_params"]
            )
        error = target_wt
        wt_fb_acc = deque(maxlen=int(DERIVATIVE_WINDOW / T_STEP) + 1)
        base_raw_twist = np.array(
            [0, 0, 0] + self.ctrl_params["rot_axis"], dtype=np.float
        )

        while error > err_tolerance:
            iter_start_time = time.time()
            curr_wt = self.get_weight()
            error = target_wt - (curr_wt - self.start_wt)
            wt_fb_acc.append(curr_wt)

            p_term = min(self.ctrl_params["p_gain"] * error, MAX_ROT_VEL)
            d_term = (
                self.ctrl_params["d_gain"]
                * (min(wt_fb_acc) - max(wt_fb_acc))
                / DERIVATIVE_WINDOW
            )
            self.vel = p_term + d_term

            delta_vel = self.vel - self.last_vel
            if np.sign(delta_vel) == 1 and delta_vel / T_STEP > MAX_ROT_ACC:
                self.vel = self.last_vel + MAX_ROT_ACC * T_STEP
            elif np.sign(delta_vel) == -1 and delta_vel / T_STEP < MIN_ROT_ACC:
                self.vel = self.last_vel + MIN_ROT_ACC * T_STEP
            self.vel = np.clip(self.vel, MIN_ROT_VEL, MAX_ROT_VEL)

            raw_twist = self.vel * base_raw_twist
            if self.ctrl_params["shaking"]:
                raw_twist[:3] += shake_generator.get_twist()
            curr_pose = self.robot_mg.get_current_pose()
            twist_transform = get_transform(curr_pose, CONTAINER_OFFSET)
            twist = T.TransformTwist(raw_twist, twist_transform)
            twist = T.numpy2twist(twist)

            self.robot_mg.send_cartesian_vel_trajectory(twist)
            accel = self.vel - self.last_vel
            self.last_vel = self.vel

            if (
                np.abs(get_rotation(start_T, T.pose2matrix(curr_pose))[0])
                >= ANGLE_LIMIT
            ):
                rospy.logerr(
                    "Container does not seem to have sufficient ingredient quantity..."
                )
                return False

            self.rate.sleep()
            if self.log_data:
                self.out_file.write(
                    "{:10.2f} {:10.4f} {:10.4f} {:10.4f}\n".format(
                        error, self.last_vel, accel, time.time() - iter_start_time
                    )
                )

        # Retract the container to the starting position
        while True:
            iter_start_time = time.time()
            curr_wt = self.get_weight()
            error = target_wt - (curr_wt - self.start_wt)

            curr_pose = self.robot_mg.get_current_pose()
            ang, ax = get_rotation(start_T, T.pose2matrix(curr_pose))
            ang = np.sign(np.dot(ax, self.ctrl_params["rot_axis"])) * ang

            if ang < 0.005:
                break

            self.vel = -2 * ang
            delta_vel = self.vel - self.last_vel
            if np.sign(delta_vel) == 1 and delta_vel / T_STEP > MAX_ROT_ACC:
                self.vel = self.last_vel + MAX_ROT_ACC * T_STEP
            elif np.sign(delta_vel) == -1 and delta_vel / T_STEP < MIN_ROT_ACC:
                self.vel = self.last_vel + MIN_ROT_ACC * T_STEP
            self.vel = np.clip(self.vel, MIN_ROT_VEL, MAX_ROT_VEL)

            raw_twist = self.vel * base_raw_twist
            twist_transform = get_transform(curr_pose, CONTAINER_OFFSET)
            twist = T.TransformTwist(raw_twist, twist_transform)
            twist = T.numpy2twist(twist)

            self.robot_mg.send_cartesian_vel_trajectory(twist)
            accel = self.vel - self.last_vel
            self.last_vel = self.vel

            self.rate.sleep()
            if self.log_data:
                self.out_file.write(
                    "{:10.2f} {:10.4f} {:10.4f} {:10.4f}\n".format(
                        error, self.last_vel, accel, time.time() - iter_start_time
                    )
                )

        rospy.loginfo("PD control phase completed...")
        return True
