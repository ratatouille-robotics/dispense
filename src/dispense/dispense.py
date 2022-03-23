#!/usr/bin/env python3
import os
import yaml
import time
import numpy as np
from datetime import datetime
from collections import deque
from pathlib import Path
from typing import Union, List

import rospy
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_matrix, translation_matrix

from motion.commander import RobotMoveGroup
from sensor_interface.msg import Weight

import dispense.transforms as T


T_STEP = 0.1
MAX_ROT_ACC = np.pi / 4
MIN_ROT_ACC = -2 * MAX_ROT_ACC
MAX_ROT_VEL = np.pi / 32
MIN_ROT_VEL = -4 * MAX_ROT_VEL

CONTAINER_OFFSET = np.array([0.005, 0.005, 0.250], dtype=np.float)


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


class Dispenser:
    def __init__(self, robot_mg):
        self.wt_subscriber = rospy.Subscriber(
            "/cooking_pot/weighing_scale", Weight, callback=self._weight_callback
        )
        self.rate = rospy.Rate(10.0)
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
        self, ingredient, target_wt, err_tolerance=None, log_data=True
    ):
        # load ingredinet specific parameters
        self.rate.sleep()
        dir = Path(__file__).parent.parent.parent
        with open(dir / f"config/ingredient_params/{ingredient}.yaml", "r") as f:
            params = yaml.safe_load(f)
        err_tolerance = params["tolerance"] if err_tolerance is None else err_tolerance
        self.ctrl_params = params["controller"]

        # set run-specific params
        self.log_data = log_data
        self.rate.sleep()
        self.start_wt = self.get_weight()
        self.vel = self.last_vel = 0
        self.start_joint = self.robot_mg.get_current_joints()
        self.rot_dir = 1 if self.ctrl_params["rot_dir"] == "clockwise" else -1

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
            self.run_pd_control(target_wt, err_tolerance)

        # Move robot to start position
        self.robot_mg.go_to_joint_state(
            self.start_joint,
            cartesian_path=False,
            velocity_scaling=0.4,
            acc_scaling=0.4,
        )
        rospy.loginfo("Ingredient dispensed...")
        if self.log_data:
            self.out_file.close()

    def run_pd_control(self, target_wt, err_tolerance):
        error = last_error = target_wt

        while error > err_tolerance:
            iter_start_time = time.time()
            dispensed_wt = self.get_weight() - self.start_wt
            error = target_wt - dispensed_wt

            p_term = min(self.ctrl_params["p_gain"] * error, MAX_ROT_VEL)
            d_term = self.ctrl_params["d_gain"] * (error - last_error)
            self.vel = p_term + d_term

            delta_vel = self.vel - self.last_vel
            if np.sign(delta_vel) == 1 and delta_vel / T_STEP > MAX_ROT_ACC:
                self.vel = self.last_vel + MAX_ROT_ACC * T_STEP
            elif np.sign(delta_vel) == -1 and delta_vel / T_STEP < MIN_ROT_ACC:
                self.vel = self.last_vel + MIN_ROT_ACC * T_STEP
            self.vel = np.clip(self.vel, MIN_ROT_VEL, MAX_ROT_VEL)

            raw_twist = np.array(
                [0, 0, 0, 0, 0, self.rot_dir * self.vel], dtype=np.float
            )
            twist_transform = get_transform(
                self.robot_mg.get_current_pose(), CONTAINER_OFFSET
            )
            twist = T.TransformTwist(raw_twist, twist_transform)
            twist = T.numpy2twist(twist)

            self.robot_mg.send_cartesian_vel_trajectory(twist)
            accel = self.vel - self.last_vel
            self.last_vel = self.vel
            last_error = error

            self.rate.sleep()
            if self.log_data:
                self.out_file.write(
                    "{:10.2f} {:10.4f} {:10.4f} {:10.4f}\n".format(
                        error, self.last_vel, accel, time.time() - iter_start_time
                    )
                )
        rospy.loginfo("PD control phase completed...")
