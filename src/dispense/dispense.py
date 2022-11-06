#!/usr/bin/env python3
import os
import time
import numpy as np
from numbers import Number
from datetime import datetime
from collections import deque
from typing import List, Optional, Tuple, Union

import signal
import rospy
from geometry_msgs.msg import Pose, Twist
from tf.transformations import (
    quaternion_matrix,
    translation_matrix,
    rotation_from_matrix,
)

from motion.utils import make_pose
from motion.commander import RobotMoveGroup
from sensor_interface.msg import Weight

import dispense.transforms as T
import dispense.primitives as prim


T_STEP = 0.1
CONTROL_STEP = 0.005

MAX_ROT_ACC = np.pi / 4
MIN_ROT_ACC = -2 * MAX_ROT_ACC
MAX_ROT_VEL = np.pi / 32
MIN_ROT_VEL = -2 * MAX_ROT_VEL

ANGLE_LIMIT = {
    "regular": {"corner": (1 / 3) * np.pi, "edge": (1 / 2) * np.pi},
    "spout": {"corner": (2 / 5) * np.pi}
}

DERIVATIVE_WINDOW = 0.1  # Time window over which to calculate derivative. Has to be greater than equal to T_STEP
LOGICAL_TIMEOUT_WINDOW = 30  # Timeout window after which to stop dispening in logical control
MIN_WT_DISPENSED = 0.1  # Minimum weight to dispense to avoid timeout
WEIGHING_SCALE_FLUCTUATION = 0.3

# offsets from wrist_link_3/flange/tool0
CONTAINER_OFFSET = {
    "regular": np.array([0.040, 0.060, 0.250], dtype=np.float),
    "spout": np.array([0.035, 0.150, 0.250], dtype=np.float),
    "holes": np.array([0.040, 0.060, 0.250], dtype=np.float),
}
#TODO : After testing, clean up code by removing two of the 3 options below
# new fine table
POURING_POSES = {
    "regular": {
        "corner": ([-0.03028, 0.3199, 0.4599], [-0.04147, 0.908, 0.32678, 0.2587]),
        "edge": ([0.154685, 0.3851, 0.500], [0.4147, -0.8723, -0.24315, 0.089]),
    },
    "spout": {"corner": ([-0.03022, 0.24, 0.440], [0.00817, 0.903, 0.3557, 0.2396])},
    "holes": {"corner": ([0.0497, 0.33, 0.405], [-0.771, 0.287, 0.207, 0.529])}
}

# #new table
# POURING_POSES = {
#     "regular": {
#         "corner": ([-0.03028, 0.2999, 0.5099], [-0.04147, 0.908, 0.32678, 0.2587]),
#         "edge": ([0.224685, 0.3851, 0.520], [0.4147, -0.8723, -0.24315, 0.089]),
#     },
#     "spout": {"corner": ([-0.03022, 0.264, 0.460], [0.00817, 0.903, 0.3557, 0.2396])},
#     "holes": {"corner": ([0.0697, 0.36, 0.520], [-0.771, 0.287, 0.207, 0.529])}
# }

#original table
# POURING_POSES = {
#     "regular": {
#         "corner": ([-0.300, -0.030, 0.510], [0.671, -0.613, -0.414, 0.048]),
#         "edge": ([-0.385, 0.225, 0.520], [0.910, -0.324, -0.109, 0.235]),
#     },
#     "spout": {"corner": ([-0.265, -0.03, 0.460], [0.633, -0.645, -0.421, 0.082])},
#     "holes": {"corner": ([-0.360, 0.070, 0.520], [0.749, 0.342, -0.520, -0.228])}
# }


def get_transform(reference_T_flange: Pose, container_offset: Union[List, np.ndarray]) -> np.ndarray:
    """
    Get the pose of the of the container pouring edge/corner with respect to a frame positioned at the 
    flange point and oriented along the base frame

    Inputs:
        reference_T_flange     - Current pose of the robot
        container_offset       - Offset of the container pouring edge/corner in the flange/tool0 frame
    """
    reference_T_flange_rotation = quaternion_matrix(T.quaternion2numpy(reference_T_flange.orientation))
    flange_T_container_tip = translation_matrix(container_offset)
    required_transform = np.matmul(reference_T_flange_rotation, flange_T_container_tip)
    return required_transform


def get_rotation(reference_T_start: np.ndarray, reference_T_end: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Find the rotational displacement between two poses
    """
    start_T_reference = T.TransInv(reference_T_start)
    start_T_end = np.matmul(start_T_reference, reference_T_end)
    angle, axis, _ = rotation_from_matrix(start_T_end)
    return angle, axis

# class GracefulKiller:
#   kill_now = False
#   def __init__(self):
#     signal.signal(signal.SIGINT, self.exit_gracefully)
#     signal.signal(signal.SIGTERM, self.exit_gracefully)

#   def exit_gracefully(self, *args):
#     rospy.logerr("CTRL C detected in Dispense Class")
#     self.kill_now = True
    

class Dispenser:
    def __init__(self, robot_mg: RobotMoveGroup, killer) -> None:
        # Setup comm with the weighing scale
        self.wt_subscriber = rospy.Subscriber(
            "/auto_cooking_station/weighing_scale", Weight, callback=self._weight_callback
        )
        self.rate = rospy.Rate(1 / CONTROL_STEP)
        self.robot_mg = robot_mg
        self._w_data = None
        self.robot_original_pose = None
        self.killer = killer
    def _weight_callback(self, data: float) -> None:
        self._w_data = data

    def get_weight(self) -> float:
        if self._w_data is None:
            rospy.logerr("No values received from the publisher")
            raise
        return self._w_data.weight

    def get_weight_fb(self) -> Tuple[float, bool]:
        return (
            self.get_weight(),
            (rospy.Time.now() - self._w_data.header.stamp).to_sec() < 0.5,
        )
    def myhook(self):
        rospy.logerr("shutdown time!")
        twist = Twist()
        self.robot_mg.send_cartesian_vel_trajectory(twist)
        self.robot_mg.go_to_pose_goal(
            self.robot_original_pose,
            cartesian_path=True,
            orient_tolerance=0.05,
            velocity_scaling=0.75,
            acc_scaling=0.5
        )

    def dispense_ingredient(
        self,
        ingredient_params: dict,
        target_wt: Number,
        tolerance: Union[Number, None] = None,
        log_data: bool = True,
        ingredient_wt_start: Optional[float] = None,
    ) -> Number:
        # Record current robot position
        # rospy.on_shutdown(self.myhook)
        self.robot_original_pose = self.robot_mg.get_current_pose()
        robot_original_pose = self.robot_mg.get_current_pose()
        # Send dummy velocity to avoid delayed motion start on first run
        self.robot_mg.send_cartesian_vel_trajectory(T.numpy2twist(np.zeros(6, dtype=np.float)))

        # set ingredient-specific params
        tolerance = ingredient_params["tolerance"] if tolerance is None else tolerance
        self.ctrl_params = ingredient_params["controller"]
        self.lid_type = ingredient_params["container"]["lid"]
        if self.lid_type in ["none", "slot"]:
            self.lid_type = "regular"
        self.container_offset = CONTAINER_OFFSET[self.lid_type]
        err_threshold = min(tolerance, self.ctrl_params["error_threshold"])

        # set ingredient-specific limits
        self.max_rot_vel = self.ctrl_params["vel_scaling"] * MAX_ROT_VEL
        self.min_rot_vel = self.ctrl_params["vel_scaling"] * MIN_ROT_VEL
        self.max_rot_acc = self.ctrl_params["acc_scaling"] * MAX_ROT_ACC
        self.min_rot_acc = self.ctrl_params["acc_scaling"] * MIN_ROT_ACC

        # Move to dispense-start position
        pos, orient = POURING_POSES[self.lid_type][ingredient_params["pouring_position"]]
        pre_dispense_pose = make_pose(pos, orient)
        error_check = self.robot_mg.go_to_pose_goal(
            pre_dispense_pose,
            cartesian_path=True,
            orient_tolerance=0.05,
            velocity_scaling=0.75,
            acc_scaling=0.5
        )
        if not error_check:
            rospy.logerr("handling some assert error")
            return 0
        # joint_state = self.robot_mg.get_current_joints()
        # joint_state[0] -= 1.57
        # self.robot_mg.go_to_joint_state(joint_state)
        # sumthing = self.robot_mg.get_current_pose()
        # rospy.loginfo(sumthing)
        # return
        # set run-specific params
        self.log_data = log_data
        self.start_wt = self.get_weight()
        self.last_vel = 0
        self.last_acc = 0

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
            self.angle_limit = ANGLE_LIMIT[self.lid_type][ingredient_params["pouring_position"]]
            _ = self.run_pd_control(target_wt, err_threshold)
        elif self.ctrl_params["type"] == "logical":
            _ = self.run_logical_control(target_wt, err_threshold)

        # Move to dispense-start position
        error_check =  self.robot_mg.go_to_pose_goal(
            pre_dispense_pose,
            cartesian_path=True,
            orient_tolerance=0.05,
            velocity_scaling=0.5,
            acc_scaling=0.25
        )
        if not error_check:
            rospy.logerr("handling some assert error")
            return self.get_weight() - self.start_wt

        error_check =  self.robot_mg.go_to_pose_goal(
            robot_original_pose,
            cartesian_path=True,
            orient_tolerance=0.05,
            velocity_scaling=0.75,
            acc_scaling=0.5
        )
        if not error_check:
            rospy.logerr("handling some assert error")
            return self.get_weight() - self.start_wt

        dispensed_wt = self.get_weight() - self.start_wt
        if (target_wt - dispensed_wt) > tolerance:
            rospy.logerr(
                f"Dispensed amount is below tolerance...Requested Qty: {target_wt:0.2f}g \t Dispensed Qty: {dispensed_wt:0.2f}g"
            )
        elif (dispensed_wt - target_wt) > tolerance:
            rospy.logerr(
                f"Dispensed amount exceeded the tolerance...\nRequested Qty: {target_wt:0.2f}g \t Dispensed Qty: {dispensed_wt:0.2f}g"
            )
        else:
            rospy.loginfo(
                f"Ingredient dispensed successfuly...\nRequested Qty: {target_wt:0.2f}g \t Dispensed Qty: {dispensed_wt:0.2f}g"
            )

        if self.log_data:
            self.out_file.close()
        if self.killer.kill_now:
            rospy.signal_shutdown("killed while dispensing")
        return dispensed_wt

    def run_control_loop(self, velocity: float, retract: bool = False):
        last_velocity = self.last_vel
        for _ in range(int(T_STEP / CONTROL_STEP)):
            if velocity > last_velocity:
                curr_velocity = min(last_velocity + self.max_rot_acc * CONTROL_STEP, velocity)
            else:
                curr_velocity = max(last_velocity + self.min_rot_acc * CONTROL_STEP, velocity)

            # Convert the velocity into a twist
            raw_twist = curr_velocity * self.base_raw_twist
            # Convert the velocity into a twist
            if self.trans_shake_generator is not None and (not retract or abs(self.trans_shake_generator.last_v) > 1e-3):
                raw_twist[:3] += self.trans_shake_generator.get_twist()

            # Transform the frame of the twist
            curr_pose = self.robot_mg.get_current_pose()
            twist_transform = get_transform(curr_pose, self.container_offset)
            twist = T.TransformTwist(raw_twist, twist_transform)
            twist = T.numpy2twist(twist)

            self.robot_mg.send_cartesian_vel_trajectory(twist)
            self.rate.sleep()
            last_velocity = curr_velocity

    def run_pd_control(
        self,
        target_wt: Number,
        err_threshold: Number
    ) -> bool:
        """
        Run the PD controller
        """
        error_check =  DERIVATIVE_WINDOW >= T_STEP
        if not error_check:
            rospy.logerr("handling some assert error")
            return 0
        start_T = T.pose2matrix(self.robot_mg.get_current_pose())

        # Setup shake generator
        if self.ctrl_params["shaking"] and self.ctrl_params.get("trans_shake_params") is not None:
            self.trans_shake_generator = prim.SinusoidalTrajectory(
                time_interval=CONTROL_STEP,
                **self.ctrl_params["trans_shake_params"]
            )
        else:
            self.trans_shake_generator = None

        error = target_wt
        wt_fb_acc = deque(maxlen=int(DERIVATIVE_WINDOW / T_STEP) + 1)
        self.base_raw_twist = np.array([0, 0, 0] + self.ctrl_params["rot_axis"], dtype=np.float)
        success = True

        # Run controller as long as error is not within tolerance
        while (not rospy.is_shutdown()) and (error > err_threshold) and (not self.killer.kill_now):
        # while error > err_threshold
            iter_start_time = time.time()
            curr_wt, is_recent = self.get_weight_fb()
            if not is_recent:
                rospy.logerr(
                    "Weight feedback from weighing scale is too delayed. Stopping dispensing process."
                )
                success = False
                break

            wt_fb_acc.append(curr_wt)
            error = target_wt - (curr_wt - self.start_wt)
            error_rate = -(wt_fb_acc[-1] - wt_fb_acc[0]) / DERIVATIVE_WINDOW

            p_term = self.ctrl_params["p_gain"] * error
            p_term = min(p_term, self.max_rot_vel)  # clamp p-term
            d_term = self.ctrl_params["d_gain"] * error_rate
            pid_vel = p_term + d_term

            # Check if trying to go down near starting position
            curr_pose = self.robot_mg.get_current_pose()
            if (np.abs(get_rotation(start_T, T.pose2matrix(curr_pose))[0]) <= 0.1):            
                if(pid_vel < 0):
                    rospy.logerr("Trying to go down further")
                    pid_vel = 0

            # Clamp velocity based on acceleration and velocity limits
            max_vel = self.last_vel + MAX_ROT_ACC * T_STEP
            min_vel = self.last_vel + MIN_ROT_ACC * T_STEP

            pid_vel = max(min(pid_vel, max_vel), min_vel)
            pid_vel = np.clip(pid_vel, self.min_rot_vel, self.max_rot_vel)

            # Check if the angluar limits about the pouring axis have been reached
            curr_pose = self.robot_mg.get_current_pose()
            angle, axis = get_rotation(start_T, T.pose2matrix(curr_pose))
            if np.sum(axis * self.base_raw_twist[-3:]) < 0:
                angle *= -1

            if (np.abs(angle) >= self.angle_limit):
                rospy.logerr("Container does not seem to have sufficient ingredient quantity...")
                success = False
                break

            total_vel = pid_vel
            self.run_control_loop(total_vel)

            self.last_acc = (total_vel - self.last_vel) / T_STEP
            self.last_vel = total_vel

            # Check if the angluar limits about the pouring axis have been reached
            if (np.abs(get_rotation(start_T, T.pose2matrix(curr_pose))[0]) >= self.angle_limit):
                rospy.logerr("Container does not seem to have sufficient ingredient quantity...")
                success = False
                break

            if self.log_data:
                self.out_file.write(
                    "{:10.2f} {:10.4f} {:10.4f} {:10.4f}\n".format(
                        error, self.last_vel, self.last_acc, time.time() - iter_start_time
                    )
                )

        self.min_rot_vel = min(2 * MIN_ROT_VEL, self.min_rot_vel)
        # Retract the container to the starting position
        
        while (not rospy.is_shutdown()) and (not self.killer.kill_now):
        # while True:
            iter_start_time = time.time()
            curr_wt = self.get_weight()
            error = target_wt - (curr_wt - self.start_wt)

            curr_pose = self.robot_mg.get_current_pose()
            ang, ax = get_rotation(start_T, T.pose2matrix(curr_pose))
            ang = np.sign(np.dot(ax, self.ctrl_params["rot_axis"])) * ang

            if ang < 0.005:
                break

            # Ensure the velocity is within limits
            vel = -2 * ang
            vel = max(min(vel, max_vel), min_vel)
            vel = np.clip(vel, self.min_rot_vel, self.max_rot_vel)

            # Add the necessary shake twists
            raw_twist = vel * self.base_raw_twist
            if (self.trans_shake_generator is not None and abs(self.trans_shake_generator.last_v) > 1e-3):
                raw_twist[:3] += self.trans_shake_generator.get_twist()

            self.run_control_loop(vel, retract=True)
            self.last_acc = (vel - self.last_vel) / T_STEP
            self.last_vel = vel

            self.rate.sleep()
            if self.log_data:
                self.out_file.write(
                    "{:10.2f} {:10.4f} {:10.4f} {:10.4f}\n".format(
                        error, self.last_vel, self.last_acc, time.time() - iter_start_time
                    )
                )

        rospy.loginfo("PD control phase completed...")
        
        return success

    def run_logical_control(self, target_wt: Number, err_threshold: Number) -> bool:
        """
        Run a controller that keeps shaking until the specified quantity is dispensed
        """
        if self.ctrl_params.get("trans_shake_params") is not None:
            shake_generator = prim.SinusoidalTrajectory(
                time_interval=CONTROL_STEP,
                **self.ctrl_params["trans_shake_params"]
            )
        else:
            raise ValueError

        error = target_wt
        self.robot_mg.send_cartesian_vel_trajectory(T.numpy2twist(np.zeros(6, dtype=np.float)))
        success = True
        dispening_history = deque(maxlen=int(LOGICAL_TIMEOUT_WINDOW / CONTROL_STEP))
        
        while (not self.killer.kill_now) and (not rospy.is_shutdown()) and (error > max(err_threshold - WEIGHING_SCALE_FLUCTUATION, 0) or abs(shake_generator.last_v) > 1e-3):
        # while error > max(err_threshold - WEIGHING_SCALE_FLUCTUATION, 0) or abs(shake_generator.last_v) > 1e-3:
            iter_start_time = time.time()
            curr_wt, is_recent = self.get_weight_fb()
            if not is_recent:
                rospy.logerr("Weight feedback from weighing scale is too delayed. Stopping dispensing process.")
                success = False
                break

            error = target_wt - (curr_wt - self.start_wt)
            dispening_history.append(curr_wt)

            # Generate the twist and transform it into the right frame
            raw_twist = np.zeros(6, dtype=np.float)
            raw_twist[:3] += shake_generator.get_twist()
            curr_pose = self.robot_mg.get_current_pose()
            twist_transform = get_transform(curr_pose, self.container_offset)
            twist = T.TransformTwist(raw_twist, twist_transform)
            twist = T.numpy2twist(twist)

            self.robot_mg.send_cartesian_vel_trajectory(twist)

             # Check if dispensing is still going on
            # if (
            #     dispening_history.maxlen == len(dispening_history) and 
            #     np.mean(list(dispening_history)[-10:]) - np.mean(list(dispening_history)[:10]) < MIN_WT_DISPENSED
            # ):
            #     rospy.logerr("Container does not seem to have sufficient ingredient quantity...")
            #     success = False
            #     break

            self.rate.sleep()
            if self.log_data:
                self.out_file.write(
                    "{:10.2f} {:10.4f} {:10.4f} {:10.4f}\n".format(
                        error, 0, 0, time.time() - iter_start_time
                    )
                )

        self.robot_mg.send_cartesian_vel_trajectory(T.numpy2twist(np.zeros(6, dtype=np.float)))

        rospy.loginfo("Logical control phase completed...")

        return success
