#!/usr/bin/env python3
import sys
import rospy

from motion.commander import RobotMoveGroup
from motion.utils import make_pose

from dispense.dispense import Dispenser


def run():
    rospy.init_node("ur5e_dispense_test")
    robot_mg = RobotMoveGroup()

    input_wt = input("Enter desired ingredient quantity (in grams): ")
    while (input_wt.isdigit() and float(input_wt) > 0 and float(input_wt) <= 1000):
        # Move to home position
        home_pose = make_pose([-0.2, -0.1, 0.5], [-0.5, 0.5, 0.5, -0.5])
        robot_mg.go_to_pose_goal(home_pose, cartesian_path=True, velocity_scaling=0.3)

        # Move to pre-dispense position
        target_joint = [-0.0804, -1.6917, 1.6867, -2.2644, -2.1241, 2.6997]
        robot_mg.go_to_joint_state(
            target_joint, cartesian_path=False, velocity_scaling=0.15
        )

        # Dispense ingredient
        dispenser = Dispenser(robot_mg)
        dispenser.dispense_ingredient("lentils", float(input_wt))

        # Return to home position
        robot_mg.go_to_pose_goal(home_pose, cartesian_path=True, velocity_scaling=0.3)

        # Get next entry from user
        input_wt = input("Enter desired ingredient quantity (in grams): ")


if __name__ == "__main__":
    try:
        run()
    except rospy.ROSInterruptException:
        sys.exit(1)
