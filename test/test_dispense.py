#!/usr/bin/env python3
import sys
import yaml
import rospy
import pathlib

from motion.commander import RobotMoveGroup
from motion.utils import make_pose

from dispense.dispense import Dispenser

HOME_JOINT = [-0.4412, -2.513700624505514, 2.5439, -3.1718, -1.1295, 3.1416]
INGREDIENT = "lentils"

POURING_POSES = {
    "regular": {
        "corner": ([-0.425, 0.05, 0.5], [0.6961, -0.5842, -0.4161, 0.0310]),
        "edge": ([-0.435, 0.250, 0.485], [0.859, -0.359, -0.155, 0.331]),
    }
}


def run():
    rospy.init_node("ur5e_dispense_test")
    robot_mg = RobotMoveGroup()

    input_wt = input("Enter desired ingredient quantity (in grams): ")
    while input_wt.isdigit() and float(input_wt) > 0 and float(input_wt) <= 1000:
        # Move to home position
        assert robot_mg.go_to_joint_state(
            HOME_JOINT, cartesian_path=True, velocity_scaling=0.15
        )

        # Load ingredient-specific params
        config_dir = pathlib.Path(__file__).parent.parent
        with open(config_dir / f"config/ingredient_params/{INGREDIENT}.yaml", "r") as f:
            params = yaml.safe_load(f)

        # Move to pre-dispense position
        pos, orient = POURING_POSES[params["container"]][params["pouring_position"]]
        pre_dispense_pose = make_pose(pos, orient)
        assert robot_mg.go_to_pose_goal(
            pre_dispense_pose,
            cartesian_path=True,
            orient_tolerance=0.05,
            velocity_scaling=0.15,
        )

        # Dispense ingredient
        dispenser = Dispenser(robot_mg)
        dispenser.dispense_ingredient(params, float(input_wt))

        # Return to home position
        assert robot_mg.go_to_joint_state(
            HOME_JOINT, cartesian_path=True, velocity_scaling=0.15
        )

        # Get next entry from user
        input_wt = input("Enter desired ingredient quantity (in grams): ")


if __name__ == "__main__":
    try:
        run()
    except rospy.ROSInterruptException:
        sys.exit(1)
