#!/usr/bin/env python3
import sys
import yaml
import rospy
import pathlib

from motion.commander import RobotMoveGroup
from motion.utils import make_pose

from dispense.dispense import Dispenser

HOME_JOINT = [-0.4412, -2.513700624505514, 2.5439, -3.1718, -1.1295, 3.1416]
INGREDIENT = "peanuts"

POURING_POSES = {
    "regular": {
        "corner": ([-0.410, -0.020, 0.500], [0.671, -0.613, -0.414, 0.048]),
        "edge": ([-0.435, 0.250, 0.485], [0.852, -0.455, -0.205, 0.157]),
    },
    "liquid": {
        "corner": ([-0.365, -0.03, 0.440], [0.633, -0.645, -0.421, 0.082])
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
        _ = dispenser.dispense_ingredient(params, float(input_wt))

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
