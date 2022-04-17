#!/usr/bin/env python3
import sys
import yaml
import rospy
import pathlib

from motion.commander import RobotMoveGroup
from motion.utils import make_pose

from dispense.dispense import Dispenser

PRE_DISPENSE = [-1.2334, -2.2579, 2.1997, -2.6269, -0.3113, 2.6590]
INGREDIENT = "peanuts"

POURING_POSES = {
    "regular": {
        "corner": ([-0.300, -0.040, 0.510], [0.671, -0.613, -0.414, 0.048]),
        "edge": ([-0.385, 0.240, 0.510], [0.910, -0.324, -0.109, 0.235]),
    },
    "liquid": {"corner": ([-0.265, -0.03, 0.460], [0.633, -0.645, -0.421, 0.082])},
}


def run():
    rospy.init_node("ur5e_dispense_test")
    robot_mg = RobotMoveGroup()

    input_wt = input("Enter desired ingredient quantity (in grams): ")
    while input_wt.isdigit() and float(input_wt) > 0 and float(input_wt) <= 1000:
        # Move to pre-dispense position
        assert robot_mg.go_to_joint_state(
            PRE_DISPENSE, cartesian_path=True, velocity_scaling=0.15
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
            velocity_scaling=0.3,
        )

        # Dispense ingredient
        dispenser = Dispenser(robot_mg)
        _ = dispenser.dispense_ingredient(params, float(input_wt))

        # Return to pre-dispense position
        assert robot_mg.go_to_joint_state(
            PRE_DISPENSE, cartesian_path=True, velocity_scaling=0.3
        )

        # Get next entry from user
        input_wt = input("Enter desired ingredient quantity (in grams): ")


if __name__ == "__main__":
    try:
        run()
    except rospy.ROSInterruptException:
        sys.exit(1)
