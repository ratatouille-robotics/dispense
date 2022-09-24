#!/usr/bin/env python3
import sys
import yaml
import rospy
import pathlib

from motion.commander import RobotMoveGroup

from dispense.dispense import Dispenser


INGREDIENT = "peanuts"

DISPENSE_HOME = [-1.2334, -2.2579, 2.1997, -2.6269, -0.3113, 2.6590]


def acquire_input(message: str) -> float:
    """
    Get weight to be dispensed from the user
    """
    input_wt = input(message)

    try:
        input_wt = float(input_wt)
    except ValueError:
        input_wt = -1

    return input_wt


def run():
    rospy.init_node("ur5e_dispense_test")
    robot_mg = RobotMoveGroup()

    input_wt = acquire_input("Enter desired ingredient quantity (in grams): ")

    while (input_wt) > 0 and float(input_wt) <= 1000:
        # Move to dispense-home position
        assert robot_mg.go_to_joint_state(
            DISPENSE_HOME, cartesian_path=True, velocity_scaling=0.15
        )

        # Load ingredient-specific params
        config_dir = pathlib.Path(__file__).parent.parent
        with open(config_dir / f"config/ingredient_params/{INGREDIENT}.yaml", "r") as f:
            params = yaml.safe_load(f)

        # Dispense ingredient
        dispenser = Dispenser(robot_mg)
        _ = dispenser.dispense_ingredient(params, float(input_wt))

        # Return to pre-dispense position
        assert robot_mg.go_to_joint_state(
            DISPENSE_HOME, cartesian_path=True, velocity_scaling=0.3
        )

        # Get next entry from user
        input_wt = acquire_input("Enter desired ingredient quantity (in grams): ")


if __name__ == "__main__":
    try:
        run()
    except rospy.ROSInterruptException:
        sys.exit(1)