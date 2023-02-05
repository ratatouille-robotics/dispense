#!/usr/bin/env python3
import os
import sys
import csv
import yaml
import time
import rospy
import pathlib
import numpy as np

from datetime import datetime

from motion.commander import RobotMoveGroup
from dispense.dispense import Dispenser


DISPENSE_HOME = [-1.2334, -2.2579, 2.1997, -2.6269, -0.3113, 2.6590]
LOG_DIR = "src/dispense/logs"

AUTOMATIC_RUN_MODE = True

# Only for running in AUTOMATIC_RUN_MODE
AVAILABLE_WEIGHT = 700
REFILL_THREHOLD = 40

INGREDIENT = "peanuts"
MINIMUM_WEIGHT = 15
MAXIMUM_WEIGHT = 120
NUM_RUNS = 25


def acquire_input() -> float:
    """
    Get weight to be dispensed from the user
    """
    input_wt = input("Enter desired ingredient quantity (in grams). Enter -1 to stop: ")

    try:
        input_wt = float(input_wt)
    except ValueError:
        input_wt = -1.0

    return input_wt


def run(log_results=False):
    rospy.init_node("ur5e_dispense_test")
    robot_mg = RobotMoveGroup()
    dispenser = Dispenser(robot_mg)
    num_runs = 0

    if log_results:
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
        log_file = "{0}_eval_{1}.csv".format(
            INGREDIENT, datetime.now().strftime("%b-%d--%H-%M-%S")
        )
        out_file = open(LOG_DIR + "/" + log_file, "w")
        csv_writer = csv.writer(out_file)
        csv_writer.writerow(["S.No", "Requested", "Dispensed", "Time Taken"])

    # Load ingredient-specific params
    config_dir = pathlib.Path(__file__).parent.parent
    with open(config_dir / f"config/ingredient_params/{INGREDIENT}.yaml", "r") as f:
        params = yaml.safe_load(f)
    available_wt = AVAILABLE_WEIGHT

    while num_runs < NUM_RUNS:
        # Move to dispense-home position
        assert robot_mg.go_to_joint_state(
            DISPENSE_HOME, cartesian_path=True, velocity_scaling=0.15
        )
        if AUTOMATIC_RUN_MODE:
            requested_wt = np.random.uniform(low=MINIMUM_WEIGHT, high=MAXIMUM_WEIGHT)
            if (requested_wt + REFILL_THREHOLD > available_wt):
                usr_input = input("Refill Container and enter available weight (in grams): ")
                available_wt = float(usr_input)
        else:
            requested_wt = acquire_input()
            if requested_wt < 0 or requested_wt > 1000:
                break

        num_runs += 1

        # Dispense ingredient
        start_time = time.time()
        dispensed_wt = dispenser.dispense_ingredient(params, requested_wt)
        dispense_time = time.time() - start_time
        if AUTOMATIC_RUN_MODE:
            available_wt -= dispensed_wt

        if log_results:
            csv_writer.writerow(
                [num_runs, np.round(requested_wt, 2), np.round(dispensed_wt, 2), np.round(dispense_time, 1)]
            )
            out_file.flush()

    if log_results:
        out_file.close()


if __name__ == "__main__":
    try:
        run(log_results=False)
    except rospy.ROSInterruptException:
        sys.exit(1)
