# Instructions

The following packages are necessary to run the ingredient dispensing:

- [ur_motion](https://github.com/ratatouille-robotics/ur_motion)

- [sensor_interface](https://github.com/ratatouille-robotics/sensor_interface)
<BR> <BR>

The following steps have to executed to run ingredient dispensing in isolation:

1. Turn on the robot. Connect the robot and control PC using LAN. Turn on the weighing scale and connect it to the control PC.

2. On the ROS side, start the Universal Robot ROS Driver to interface with the robot

    ```
    roslaunch ur_motion ur5e_bringup.launch robot_ip:=10.0.0.2
    ```

3. Start the MoveIt Motion Planning interface
    ```
    roslaunch ur5e_moveit_config ur5e_moveit_planning_execution.launch
    ```

    If visualization is necessary, you can also start RViz
    ```
    roslaunch ur5e_moveit_config moveit_rviz.launch rviz_config:=$(rospack find ur5e_moveit_config)/launch/moveit.rviz
    ```

4. Start the streaming of weighing scale measurements by running
    ```
    roslaunch sensor_interface start_sensors.launch force_torque:=0 auto_cooking_station:=0 sensing_station:=0
    ```

5. Move the robot to a convenient position and place the container containing the desired ingredient onto the robot gripper.

6. Load the correct program on the UR pedant. Run the program so that the robot can
receive commands from the control PC.

7. To begin dispensing of the ingredient, change the variable `INGREDIENT` in the `test_dispense.py` script to the loaded ingredient name.

8. Run the following command to begin the dispensing sequence
    ```
    rosrun dispense test_dispense.py
    ```