#!/usr/bin/env python
# coding: utf-8

"""
- We need to establish a connection to the robot with the following comand:
roslaunch ur_robot_driver ur3_bringup.launch robot_ip:=10.31.56.102 kinematics_config:=${HOME}/Calibration/ur3_calibration.yaml

- Then, we ned to activate moovit server:
roslaunch ur3_moveit_config ur3_moveit_planning_execution.launch

- Finally, we can run the program
rosrun AIManager pick_and_place.py

"""

import copy
import random
from math import pi
import rospy
import time
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import String
from AIManager.srv import GetActions


from ur_icam_description.robotUR import RobotUR

# Publisher information
PUBLISHER = rospy.Publisher('/tasks/done', Float64MultiArray, queue_size=10)
# Global variable for myRobot
MY_ROBOT = RobotUR()


BOX_X = 0.30  # Box size in meters
BOX_Y = 0.44  # Box size in meters
BOX_CENTER_ANGULAR = [2.7776150703430176, -1.5684941450702112, 1.299912452697754, -1.3755658308612269,
                      -1.5422008673297327, -0.3250663916217249]
BOX_CENTER_CARTESIAN = [-0.31899288568, -0.00357907370787, 0.226626573286]

PICK_MOVEMENT_DISTANCE = 0.215


def generate_random_state(BOX_X, BOX_Y):
    coordinate_x = random.uniform(-BOX_X / 2, BOX_X / 2)
    coordinate_y = random.uniform(-BOX_Y / 2, BOX_Y / 2)
    return [coordinate_x, coordinate_y]


def relative_move(x, y, z):
    waypoints = []
    wpose = MY_ROBOT.get_current_pose().pose
    if x:
        wpose.position.x -= x  # First move up (x)
        waypoints.append(copy.deepcopy(wpose))
    if y:
        wpose.position.y -= y  # Second move forward/backwards in (y)
        waypoints.append(copy.deepcopy(wpose))
    if z:
        wpose.position.z += z  # Third move sideways (z)
        waypoints.append(copy.deepcopy(wpose))

    MY_ROBOT.exec_cartesian_path(waypoints)


def calculate_relative_movement(relative_coordinates):
    absolute_coordinates_x = BOX_CENTER_CARTESIAN[0] - relative_coordinates[0]
    absolute_coordinates_y = BOX_CENTER_CARTESIAN[1] - relative_coordinates[1]

    current_pose = MY_ROBOT.get_current_pose()

    x_movement = current_pose.pose.position.x - absolute_coordinates_x
    y_movement = current_pose.pose.position.y - absolute_coordinates_y

    return x_movement, y_movement


def calculate_current_coordinates():
    absolut_coordinate_x=MY_ROBOT.get_current_pose().pose.position.x
    absolut_coordinate_y=MY_ROBOT.get_current_pose().pose.position.y

    relative_coordinate_x = BOX_CENTER_CARTESIAN[0] - absolut_coordinate_x
    relative_coordinate_y = BOX_CENTER_CARTESIAN[1] - absolut_coordinate_y

    return [relative_coordinate_x,relative_coordinate_y]


# This function defines the movements that robot should make depending on the action listened
def take_action(action):
    distance = 0.02 # Movement in metres
    print(action)
    if action == 'north':
        take_north(distance)
    elif action == 'south':
        take_south(distance)
    elif action == 'east':
        take_east(distance)
    elif action == 'west':
        take_west(distance)
    elif action == 'pick':
        pick_and_place(MY_ROBOT.get_current_pose().pose.position.z - PICK_MOVEMENT_DISTANCE)
    elif action == 'random_state':
        go_to_random_state()


# Action north: positive x
def take_north(distance):
    relative_move(distance, 0, 0)


# Action south: negative x
def take_south(distance):
    relative_move(-distance, 0, 0)


# Action east: negative y
def take_east(distance):
    relative_move(0, -distance, 0)


# Action west: positive y
def take_west(distance):
    relative_move(0, distance, 0)


# Action pick: Pick and place
def pick_and_place(z_distance):
    # MY_ROBOT.open_gripper()
    # rospy.sleep(3)
    relative_move( 0, 0, -z_distance)
    # MY_ROBOT.close_gripper()
    # rospy.sleep(3)
    relative_move(0, 0, z_distance)


def go_to_random_state():
    # Move robot to random positions using relative moves. Get coordinates
    relative_coordinates = generate_random_state(BOX_X, BOX_Y)
    # Calculate the new coordinates
    x_movement, y_movement = calculate_relative_movement(relative_coordinates)
    # Move the robot to the random state
    relative_move(x_movement, y_movement, 0)


def get_action():
    relative_coordinates = calculate_current_coordinates()
    rospy.wait_for_service('get_actions')
    try:
        get_actions = rospy.ServiceProxy('get_actions', GetActions)
        return get_actions(relative_coordinates[0], relative_coordinates[1]).action
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)


# Function to control the moment when the actions are done, so they can be published.
# def action_is_done():
#     # expected_coordinates=calculate_current_coordinates
#     # while relative_coordinates != expected_coordinates:
#     # time.sleep(0.5)
#     relative_coordinates = calculate_current_coordinates()
#
#     data_to_send = Float64MultiArray()  # the data to be sent, initialise the array
#     data_to_send.data = relative_coordinates  # assign the array with the value you want to send
#     PUBLISHER.publish(data_to_send)


# def callback(data):
#     rospy.loginfo(rospy.get_caller_id() + 'performing action %s',data.data )
#     # Select the action for the current state based on the actions published by the talker
#     take_action(data.data)


# def listener():
#
#     # In ROS, nodes are uniquely named. If two nodes with the same
#     # name are launched, the previous one is kicked off. The
#     # anonymous=True flag means that rospy will choose a unique
#     # name for our 'listener' node so that multiple listeners can
#     # run simultaneously.
#     # rospy.init_node('arm_controller', anonymous=True)
#
#     rospy.Subscriber('/tasks/action', String, callback)
#
#     # spin() simply keeps python from exiting until this node is stopped
#     rospy.spin()


if __name__ == '__main__':

    rospy.init_node('robotUR')
    # Test of positioning with angular coordinates
    targetReached = MY_ROBOT.go_to_joint_state(BOX_CENTER_ANGULAR)

    if targetReached:
        print("Target reachead")
    else:
        print("Target not reached")

    # Let's put the robot in a random position to start, creation of new state
    take_action('random_state')

    # Init listener node
    # listener()

    while True:
        action = get_action()
        take_action(action)