#!/usr/bin/env python
# coding: utf-8

"""
- We need to connect the camera and the nodes
roslaunch ur_icam_description webcam.launch

- We need to establish a connection to the robot with the following comand:
roslaunch ur_robot_driver ur3_bringup.launch robot_ip:=10.31.56.102 kinematics_config:=${HOME}/Calibration/ur3_calibration.yaml

- Then, we ned to activate moovit server:
roslaunch ur3_moveit_config ur3_moveit_planning_execution.launch

- Activate the talker
rosrun AIManager talker.py

- Activate the node
rosrun AIManager distance_sensor_simulator.py

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
from std_msgs.msg import Float32
from AIManager.srv import GetActions
from Environment import Environment


from ur_icam_description.robotUR import RobotUR

# Publisher information
# PUBLISHER = rospy.Publisher('/tasks/done', Float64MultiArray, queue_size=10)
# Publisher for the topic switch_on_off
PUBLISHER = rospy.Publisher('switch_on_off', Bool)
# Global variable for myRobot
MY_ROBOT = RobotUR()

PICK_MOVEMENT_DISTANCE = 0.215


def generate_random_state():
    coordinate_x = random.uniform(-Environment.X_LENGTH / 2, Environment.X_LENGTH / 2)
    coordinate_y = random.uniform(-Environment.Y_LENGTH / 2, Environment.Y_LENGTH / 2)
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
    absolute_coordinates_x = Environment.CARTESIAN_CENTER[0] - relative_coordinates[0]
    absolute_coordinates_y = Environment.CARTESIAN_CENTER[1] - relative_coordinates[1]

    current_pose = MY_ROBOT.get_current_pose()

    x_movement = current_pose.pose.position.x - absolute_coordinates_x
    y_movement = current_pose.pose.position.y - absolute_coordinates_y

    return x_movement, y_movement


def calculate_current_coordinates():
    absolut_coordinate_x = MY_ROBOT.get_current_pose().pose.position.x
    absolut_coordinate_y = MY_ROBOT.get_current_pose().pose.position.y

    relative_coordinate_x = Environment.CARTESIAN_CENTER[0] - absolut_coordinate_x
    relative_coordinate_y = Environment.CARTESIAN_CENTER[1] - absolut_coordinate_y

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
        pick_and_place()
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
def pick_and_place():
    # In this function we should read the distance to the object
    up_distance = 0  # Variable were we store the distance that we have move the robot so that we can go back to the
    # original pose
    distance_ok = False  # False until target is reached
    while not distance_ok:
        # Check if the distance is the correct one
        # TODO : check if the distance is in the correct measures
        distance = rospy.wait_for_message('distance', Float32).data  # We retrieve sensor distance

        if distance <= Environment.PICK_DISTANCE:  # If distance is equal or smaller than the target distance
            # TODO : Check what kind of msg the subscriber is waiting
            PUBLISHER.publish(True)  # We try to pick the object enabling the vacuum gripper
            time.sleep(2)  # We wait some time to let the vacuum pick the object
            distance_ok = True  # End the loop by setting this variable to True
        else:  # If the distance to the objects is higher than the target distance
            difference = distance - Environment.PICK_DISTANCE  # We calculate the difference between the two distances
            up_distance += difference  # We increment the up_distance variable
            relative_move(0, 0, -difference)  # We try to move to the desired distance

    relative_move(0, 0, up_distance)  # We went back to the original pose

    object_gripped = rospy.wait_for_message('object_gripped', Bool).data
    if object_gripped:  # If we have gripped an object we place it into the desired point
        take_place()

def go_to_random_state():
    # Move robot to random positions using relative moves. Get coordinates
    relative_coordinates = generate_random_state()
    # Calculate the new coordinates
    x_movement, y_movement = calculate_relative_movement(relative_coordinates)
    # Move the robot to the random state
    relative_move(x_movement, y_movement, 0)

# Function to define the place for placing the grasped objects
def take_place():
    # First, we get the cartesian coordinates of one of the corner
    x_box,y_box = Environment.get_relative_corner(0)
    x_move, y_move = calculate_relative_movement([x_box,y_box])
    # We move the robot to the corner of the box
    relative_move(x_move, y_move, 0)
    # We calculate the trajectory for our robot to reach the box
    trajectory_x = MY_ROBOT.get_current_pose().pose.position.x - Environment.PLACE_CARTESIAN_CENTER[0]
    trajectory_y = MY_ROBOT.get_current_pose().pose.position.y - Environment.PLACE_CARTESIAN_CENTER[1]
    trajectory_z = - Environment.CARTESIAN_CENTER[2] + Environment.PLACE_CARTESIAN_CENTER[2]
    # We move the robot to the coordinates desired to place the object
    relative_move( 0, 0, trajectory_z )
    relative_move( 0, trajectory_y, 0 )
    relative_move( trajectory_x, 0, 0 )
    # Then, we left the object
    relative_move( 0, 0, -0.05)
    # Then, we switch off the vacuum gripper so the object can be placed
    # TODO : Check the correct information to send the topic
    PUBLISHER.publish(False)
    # Wait some seconds, in order to the msg to arrive to the gripper
    time.sleep(2)
    # Then the robot goes up
    relative_move(0, 0, 0.05)
    # Final we put the robot in the center of the box, the episode should finish now
    MY_ROBOT.go_to_joint_state(Environment.ANGULAR_CENTER)
    # TODO : publicar si se ha acabado el episodio

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
    targetReached = MY_ROBOT.go_to_joint_state(Environment.ANGULAR_CENTER)

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