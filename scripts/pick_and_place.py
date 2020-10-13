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
from geometry_msgs.msg import Pose

from ur_icam_description.robotUR import RobotUR

#Publisher information
PUBLISHER = rospy.Publisher('/tasks/done', String, queue_size=10)
#Global variable for myRobot
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

def pick_and_place(z_distance):
    # MY_ROBOT.open_gripper()
    # rospy.sleep(3)
    relative_move( 0, 0, -z_distance)
    # MY_ROBOT.close_gripper()
    # rospy.sleep(3)
    relative_move(0, 0, z_distance)

def callback(data):
    rospy.loginfo(rospy.get_caller_id() + 'performing action %s',data.data )
    # Select the action for the current state based on the actions published by the talker
    take_action(data.data)
    PUBLISHER.publish('True')

def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('arm_controller', anonymous=True)

    rospy.Subscriber('/tasks/action', String, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

# This function defines the movements that robot should make depending on the action listened
def take_action(x_movement, y_movement,action):
    distance=0.02 #Movement in metres

    if action is 'north':
        x_movement, y_movement=take_north(distance)
        relative_move(x_movement, y_movement, 0)
    elif action is 'south':
        x_movement, y_movement=take_south(distance)
        relative_move(x_movement, y_movement, 0)
    elif action is 'east':
        x_movement, y_movement=take_east(distance)
        relative_move(x_movement, y_movement, 0)
    elif action is 'west':
        x_movement, y_movement=take_west(distance)
        relative_move(x_movement, y_movement, 0)
    elif action is 'pick':
        pick_and_place(MY_ROBOT.get_current_pose().pose.position.z - PICK_MOVEMENT_DISTANCE)

    return x_movement, y_movement

#Action north: positive x

def take_north(distance):
    current_pose = MY_ROBOT.get_current_pose()
    x = current_pose.pose.position.x + distance
    y = current_pose.pose.position.y
    return x,y

# Action south: negative x
def take_south(distance):
    current_pose = MY_ROBOT.get_current_pose()
    x = current_pose.pose.position.x - distance
    y = current_pose.pose.position.y
    return x,y

# Action east: negative y
def take_east(distance):
    current_pose = MY_ROBOT.get_current_pose()
    x = current_pose.pose.position.x
    y = current_pose.pose.position.y - distance
    return x,y
# Action west: positive y
def take_west(distance):
    current_pose = MY_ROBOT.get_current_pose()
    x = current_pose.pose.position.x
    y = current_pose.pose.position.y + distance
    return x,y

#Action pick: Pick and place
#def take_pick(x_movement, y_movement,distance):
#    return x, y

def go_to_random_state():
    # Move robot to random positions using relative moves. Get coordinates
    relative_coordinates = generate_random_state(BOX_X, BOX_Y)
    # Calculate the new coordinates
    x_movement, y_movement = calculate_relative_movement(relative_coordinates)
    # Move the robot to the random state
    relative_move(x_movement, y_movement, 0)


if __name__ == '__main__':

    rospy.init_node('robotUR')
    # Test of positioning with angular coordinates
    targetReached = MY_ROBOT.go_to_joint_state(BOX_CENTER_ANGULAR)

    if targetReached:
        print("Target reachead")
    else:
        print("Target not reached")

    try:
            #Let's put the robot in a random position to start
            go_to_random_state()

            # Init listener node
            listener()

            #relative_move(x_movement, y_movement, 0)
            # MY_ROBOT.open_gripper()
            # rospy.sleep(3)

            print(MY_ROBOT.get_current_pose().pose.position.z - PICK_MOVEMENT_DISTANCE)

            #pick_and_place(MY_ROBOT.get_current_pose().pose.position.z - PICK_MOVEMENT_DISTANCE)
    except KeyboardInterrupt:
        print('interrupted!')

##TODO


# print("Press ENTER to continue")
# raw_input()
# pose_goal = Pose()
# pose_goal.position.x = 0.4
# pose_goal.position.y = 0.1
# pose_goal.position.z = 0.4
# MY_ROBOT.go_to_pose_goal(pose_goal)
# print("The end")