#!/usr/bin/env python
"""
Code used to train the UR3 robot to perform a pick and place task using Reinforcement Learning and Image Recognition.
This code does not perform actions directly into the robot, it just posts actions in a ROS topic and
gathers state information from another ROS topic.
"""

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from ImageController import ImageController
import random


def gather_state_info(img_controller):
    """
    This method gather information about the ur3 robot state by reading several ROS topics
    :param img_controller: class which will allow us to save sensor_msgs images
    """

    current_coordinates = rospy.wait_for_message('/tasks/done', Float64MultiArray)  # To know if the previous task
    # has been completed and get the current coordinates of the robot

    msg = rospy.wait_for_message('/usb_cam/image_raw', Image)  # We retrieve state image
    img_controller.record_image(msg)  # We save the image in the replay memory
    # TODO: Gather information about the new state

    return current_coordinates.data


def rl_algorithm(current_coordinates):
    """
    This function implements a Reinforcement Learning algorithm to controll the UR3 robot.
    :return: action taken
    """
    print(current_coordinates)
    # TODO: Create Rl Algorithm (Random action is taken now
    actions = ['north', 'south', 'east', 'west', 'pick']

    if abs(current_coordinates[0]) > 0.14 or abs(current_coordinates[1]) > 0.2:
        print("Estado terminal")
        return 'random_state'
    else:
        print("Estado normal")
        idx = random.randint(0, 4)
        return actions[idx]


def talker(publisher, action):
    """
    Publish the selected action into the ROS topic
    :param action:
    :type publisher: object
    """
    rospy.loginfo(action)  # Log in console
    publisher.publish(action)  # Publish action


def main():
    publisher = rospy.Publisher('/tasks/action', String, queue_size=10)  # Publisher definition
    rospy.init_node('ai_manager', anonymous=True)  # ROS node initialization
    image = ImageController(capacity=10)  # Image controller initialization

    while True:
        current_coordinates = gather_state_info(image)  # Gathers state information
        action = rl_algorithm(current_coordinates)  # Select action and train the algorithm
        talker(publisher, action)  # Publish the action


if __name__ == '__main__':
    try:
        print("Funciona")
        main()
    except rospy.ROSInterruptException:
        pass