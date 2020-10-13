#!/usr/bin/env python
"""
Code used to train the UR3 robot to perform a pick and place task using Reinforcement Learning and Image Recognition.
This code does not perform actions directly into the robot, it just posts actions in a ROS topic and
gathers state information from another ROS topic.
"""

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from ImageController import ImageController
import random


def gather_state_info(img_controller):
    """
    This method gather information about the ur3 robot state by reading several ROS topics
    :param img_controller: class which will allow us to save sensor_msgs images
    """

    task_done = rospy.wait_for_message('/tasks/done', Bool)  # To know if the previous task has been completed
    if task_done.data:
        print('Task has been completed')

    msg = rospy.wait_for_message('/usb_cam/image_raw', Image)  # We retrieve state image
    img_controller.record_image(msg)  # We save the image in the replay memory
    # TODO: Gather information about the new state


def rl_algorithm():
    """
    This function implements a Reinforcement Learning algorithm to controll the UR3 robot.
    :return: action taken
    """
    # TODO: Create Rl Algorithm (Random action is taken now
    actions = ['north', 'south', 'east', 'west', 'pick']
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
        gather_state_info(image)  # Gathers state information
        action = rl_algorithm()  # Select action and train the algorithm
        talker(publisher, action)  # Publish the action


if __name__ == '__main__':
    try:
        print("Funciona")
        main()
    except rospy.ROSInterruptException:
        pass