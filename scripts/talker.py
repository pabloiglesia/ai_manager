#!/usr/bin/env python
"""
Code used to train the UR3 robot to perform a pick and place task using Reinforcement Learning and Image Recognition.
This code does not perform actions directly into the robot, it just posts actions in a ROS topic and
gathers state information from another ROS topic.
"""

import rospy
from sensor_msgs.msg import Image

from ImageController import ImageController
from Environment import Environment
from ai_manager.srv import GetActions, GetActionsResponse

import random

# Global Image Controller
IMAGE_CONTROLLER = ImageController()


def gather_state_info():
    """
    This method gather information about the ur3 robot state by reading several ROS topics
    :param img_controller: class which will allow us to save sensor_msgs images
    """
    msg = rospy.wait_for_message('/usb_cam/image_raw', Image)  # We retrieve state image
    IMAGE_CONTROLLER.record_image(msg)  # We save the image in the replay memory
    # TODO: Gather information about the new state


def rl_algorithm(current_coordinates):
    """
    This function implements a Reinforcement Learning algorithm to controll the UR3 robot.
    :return: action taken
    """
    # TODO: Create Rl Algorithm (Random action is taken now
    actions = ['north', 'south', 'east', 'west', 'pick']

    if Environment.is_terminal_state(current_coordinates):
        rospy.loginfo("Terminal state")
        return 'random_state'
    else:
        idx = random.randint(0, 4)
        return actions[idx]


def main():
    # publisher = rospy.Publisher('/tasks/action', String, queue_size=10)  # Publisher definition
    rospy.init_node('ai_manager', anonymous=True)  # ROS node initialization
    get_actions_server()


def handle_get_actions(req):
    current_coordinates = [req.x,req.y]
    gather_state_info()  # Gathers state information
    action = rl_algorithm(current_coordinates)
    rospy.loginfo("Returning action for coordinates {} and {}: {}".format(req.x, req.y, action))
    return GetActionsResponse(action)


def get_actions_server():
    s = rospy.Service('get_actions', GetActions, handle_get_actions)
    rospy.loginfo("Ready to send actions.")
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass