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

#from __future__ import print_function
from AIManager.srv import GetActions, GetActionsResponse

# Global Image Controller
IMAGE_CONTROLLER = ImageController()

def gather_state_info():
    """
    This method gather information about the ur3 robot state by reading several ROS topics
    :param img_controller: class which will allow us to save sensor_msgs images
    """

    # current_coordinates = rospy.wait_for_message('/tasks/done', Float64MultiArray)  # To know if the previous task
    # has been completed and get the current coordinates of the robot

    msg = rospy.wait_for_message('/usb_cam/image_raw', Image)  # We retrieve state image
    IMAGE_CONTROLLER.record_image(msg)  # We save the image in the replay memory
    # TODO: Gather information about the new state

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


# def talker(publisher, action):
#     """
#     Publish the selected action into the ROS topic
#     :param action:
#     :type publisher: object
#     """
#     rospy.loginfo(action)  # Log in console
#     publisher.publish(action)  # Publish action


def main():
    # publisher = rospy.Publisher('/tasks/action', String, queue_size=10)  # Publisher definition
    rospy.init_node('ai_manager', anonymous=True)  # ROS node initialization
    get_actions_server()
    # image = ImageController(capacity=10)  # Image controller initialization

    # while True:

        # action = rl_algorithm(current_coordinates)  # Select action and train the algorithm
        # talker(publisher, action)  # Publish the action

def handle_get_actions(req):
    print("Returning action for coordinates {} and {}".format(req.x, req.y))
    current_coordinates = [req.x,req.y]
    gather_state_info()  # Gathers state information
    action = rl_algorithm(current_coordinates)
    print(action)
    return GetActionsResponse(action)

def get_actions_server():
    #rospy.init_node('get_actions_server')
    s = rospy.Service('get_actions', GetActions, handle_get_actions)
    print("Ready to send actions.")
    rospy.spin()

if __name__ == '__main__':
    try:
        main()

    except rospy.ROSInterruptException:
        pass