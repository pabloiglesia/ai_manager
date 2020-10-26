#!/usr/bin/env python
"""
Code used to train the UR3 robot to perform a pick and place task using Reinforcement Learning and Image Recognition.
This code does not perform actions directly into the robot, it just posts actions in a ROS topic and
gathers state information from another ROS topic.
"""

import rospy
import torch

from RLAlgorithm import RLAlgorithm
from Environment import Environment
from ai_manager.srv import GetActions, GetActionsResponse

rospy.init_node('ai_manager', anonymous=True)  # ROS node initialization
# Global Image Controller
RL_ALGORITHM = RLAlgorithm()


def rl_algorithm(current_coordinates, object_gripped):
    """
    This function implements a Reinforcement Learning algorithm to controll the UR3 robot.
    :return: action taken
    """
    previous_state = RL_ALGORITHM.current_state
    previous_action = RL_ALGORITHM.current_action
    previous_action_idx = RL_ALGORITHM.current_action_idx
    RL_ALGORITHM.em.gather_image_state()  # Gathers current state image
    RL_ALGORITHM.current_state = RL_ALGORITHM.State(current_coordinates[0], current_coordinates[1], object_gripped,
                                                    RL_ALGORITHM.em.image_msg)
    reward = RL_ALGORITHM.em.calculate_reward()

    action = RL_ALGORITHM.agent.select_action(RL_ALGORITHM.current_state, RL_ALGORITHM.policy_net)

    if action != 'random_state':
        if RL_ALGORITHM.agent.current_step > 1:

            RL_ALGORITHM.memory.push(
                RL_ALGORITHM.Experience(
                    previous_state.image_raw,
                    torch.tensor([[previous_state.coordinate_x, previous_state.coordinate_y]]),
                    torch.tensor([previous_action_idx], device=RL_ALGORITHM.device),
                    RL_ALGORITHM.current_state.image_raw,
                    torch.tensor([[RL_ALGORITHM.current_state.coordinate_x, RL_ALGORITHM.current_state.coordinate_y]]),
                    torch.tensor([reward], device=RL_ALGORITHM.device)
                ))

            rospy.loginfo("Step: {}, Episode: {}, Previous reward: {}, Previous action: {}".format(
                RL_ALGORITHM.agent.current_step - 1,
                RL_ALGORITHM.episode,
                reward,
                previous_action))

        RL_ALGORITHM.train_net()

    return action


def handle_get_actions(req):
    object_gripped = req.object_gripped
    current_coordinates = [req.x, req.y]
    action = rl_algorithm(current_coordinates, object_gripped)

    # RL_ALGORITHM.plot()

    return GetActionsResponse(action)


def get_actions_server():
    s = rospy.Service('get_actions', GetActions, handle_get_actions)
    rospy.loginfo("Ready to send actions.")
    rospy.spin()


if __name__ == '__main__':
    try:
        get_actions_server()
    except rospy.ROSInterruptException:
        pass
