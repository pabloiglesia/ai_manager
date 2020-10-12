#!/usr/bin/env python
## Simple talker demo that published std_msgs/Strings messages
## to the 'chatter' topic

import rospy
from std_msgs.msg import String
import random

def gather_state_info():
    print('TODO: Gather information about the new state')


def rl_algorithm():
    print('TODO: Create Rl Algorithm (Random action is taken now)')
    actions = ['north', 'south', 'east', 'west', 'pick']
    idx = random.randint(0, 4)
    return actions[idx]


def talker(publisher, action):

    rospy.loginfo(action)
    publisher.publish(action)


def main():
    publisher = rospy.Publisher('/tasks/action', String, queue_size=10)
    rospy.init_node('ai_manager', anonymous=True)

    while True:
        gather_state_info()
        action = rl_algorithm()
        talker(publisher, action)
        task_done = rospy.wait_for_message('/tasks/done', String)
        if task_done.data == 'True':
            print('Task has been completed')


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
