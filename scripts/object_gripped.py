#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import Bool

import random

def talker():
    pub = rospy.Publisher('object_gripped', Bool)
    rospy.init_node('object_gripped', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        object_gripped = random.random() > 0.7
        rospy.loginfo("Object_gripped: {}".format(object_gripped))
        pub.publish(object_gripped)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass