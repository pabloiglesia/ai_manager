#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import Float32

from ur_icam_description.robotUR import RobotUR

# Global variable to calibrate distance between the robot and the table
PICK_MOVEMENT_DISTANCE = 0.215

# Global variable for myRobot
MY_ROBOT = RobotUR()

def talker():
    pub = rospy.Publisher('/distance', Float32)
    rospy.init_node('distance_sensor', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        distance = MY_ROBOT.get_current_pose().pose.position.z - PICK_MOVEMENT_DISTANCE
        rospy.loginfo("Measure distance: {}".format(distance))
        pub.publish(distance)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass