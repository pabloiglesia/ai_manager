# coding: utf-8

# import cv2, cv_bridge
import os
import time

import rospy
from PIL import Image as PILImage
from sensor_msgs.msg import Image
# import cv_bridge

"""
This class is used to manage sensor_msgs Images.
"""


class ImageController:
    def __init__(self, path=os.path.dirname(os.path.realpath(__file__)), image_topic='/usb_cam/image_raw'):
        self.ind_saved_images = 0  # Index which will tell us the number of images that have been saved
        # self.bridge = cv_bridge.CvBridge()
        self.success_path = "{}/success".format(path)  # Path where the images are going to be saved
        self.fail_path = "{}/fail".format(path)  # Path where the images are going to be saved
        self.image_topic = image_topic

        # If it does not exist, we create the path folder in our workspace
        try:
            os.stat(self.success_path)
        except:
            os.mkdir(self.success_path)

        # If it does not exist, we create the path folder in our workspace
        try:
            os.stat(self.fail_path)
        except:
            os.mkdir(self.fail_path)

    def get_image(self):
        msg = rospy.wait_for_message(self.image_topic, Image)

        return self.to_pil(msg), msg.width, msg.height

    def record_image(self, img, success):
        # img = self.to_cv2(msg)
        # cv2.imwrite('{}/img{:04d}.png'.format(self.path, self.indImage), img)
        # img = PILImage.fromarray(array_img)
        path = self.success_path if success else self.fail_path  # The path were we want to save the image is
        # different depending on success info

        image_path = '{}/img{}.png'.format(  # Saving image
            path,  # Path
            time.time())  # FIFO queue

        img.save(image_path)

        self.ind_saved_images += 1  # Index increment

    def to_pil(self, msg, display=False):
        # img = self.bridge.imgmsg_to_cv2(msg)
        # if display:
        #     cv2.namedWindow("window", 1)
        #     cv2.imshow("window", self.image)
        #     cv2.waitKey(5)
        size = (msg.width, msg.height)  # Image size
        img = PILImage.frombytes('RGB', size, msg.data)  # sensor_msg to Image
        return img
