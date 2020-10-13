# coding: utf-8

# import cv2, cv_bridge
import os
from PIL import Image

"""
This class is used to manage sensor_msgs Images.
"""


class ImageController:
    def __init__(self, path=os.path.dirname(os.path.realpath(__file__)), capacity=64):
        self.ind_saved_images = 0  # Index which will tell us the number of images that have been saved
        # self.bridge = cv_brdgie.CvBridge()
        self.path = "{}/images".format(path)  # Path where the images are going to be saved
        self.capacity = capacity  # Number of images that we want to be stored. It will work as a FIFO queue

    def record_image(self, msg):
        # img = self.to_cv2(msg)
        # cv2.imwrite('{}/img{:04d}.png'.format(self.path, self.indImage), img)

        size = (msg.width,msg.height)  # Image size
        img = Image.frombytes('RGB', size, msg.data)  # sensor_msg to Image
        img.save('{}/img{:04d}.png'.format(  # Saving image
            self.path,  # Path
            self.ind_saved_images % self.capacity)  # FIFO queue
        )

        self.ind_saved_images += 1  # Index increment

    # def to_cv2(self, msg, display=False):
    #     img = self.bridge.imgmsg_to_cv2(msg)
    #     if display:
    #         cv2.namedWindow("window", 1)
    #         cv2.imshow("window", self.image)
    #         cv2.waitKey(5)
    #     return img
