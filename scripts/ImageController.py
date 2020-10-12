# coding: utf-8

import cv2, cv_bridge
import os

class ImageController:
    def __init__(self, path=os.path.dirname(os.path.realpath(__file__))):
        self.indImage = 0
        self.bridge = cv_bridge.CvBridge()
        self.path = "{}/images".format(path)

    def record_image(self, msg):
        # try:
        img = self.to_cv2(msg)
        cv2.imwrite('{}/img{:04d}.png'.format(self.path, self.indImage), img)
        self.indImage += 1
        #     return True
        # except:
        #     return False

    def to_cv2(self, msg, display=False):
        img = self.bridge.imgmsg_to_cv2(msg)
        if display:
            cv2.namedWindow("window", 1)
            cv2.imshow("window", self.image)
            cv2.waitKey(5)
        return img