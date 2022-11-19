#!/usr/bin/env python

#######################################################
#                   Loop Closure System
#  Master's degree in Computer Science at the Federal University of ABC;
#  Title of Work: Loop Closure Detection in Visual SLAM Based on Convolutional Neural Network
#  Student: Fabiana Naomi Iegawa;
#  Advisor: Wagner Tanaka Botelho;
#
#  Description: Gazebo ROS commands to capture and save images during navigation.
#
#  Modification Date: 19/11/2022
#######################################################

import rospy
import roslib
from gazebo_msgs.srv import GetModelState
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import time


bridge = CvBridge()

class Save():
    def __init__(self):
        self.img_number = 0
        image_sub = rospy.Subscriber('front/image_raw', Image, self.callback)

    def callback(self, msg):
        try:
            cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
        else:
            cv2.imwrite("/home/dataset/sequences/v7/{}".format(self.img_number)+".png", cv2_img)
            self.img_number += 1
            time.sleep(4)

if __name__ == '__main__':
    rospy.init_node('image', anonymous=True)
    rate = rospy.Rate(10)
    image_node = Save()
    rospy.spin()
