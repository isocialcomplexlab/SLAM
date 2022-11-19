#!/usr/bin/env python

#######################################################
#                   Loop Closure System
#  Master's degree in Computer Science at the Federal University of ABC;
#  Title of Work: Loop Closure Detection in Visual SLAM Based on Convolutional Neural Network
#  Student: Fabiana Naomi Iegawa;
#  Advisor: Wagner Tanaka Botelho;
#
#  Description: Gazebo ROS commands to navigate with the robot and save its positions.
#
#  Modification Date: 19/11/2022
#######################################################

import rospy
import roslib
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import GetModelState
from nav_msgs.msg import Odometry
import time


rospy.init_node('robot', anonymous=True)
vel_publisher = rospy.Publisher('/twist_marker_server/cmd_vel', Twist, queue_size=10)
rate = rospy.Rate(10)
vel_msg = Twist()


def move(vel,ang):
    if ang == True:
        vel_msg.linear.x = vel
        vel_publisher.publish(vel_msg)
        time.sleep(2)
        vel_msg.linear.x = 0
        vel_publisher.publish(vel_msg)
    else:
        vel_msg.angular.z = vel
        vel_publisher.publish(vel_msg)
        time.sleep(1)
        vel_msg.angular.z = 0
        vel_publisher.publish(vel_msg)
    return

def main():
    while not rospy.is_shutdown():
        vel_command = raw_input('comando: ')

        if vel_command == 'w':
            move(1.5,True)
        elif vel_command == 's':
            move(-1.5,True)
        elif vel_command == 'a':
            move(1.0,False)
        elif vel_command == 'd':
            move(-1.0,False)
        a = rospy.wait_for_message('/jackal_velocity_controller/odom', Odometry, timeout=10000)
        with open("/dataset/poses/office_gazebo_v7.txt", 'a') as f:
            f.write(str(a.pose.pose)+'\n')

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException as e:
        raise
