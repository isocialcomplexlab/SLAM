import rospy

from sensor_msgs.msg import Image

from cv_bridge import CvBridge

import cv2

class ImageSaver(object):
    def __init__(self):
        self.bridge = CvBridge()
        
    def image_callback(self, msg):
        try:
            # Convert your ROS Image message to OpenCV2
            cv2_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            print(e)
        else:
            cv2.imwrite('images/camera_image.jpeg', cv2_img)
            rospy.sleep(0.25)

if __name__ == '__main__':
    img_s = ImageSaver()
    rospy.init_node('img_saver')

    image_topic = "/camera/rgb/image_raw"

    rospy.Subscriber(image_topic, Image, img_s.image_callback)

    rospy.spin()
