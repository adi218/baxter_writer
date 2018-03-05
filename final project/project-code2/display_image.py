import rospy
import numpy as np
import cv2
import cv_bridge
import os

from sensor_msgs.msg import (
    Image,
)


def send_image(num):
    """
   Send the image located at the specified path to the head
   display on Baxter.

   @param path: path to the image file to load and send
   """
    path = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images'), num+'.png')
    rospy.init_node('rsdk_xdisplay_image', anonymous=True)
    img = cv2.imread(path)
    # print(img)
    print(type(img))
    msg = cv_bridge.CvBridge().cv2_to_imgmsg(img, encoding="bgr8")
    pub = rospy.Publisher('/robot/xdisplay', Image, latch=True)
    pub.publish(msg)
    # Sleep to allow for image to be published.
    rospy.sleep(1)
