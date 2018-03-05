import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os


bridge = CvBridge()
var = None

class Headcam:

    def image_callback(self, msg):
        print("Received an image!")
        path = os.path.dirname(os.path.abspath(__file__))
        try:
            # Convert your ROS Image message to OpenCV2
            cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError, e:
            print(e)
        else:
            global var
            var.unregister()
            # Save your OpenCV2 image as a jpeg
            cv2.imwrite(os.path.join(os.path.join(path , 'recnsolve'),'camera_image.jpg'), cv2_img)


    def main(self):
        # rospy.init_node('image_listener')
        # Define your image topic

        image_topic = "/cameras/head_camera/image"
        # Set up your subscriber and define its callback
        global var
        var=rospy.Subscriber(image_topic, Image, self.image_callback)

