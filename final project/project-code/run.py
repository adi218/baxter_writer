from draw2 import Draw
import initialize as ini
from headcam import Headcam
import call_solver
import display_image
import sys
import os
import rospy


def main():

    ini.load_gazebo_models()
    raw_input("Press Enter after placing the math equation...")

    cam = Headcam()

    cam.main()
    raw_input("Press Enter after solving the math equation...")
    path = os.path.dirname(os.path.abspath(__file__))
    f = open(os.path.join(os.path.join(path, "recnsolve"),"answer.txt"), "r")
    num = str(f.read()).rstrip()

    display_image.send_image(num)
    ini.main()
    raw_input("Press Enter after baxter reaches initial pose...")
    dr = Draw()
    dr.move(num)
    raw_input("Press Enter to exit...")
    rospy.on_shutdown(ini.delete_gazebo_models)


if __name__ == '__main__':
    main()


