import rospy

# baxter_interface - Baxter Python API
import baxter_interface

rospy.init_node('Hello_Baxter')

limb = baxter_interface.Limb('right')

# get the right limb's current joint angles
angles = limb.joint_angles()

# print the current joint angles
print angles
