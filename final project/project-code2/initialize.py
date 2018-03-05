import os
import rospy

from gazebo_msgs.srv import (
    SpawnModel,
    DeleteModel,
)
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import (
    Header,
    Empty,
)

from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)

import baxter_interface


class Initialize(object):
    def __init__(self, limb, hover_distance=0.15, verbose=True):
        self._limb_name = limb  # string
        self._hover_distance = hover_distance  # in meters
        self._verbose = verbose  # bool
        self._limb = baxter_interface.Limb(limb)
        self._gripper = baxter_interface.Gripper(limb)
        ns = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
        self._iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
        rospy.wait_for_service(ns, 5.0)
        # verify robot is enabled
        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()

    def move_to_start(self, start_angles1=None):
        print("Moving the {0} arm to start pose...".format(self._limb_name))
        if not start_angles1:
            start_angles1 = dict(zip(self._joint_names, [0] * 7))
        self._guarded_move_to_joint_position(start_angles1)

    def _guarded_move_to_joint_position(self, joint_angles):
        if joint_angles:
            self._limb.move_to_joint_positions(joint_angles)
        else:
            rospy.logerr("No Joint Angles provided for move_to_joint_positions. Staying put.")


def load_gazebo_models(table_pose=Pose(position=Point(x=1.6, y=0.0, z=0), orientation=Quaternion(x=0, y=0, z=1.580587, w=0)),
                       table_reference_frame="world"):
    # Get Models' Path
    model_path = os.path.dirname(os.path.realpath(__file__)) + "/models/"
    # Load Table SDF
    table_xml = ''
    with open(model_path + "table_marble/model.sdf", "r") as table_file:
        table_xml = table_file.read().replace('\n', '')
    # Spawn Table SDF
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_sdf = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_sdf = spawn_sdf("table_marble", table_xml, "/",
                             table_pose, table_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))


def delete_gazebo_models():
    # This will be called on ROS Exit, deleting Gazebo models
    try:
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        resp_delete = delete_model("table_marble")
    except rospy.ServiceException, e:
        rospy.loginfo("Delete Model service call failed: {0}".format(e))


def main():

    # rospy.init_node("ik_")
    # Load Gazebo Models via Spawning Services
    # load_gazebo_models()
    rospy.wait_for_message("/robot/sim/started", Empty)

    limb = 'right'
    hover_distance = 0.15  # meters
    # Starting Joint angles for right arm

    starting_writing_joint_angles_1 = {'right_s0': -0.8837920134231201, 'right_s1': 0.5610749374565493,
                                       'right_w0': 0.09155365297161566, 'right_w1': 0.13208696505478645,
                                       'right_w2': -0.09887268598462917, 'right_e0': -0.263600519484231,
                                       'right_e1': 0.24032704408101324}

    starting_writing_joint_angles_2 = {'right_s0': -0.36460405536661344, 'right_s1': -0.463269273147751, 'right_w0': 0.09155318342126062,
     'right_w1': 0.13216842670663276, 'right_w2': -0.09887269459709458, 'right_e0': -0.26359607590513967,
     'right_e1': 0.45098779259830035}

    starting_writing_joint_angles_3 = {'right_s0': 0.9450636184809138, 'right_s1': -0.7683361928790218,
                                        'right_w0': 0.09079207159148694, 'right_w1': 1.4192452921308698,
                                        'right_w2': 0.5537175713445874, 'right_e0': -0.06775831108183361,
                                        'right_e1': 1.017214395244725}

    drw = Initialize(limb, hover_distance)

    drw.move_to_start(starting_writing_joint_angles_1)

    drw.move_to_start(starting_writing_joint_angles_2)

    drw.move_to_start(starting_writing_joint_angles_3)
    rospy.sleep(1.0)
