import sys
import rospy
import moveit_commander
from geometry_msgs.msg import PoseStamped, Pose
from moveit_msgs.msg import OrientationConstraint, Constraints
import digits as dg
import baxter_interface


class Draw:

    def move(self, num):

        moveit_commander.roscpp_initialize(sys.argv)
        # rospy.init_node("moveit_node")


        robot = moveit_commander.RobotCommander()
        scene = moveit_commander.PlanningSceneInterface()
        group = moveit_commander.MoveGroupCommander('right_arm')
        right_arm = baxter_interface.Limb("right")
        right_arm_pose = right_arm.endpoint_pose()
        # print(right_arm_pose)
        initial_orient = [right_arm_pose['orientation'].x, right_arm_pose['orientation'].y,
                          right_arm_pose['orientation'].z, right_arm_pose['orientation'].w]

        dig_class = dg.Digits()
        digit_shapes = dig_class.digit
        num_shape = digit_shapes[str(num)]
        i = 0
        # print(num_shape, num, len(num_shape))
        while i < len(num_shape):
            (x, y) = num_shape[i]
            pose_target = PoseStamped()
            pose_target.header.frame_id = "base"
            # print(pose_target.pose.position)
            right_arm_pose = right_arm.endpoint_pose()
            x = right_arm_pose['position'].x
            y = right_arm_pose['position'].y
            z = right_arm_pose['position'].z
            print(right_arm_pose)
            pose_target.pose.position.x = x + num_shape[i][0] * 0.1
            pose_target.pose.position.y = y + num_shape[i][1] * 0.1
            pose_target.pose.position.z = z
            pose_target.pose.orientation.x = initial_orient[0]
            pose_target.pose.orientation.y = initial_orient[1]
            pose_target.pose.orientation.z = initial_orient[2]
            pose_target.pose.orientation.w = initial_orient[3]
            group.set_pose_target(pose_target)
            group.set_start_state_to_current_state()
            orien_const = OrientationConstraint()
            orien_const.link_name = "right_gripper"
            orien_const.header.frame_id = "base"
            orien_const.orientation.x = initial_orient[0]
            orien_const.orientation.y = initial_orient[1]
            orien_const.orientation.z = initial_orient[2]
            orien_const.orientation.w = initial_orient[3]
            orien_const.absolute_x_axis_tolerance = 0.1
            orien_const.absolute_y_axis_tolerance = 0.1
            orien_const.absolute_z_axis_tolerance = 0.1
            orien_const.weight = 1.0
            consts = Constraints()
            consts.orientation_constraints = [orien_const]
            group.set_path_constraints(consts)
            plan = group.plan()
            group.execute(plan)
            i += 1