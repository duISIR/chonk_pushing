#!/usr/bin/env python3

import rospy
import actionlib
from sensor_msgs.msg import Joy
from pushing_msgs.msg import TriggerCmdAction, TriggerCmdGoal
from pushing_msgs.msg import CmdDualPoseAction, CmdDualPoseGoal

class TeleOp2DClient(object):
    """docstring for TeleOp2DClient."""

    def __init__(self, name) -> None:
        # initialization message
        self._name = name
        rospy.loginfo("%s: Initialized action client class.", self._name)
        #################################################################################
        # get parameters
        self._freq = rospy.get_param('~freq', 100)
        self._duration = rospy.get_param('~duration', 2.0)
        self._target_pos_x_R = rospy.get_param('~target_position_x_R', 0.3)
        self._target_pos_y_R = rospy.get_param('~target_position_y_R', -0.2)
        self._target_pos_z_R = rospy.get_param('~target_position_z_R', 0.79)
        self._safety_pos_x_R = rospy.get_param('~safety_position_x_R', 0.3)
        self._safety_pos_y_R = rospy.get_param('~safety_position_y_R', -0.1)
        self._safety_pos_z_R = rospy.get_param('~safety_position_z_R', 0.95)
        self._target_pos_x_L = rospy.get_param('~target_position_x_L', 0.3)
        self._target_pos_y_L = rospy.get_param('~target_position_y_L', 0.2)
        self._target_pos_z_L = rospy.get_param('~target_position_z_L', 0.90)
        self._safety_pos_x_L = rospy.get_param('~safety_position_x_L', 0.3)
        self._safety_pos_y_L = rospy.get_param('~safety_position_y_L', 0.1)
        self._safety_pos_z_L = rospy.get_param('~safety_position_z_L', 0.95)
        # declare joystick substriber
        self._joy_sub = rospy.Subscriber(
            "/spacenav/joy",
            Joy,
            self.read_joy_status_cb
        )
        # declare timer for the State Machine loop
        dur = rospy.Duration(1.0/self._freq)
        self._timer = rospy.Timer(dur, self.timer_cb)
        #################################################################################
        # initialized variables
        self.joyButtonLeft = False
        self.joyButtonRight = False
        self.reachedGoal = False
        # initialize SM states
        self.stateInactive = True
        self.stateTeleOp = False
        self.stateSend2Target = False
        self.stateSend2Safety = False
        # start innactive state
        self.inactive_cb()
        #################################################################################
        # create action client to teleop robot
        self.action_client_teleop = actionlib.SimpleActionClient('/chonk/teleop_2d', TriggerCmdAction)
        # creates goal and sends to the action server
        self.goal_teleop = TriggerCmdGoal(pos_z=self._target_pos_z_R)
        #################################################################################
        # create action client for commanding pose
        self.action_client_cmd_pose = actionlib.SimpleActionClient('/chonk/cmd_pose', CmdDualPoseAction)
        # create goal
        self.goal_cmd_pose = CmdDualPoseGoal()
        self.goal_cmd_pose.poseR.position.x = self._safety_pos_x_R
        self.goal_cmd_pose.poseR.position.y = self._safety_pos_y_R
        self.goal_cmd_pose.poseR.position.z = self._safety_pos_z_R
        self.goal_cmd_pose.poseR.orientation.x = 1
        self.goal_cmd_pose.poseR.orientation.y = 0
        self.goal_cmd_pose.poseR.orientation.z = 0
        self.goal_cmd_pose.poseR.orientation.w = 0
        self.goal_cmd_pose.poseL.position.x = self._safety_pos_x_L
        self.goal_cmd_pose.poseL.position.y = self._safety_pos_y_L
        self.goal_cmd_pose.poseL.position.z = self._safety_pos_z_L
        self.goal_cmd_pose.poseL.orientation.x = 1
        self.goal_cmd_pose.poseL.orientation.y = 0
        self.goal_cmd_pose.poseL.orientation.z = 0
        self.goal_cmd_pose.poseL.orientation.w = 0
        self.goal_cmd_pose.duration = self._duration
        #################################################################################

    def read_joy_status_cb(self, msg):
        options = [False, True]
        self.joyButtonLeft = options[msg.buttons[0]]
        self.joyButtonRight = options[msg.buttons[1]]

    def timer_cb(self, event):
        """ Manage State Machine logic """
        if self.stateInactive and self.joyButtonLeft:
            self.stateInactive = False
            self.stateSend2Target = True
            self.send_cb(
                self._target_pos_x_R, self._target_pos_y_R, self._target_pos_z_R,
                self._target_pos_x_L, self._target_pos_y_L, self._target_pos_z_L,
                'Workspace')
        elif self.stateSend2Target and self.reachedGoal:
            self.stateSend2Target = False
            self.reachedGoal = False
            self.stateTeleOp = True
            self.teleop_cb()
        elif self.stateTeleOp and self.joyButtonRight:
            self.stateTeleOp = False
            self.stateSend2Safety = True
            self.action_client_teleop.cancel_goal()
            self.send_cb(
                self._safety_pos_x_R, self._safety_pos_y_R, self._safety_pos_z_R,
                self._safety_pos_x_L, self._safety_pos_y_L, self._safety_pos_z_L,
                'Safety')
        elif self.stateSend2Safety and self.reachedGoal:
            self.stateSend2Safety = False
            self.reachedGoal = False
            self.stateInactive = True
            self.inactive_cb()
        else:
            pass

    def inactive_cb(self):
        rospy.loginfo('%s: Inactive State. Press the left bottom to start.' % self._name)

    def teleop_cb(self):
        rospy.loginfo('%s: Tele-operation State. Press the right bottom to stop.' % self._name)
        # wait until actionlib server starts
        self.action_client_teleop.wait_for_server()
        # sends the goal to the action server
        self.action_client_teleop.send_goal(self.goal_teleop)

    def send_cb(self, pos_x_R, pos_y_R, pos_z_R, pos_x_L, pos_y_L, pos_z_L, string):
        rospy.loginfo('%s: Sending robot to %s.' % (self._name, string))
        # change possition
        self.goal_cmd_pose.poseR.position.x = pos_x_R
        self.goal_cmd_pose.poseR.position.y = pos_y_R
        self.goal_cmd_pose.poseR.position.z = pos_z_R
        self.goal_cmd_pose.poseL.position.x = pos_x_L
        self.goal_cmd_pose.poseL.position.y = pos_y_L
        self.goal_cmd_pose.poseL.position.z = pos_z_L
        # wait for server
        self.action_client_cmd_pose.wait_for_server()
        # send goal
        self.action_client_cmd_pose.send_goal_and_wait(self.goal_cmd_pose)
        # get result
        result = self.action_client_cmd_pose.get_result()
        self.reachedGoal = result.reached_goal
        

if __name__ == '__main__':
    # Initialize node
    rospy.init_node('teleop_2d_client', anonymous=True)
    # Initialize node class
    cmd_config_client = TeleOp2DClient(rospy.get_name())
    # execute node
    rospy.spin()
