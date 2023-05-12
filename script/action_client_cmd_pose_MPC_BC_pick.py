#! /usr/bin/env python3
import argparse

import rospy
import actionlib
from pushing_msgs.msg import CmdChonkPoseAction, CmdChonkPoseGoal
import numpy as np
import optas

class CmdPoseClient(object):
    """docstring for CmdPoseClient."""

    def __init__(self, name, client, target_pos_Donkey, target_quat_Donkey, target_pos_R, target_quat_R, target_pos_L, target_quat_L, duration) -> None:
        # initialization message
        self._name = name
        self._target_pos_Donkey = target_pos_Donkey
        self._target_quat_Donkey = target_quat_Donkey
        self._target_pos_R = target_pos_R
        self._target_quat_R = target_quat_R
        self._target_pos_L = target_pos_L
        self._target_quat_L = target_quat_L
        self._duration = duration
        rospy.loginfo("%s: Initialized action client class.", self._name)
        # create actionlib client
        self._action_client = client
#        self._action_client = actionlib.SimpleActionClient('/chonk/cmd_pose', CmdChonkPoseAction)
        # wait until actionlib server starts
        rospy.loginfo("%s: Waiting for action server to start.", self._name)
        self._action_client.wait_for_server()
        rospy.loginfo("%s: Action server started, sending goal.", self._name)
        # creates goal and sends to the action server
        goal = CmdChonkPoseGoal()
        goal.poseDonkey.position.x = self._target_pos_Donkey[0]
        goal.poseDonkey.position.y = self._target_pos_Donkey[1]
        goal.poseDonkey.position.z = self._target_pos_Donkey[2]
        goal.poseDonkey.orientation.x = self._target_quat_Donkey[0]
        goal.poseDonkey.orientation.y = self._target_quat_Donkey[1]
        goal.poseDonkey.orientation.z = self._target_quat_Donkey[2]
        goal.poseDonkey.orientation.w = self._target_quat_Donkey[3]
        goal.poseR.position.x = self._target_pos_R[0]
        goal.poseR.position.y = self._target_pos_R[1]
        goal.poseR.position.z = self._target_pos_R[2]
        goal.poseR.orientation.x = self._target_quat_R[0]
        goal.poseR.orientation.y = self._target_quat_R[1]
        goal.poseR.orientation.z = self._target_quat_R[2]
        goal.poseR.orientation.w = self._target_quat_R[3]
        goal.poseL.position.x = self._target_pos_L[0]
        goal.poseL.position.y = self._target_pos_L[1]
        goal.poseL.position.z = self._target_pos_L[2]
        goal.poseL.orientation.x = self._target_quat_L[0]
        goal.poseL.orientation.y = self._target_quat_L[1]
        goal.poseL.orientation.z = self._target_quat_L[2]
        goal.poseL.orientation.w = self._target_quat_L[3]
        goal.duration = self._duration
        # sends the goal to the action server
        rospy.loginfo("%s: Send goal request to action server.", self._name)
        self._action_client.send_goal(
            goal,
#            done_cb=self.done_cb,
#            active_cb=self.active_cb,
            feedback_cb=self.feedback_cb
        )
        # wait for the server to finish the action
        self._action_client.wait_for_result()
        rospy.loginfo("%s: Got result from action server.", self._name)


    def done_cb(self, state, result):
        rospy.loginfo("%s: Action completed with result %r" % (self._name, result.reached_goal))
        rospy.signal_shutdown("Client request completed.")

    def active_cb(self):
        rospy.loginfo("%s: Goal went active!", self._name)

    def feedback_cb(self, feedback):
        rospy.loginfo("%s: %.1f%% to completion." % (self._name, feedback.progress))

if __name__ == '__main__':
    # parse arguments from terminal
    parser = argparse.ArgumentParser(description='Client node to command robot base and end-effector pose.')
    # parse donkey arguments
    Donkey_position = [0, 0., 0.]
    parser.add_argument('--target_position_Donkey', nargs=3,
        help="Give target position of the robot in meters.",
        type=float, default=Donkey_position,
        metavar=('POS_X', 'POS_Y', 'POS_Z')
    )
    Donkey_angle = optas.deg2rad(0.)
    Donkey_quat_w = np.cos(Donkey_angle/2)
    Donkey_quat_x = 0
    Donkey_quat_y = 0
    Donkey_quat_z = np.sin(Donkey_angle/2)
    Donkey_quaternion = optas.spatialmath.Quaternion(Donkey_quat_x, Donkey_quat_y, Donkey_quat_z, Donkey_quat_w)
    parser.add_argument('--target_orientation_Donkey', nargs=4,
        help="Give target orientation as a quaternion.",
        type=float, default=[Donkey_quat_x, Donkey_quat_y, Donkey_quat_z, Donkey_quat_z],
        metavar=('QUAT_X','QUAT_Y','QUAT_Z','QUAT_W')
    )

    parser.add_argument('--target_position_R', nargs=3,
        help="Give target position of the robot in meters.",
        type=float, default=[2, 2-0.25, 1.15],
        metavar=('POS_X', 'POS_Y', 'POS_Z')
    )
    ori_R = optas.spatialmath.Quaternion.fromrpy([np.pi+np.pi/2,    np.pi/2 - np.pi/3,    0]).getquat()
    parser.add_argument('--target_orientation_R', nargs=4,
        help="Give target orientation as a quaternion.",
        type=float, default=[ori_R[0], ori_R[1], ori_R[2], ori_R[3]],
        metavar=('QUAT_X','QUAT_Y','QUAT_Z','QUAT_W')
    )
    # parse left arm arguments
    parser.add_argument('--target_position_L', nargs=3,
        help="Give target position of the robot in meters.",
        type=float, default=[2, 2+0.25, 1.15],
        metavar=('POS_X', 'POS_Y', 'POS_Z')
    )

    ori_L = optas.spatialmath.Quaternion.fromrpy([np.pi-np.pi/2,    np.pi/2 - np.pi/3,    0]).getquat()
    parser.add_argument('--target_orientation_L', nargs=4,
        help="Give target orientation as a quaternion.",
        type=float, default=[ori_L[0], ori_L[1], ori_L[2], ori_L[3]],
        metavar=('QUAT_X','QUAT_Y','QUAT_Z','QUAT_W')
    )

    parser.add_argument("--duration", help="Give duration of motion in seconds.", type=float, default=4.0)
    args = vars(parser.parse_args())


    # Initialize node
    rospy.init_node('cmd_pose_client_MPC_BC_pick', anonymous=True)
    client = actionlib.SimpleActionClient('/chonk/cmd_pose', CmdChonkPoseAction)


    # Initialize node class
    cmd_pose_client = CmdPoseClient('client', client,
        args['target_position_Donkey'],
        args['target_orientation_Donkey'],
        args['target_position_R'],
        args['target_orientation_R'],
        args['target_position_L'],
        args['target_orientation_L'],
        args['duration']
    )
    # execute node
#    rospy.spin()

    args['target_position_R'] = [3.8, 2-0.25, 1.15]
    args['target_position_L'] = [3.8, 2+0.25, 1.15]

    # Initialize node class
    cmd_pose_client = CmdPoseClient('client', client,
        args['target_position_Donkey'],
        args['target_orientation_Donkey'],
        args['target_position_R'],
        args['target_orientation_R'],
        args['target_position_L'],
        args['target_orientation_L'],
        args['duration']
    )

    args['target_position_R'] = [3.8, 2-0.148, 1.15]
    args['target_position_L'] = [3.8, 2+0.148, 1.15]

    # Initialize node class
    cmd_pose_client = CmdPoseClient('client', client,
        args['target_position_Donkey'],
        args['target_orientation_Donkey'],
        args['target_position_R'],
        args['target_orientation_R'],
        args['target_position_L'],
        args['target_orientation_L'],
        args['duration']
    )

    args['target_position_R'] = [3.8, 2-0.148, 1.25]
    args['target_position_L'] = [3.8, 2+0.148, 1.25]

    # Initialize node class
    cmd_pose_client = CmdPoseClient('client', client,
        args['target_position_Donkey'],
        args['target_orientation_Donkey'],
        args['target_position_R'],
        args['target_orientation_R'],
        args['target_position_L'],
        args['target_orientation_L'],
        args['duration']
    )

    args['target_position_R'] = [3.8, -2-0.148, 1.25]
    args['target_position_L'] = [3.8, -2+0.148, 1.25]

    # Initialize node class
    cmd_pose_client = CmdPoseClient('client', client,
        args['target_position_Donkey'],
        args['target_orientation_Donkey'],
        args['target_position_R'],
        args['target_orientation_R'],
        args['target_position_L'],
        args['target_orientation_L'],
        args['duration']
    )

    args['target_position_R'] = [3.8, -2-0.148, 1.15]
    args['target_position_L'] = [3.8, -2+0.148, 1.15]

    # Initialize node class
    cmd_pose_client = CmdPoseClient('client', client,
        args['target_position_Donkey'],
        args['target_orientation_Donkey'],
        args['target_position_R'],
        args['target_orientation_R'],
        args['target_position_L'],
        args['target_orientation_L'],
        args['duration']
    )

    args['target_position_R'] = [3.8, -2-0.24, 1.15]
    args['target_position_L'] = [3.8, -2+0.24, 1.15]

    # Initialize node class
    cmd_pose_client = CmdPoseClient('client', client,
        args['target_position_Donkey'],
        args['target_orientation_Donkey'],
        args['target_position_R'],
        args['target_orientation_R'],
        args['target_position_L'],
        args['target_orientation_L'],
        args['duration']
    )

    args['target_position_R'] = [2, -2-0.24, 1.15]
    args['target_position_L'] = [2, -2+0.24, 1.15]

    # Initialize node class
    cmd_pose_client = CmdPoseClient('client', client,
        args['target_position_Donkey'],
        args['target_orientation_Donkey'],
        args['target_position_R'],
        args['target_orientation_R'],
        args['target_position_L'],
        args['target_orientation_L'],
        args['duration']
    )

    args['target_position_R'] = [0.865, -0.12, 0.92]
    ori_R = optas.spatialmath.Quaternion.fromrpy([0,    -np.pi/2,    0]).getquat()
    args['target_orientation_R'] = [ori_R[0], ori_R[1], ori_R[2], ori_R[3]]
    args['target_position_L'] = [0.865, 0.12, 0.92]
    ori_L = optas.spatialmath.Quaternion.fromrpy([0,    -np.pi/2,    0]).getquat()
    args['target_orientation_L'] = [ori_L[0], ori_L[1], ori_L[2], ori_L[3]]

    # Initialize node class
    cmd_pose_client = CmdPoseClient('client', client,
        args['target_position_Donkey'],
        args['target_orientation_Donkey'],
        args['target_position_R'],
        args['target_orientation_R'],
        args['target_position_L'],
        args['target_orientation_L'],
        args['duration']
    )


    # execute node
#    rospy.spin()
