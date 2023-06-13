#! /usr/bin/env python3
import argparse

import rospy
import actionlib
from pushing_msgs.msg import CmdChonkPoseForceAction, CmdChonkPoseForceGoal
import numpy as np
import optas
import tf

import threading



class CmdPoseClient(object):
    """docstring for CmdPoseClient."""

    def __init__(self, name, client1, client2, m_box, target_pos_Donkey, target_quat_Donkey, target_pos_R, target_quat_R, target_force_R, target_torque_R,
                 target_pos_L, target_quat_L, target_force_L, target_torque_L, duration) -> None:
        # initialization message
        self._name = name
        self._m_box = m_box
        self._target_pos_Donkey = target_pos_Donkey
        self._target_quat_Donkey = target_quat_Donkey
        self._target_pos_R = target_pos_R
        self._target_quat_R = target_quat_R
        self._target_force_R = target_force_R
        self._target_torque_R = target_torque_R
        self._target_pos_L = target_pos_L
        self._target_quat_L = target_quat_L
        self._target_force_L = target_force_L
        self._target_torque_L = target_torque_L
        self._duration = duration
        rospy.loginfo("%s: Initialized action client class.", self._name)
        # create actionlib client
        self._action_client1 = client1
        self._action_client2 = client2

#        self._action_client = actionlib.SimpleActionClient('/chonk/cmd_pose', CmdChonkPoseAction)
        # wait until actionlib server starts
        rospy.loginfo("%s: Waiting for action server to start.", self._name)
        self._action_client1.wait_for_server()
        self._action_client2.wait_for_server()

        rospy.loginfo("%s: Action server started, sending goal.", self._name)
        # creates goal and sends to the action server
        self.goal = CmdChonkPoseForceGoal()
        self.goal.m_box = self._m_box
        self.goal.poseDonkey.position.x = self._target_pos_Donkey[0]
        self.goal.poseDonkey.position.y = self._target_pos_Donkey[1]
        self.goal.poseDonkey.position.z = self._target_pos_Donkey[2]
        self.goal.poseDonkey.orientation.x = self._target_quat_Donkey[0]
        self.goal.poseDonkey.orientation.y = self._target_quat_Donkey[1]
        self.goal.poseDonkey.orientation.z = self._target_quat_Donkey[2]
        self.goal.poseDonkey.orientation.w = self._target_quat_Donkey[3]
        self.goal.poseR.position.x = self._target_pos_R[0]
        self.goal.poseR.position.y = self._target_pos_R[1]
        self.goal.poseR.position.z = self._target_pos_R[2]
        self.goal.poseR.orientation.x = self._target_quat_R[0]
        self.goal.poseR.orientation.y = self._target_quat_R[1]
        self.goal.poseR.orientation.z = self._target_quat_R[2]
        self.goal.poseR.orientation.w = self._target_quat_R[3]
        self.goal.ForceTorqueR.force.x = self._target_force_R[0]
        self.goal.ForceTorqueR.force.y = self._target_force_R[1]
        self.goal.ForceTorqueR.force.z = self._target_force_R[2]
        self.goal.ForceTorqueR.torque.x = self._target_torque_R[0]
        self.goal.ForceTorqueR.torque.y = self._target_torque_R[1]
        self.goal.ForceTorqueR.torque.z = self._target_torque_R[2]
        self.goal.poseL.position.x = self._target_pos_L[0]
        self.goal.poseL.position.y = self._target_pos_L[1]
        self.goal.poseL.position.z = self._target_pos_L[2]
        self.goal.poseL.orientation.x = self._target_quat_L[0]
        self.goal.poseL.orientation.y = self._target_quat_L[1]
        self.goal.poseL.orientation.z = self._target_quat_L[2]
        self.goal.poseL.orientation.w = self._target_quat_L[3]
        self.goal.ForceTorqueL.force.x = self._target_force_L[0]
        self.goal.ForceTorqueL.force.y = self._target_force_L[1]
        self.goal.ForceTorqueL.force.z = self._target_force_L[2]
        self.goal.ForceTorqueL.torque.x = self._target_torque_L[0]
        self.goal.ForceTorqueL.torque.y = self._target_torque_L[1]
        self.goal.ForceTorqueL.torque.z = self._target_torque_L[2]
        self.goal.duration = self._duration
        # sends the goal to the action server
        rospy.loginfo("%s: Send goal request to action server.", self._name)

        # Create threads for sending goals
        thread1 = threading.Thread(target=self.action_client1)
        thread2 = threading.Thread(target=self.action_client2)
        # Start the threads to send goals simultaneously
        thread1.start()
        thread2.start()

        # Wait for both threads to complete
        thread1.join()
        thread2.join()

    def action_client1(self):
        self._action_client1.send_goal(
            self.goal,
#            done_cb=self.done_cb1,
#            active_cb=self.active_cb1,
            feedback_cb=self.feedback_cb1
        )
        # wait for the server to finish the action
        self._action_client1.wait_for_result()
        rospy.loginfo("%s: Got result from action server.", 'client_1')


    def action_client2(self):
        self._action_client2.send_goal(
            self.goal,
#            done_cb=self.done_cb2,
#            active_cb=self.active_cb2,
            feedback_cb=self.feedback_cb2
        )
        # wait for the server to finish the action
        self._action_client2.wait_for_result()
        rospy.loginfo("%s: Got result from action server.", 'client_2')

    def done_cb1(self, state, result):
        rospy.loginfo("%s: Action completed with result %r" % ('client_1', result.reached_goal))
        rospy.signal_shutdown("Client request completed.")

    def active_cb1(self):
        rospy.loginfo("%s: Goal went active!", 'client_1')

    def feedback_cb1(self, feedback):
        rospy.loginfo("%s: %.1f%% to completion." % ('client_1', feedback.progress))
        pass

    def done_cb2(self, state, result):
        rospy.loginfo("%s: Action completed with result %r" % ('client_2', result.reached_goal))
        rospy.signal_shutdown("Client request completed.")

    def active_cb2(self):
        rospy.loginfo("%s: Goal went active!", 'client_2')

    def feedback_cb2(self, feedback):
        rospy.loginfo("%s: %.1f%% to completion." % ('client_2', feedback.progress))
        pass

if __name__ == '__main__':
    # Initialize node
    rospy.init_node('cmd_pose_client_MPC_BC_operational_pick', anonymous=True)

#    tf_listener = tf.TransformListener()
#    tf_listener.waitForTransform('/vicon/world', '/vicon/eva_box/eva_box', rospy.Time(), rospy.Duration(4.0))
#    # read current robot joint positions
#    try:
#        (trans_box,rot_box) = tf_listener.lookupTransform('/vicon/world', '/vicon/eva_box/eva_box',  rospy.Time(0))
#        trans_box[2] += 0.2
#        print(trans_box)
#    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
#        print("error: cannot find vicon data!!!!")
    trans_box = np.zeros(3)
    trans_box[0] = 2.3
    trans_box[1] = 2.67
    trans_box[2] = 1.165
    # parse arguments from terminal
    parser = argparse.ArgumentParser(description='Client node to command robot base and end-effector pose.')
    # parse donkey arguments
    parser.add_argument("--m_box", help="Give box mass.", type=float, default=0.0)

    Donkey_position = [0, 0., 0.]
    parser.add_argument('--target_position_Donkey', nargs=3,
        help="Give target position of the robot in meters.",
        type=float, default=Donkey_position,
        metavar=('POS_X', 'POS_Y', 'POS_Z')
    )
    Donkey_angle = optas.deg2rad(0.)
    Donkey_quat_w = np.cos(0.)
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
        type=float, default=[trans_box[0]+0.22, trans_box[1]-0.7, trans_box[2]],
        metavar=('POS_X', 'POS_Y', 'POS_Z')
    )
    ori_R = optas.spatialmath.Quaternion.fromrpy([np.pi+np.pi/2,    np.pi/2 ,    np.pi/2]).getquat()
    parser.add_argument('--target_orientation_R', nargs=4,
        help="Give target orientation as a quaternion.",
        type=float, default=[ori_R[0], ori_R[1], ori_R[2], ori_R[3]],
        metavar=('QUAT_X','QUAT_Y','QUAT_Z','QUAT_W')
    )
    parser.add_argument('--target_force_R', nargs=3,
        help="Give target position of the robot in meters.",
        type=float, default=[0, 0, 0],
        metavar=('force_X', 'force_Y', 'force_Z')
    )
    parser.add_argument('--target_torque_R', nargs=3,
        help="Give target orientation as a quaternion.",
        type=float, default=[0, 0, 0],
        metavar=('torque_X','torque_Y','torque_Z')
    )
    # parse left arm arguments
    parser.add_argument('--target_position_L', nargs=3,
        help="Give target position of the robot in meters.",
        type=float, default=[trans_box[0]-0.22, trans_box[1]-0.7, trans_box[2]],
        metavar=('POS_X', 'POS_Y', 'POS_Z')
    )

    ori_L = optas.spatialmath.Quaternion.fromrpy([np.pi-np.pi/2,    np.pi/2 ,    np.pi/2]).getquat()
    parser.add_argument('--target_orientation_L', nargs=4,
        help="Give target orientation as a quaternion.",
        type=float, default=[ori_L[0], ori_L[1], ori_L[2], ori_L[3]],
        metavar=('QUAT_X','QUAT_Y','QUAT_Z','QUAT_W')
    )
    parser.add_argument('--target_force_L', nargs=3,
        help="Give target position of the robot in meters.",
        type=float, default=[0, 0, 0],
        metavar=('force_X', 'force_Y', 'force_Z')
    )
    parser.add_argument('--target_torque_L', nargs=3,
        help="Give target orientation as a quaternion.",
        type=float, default=[0, 0, 0],
        metavar=('torque_X','torque_Y','torque_Z')
    )

    parser.add_argument("--duration", help="Give duration of motion in seconds.", type=float, default=8.0)
    args = vars(parser.parse_args())

#    client = actionlib.SimpleActionClient('/chonk/cmd_pose', CmdChonkPoseForceAction)
    client1 = actionlib.SimpleActionClient('/chonk/cmd_pose', CmdChonkPoseForceAction)
    client2 = actionlib.SimpleActionClient('/chonk/cmd_force', CmdChonkPoseForceAction)


    force = 20
    m_box = 1.2

    # Initialize node class
    args['duration']=10
    cmd_pose_client = CmdPoseClient('client', client1, client2,
        args['m_box'],
        args['target_position_Donkey'],
        args['target_orientation_Donkey'],
        args['target_position_R'],
        args['target_orientation_R'],
        args['target_force_R'],
        args['target_torque_R'],
        args['target_position_L'],
        args['target_orientation_L'],
        args['target_force_L'],
        args['target_torque_L'],
        args['duration']
    )
    # execute node
#    rospy.spin()

    ori_R = optas.spatialmath.Quaternion.fromrpy([np.pi+np.pi/2,    np.pi/2 - np.pi/3,    np.pi/2]).getquat()
    args['target_orientation_R'] = [ori_R[0], ori_R[1], ori_R[2], ori_R[3]]
    ori_L = optas.spatialmath.Quaternion.fromrpy([np.pi-np.pi/2,    np.pi/2 - np.pi/3,    np.pi/2]).getquat()
    args['target_orientation_L'] = [ori_L[0], ori_L[1], ori_L[2], ori_L[3]]

    args['target_position_R'] = [trans_box[0]+0.22, trans_box[1], trans_box[2]]
    args['target_position_L'] = [trans_box[0]-0.22, trans_box[1], trans_box[2]]

    # Initialize node class
    args['duration']=8.0
    cmd_pose_client = CmdPoseClient('client', client1, client2,
        args['m_box'],
        args['target_position_Donkey'],
        args['target_orientation_Donkey'],
        args['target_position_R'],
        args['target_orientation_R'],
        args['target_force_R'],
        args['target_torque_R'],
        args['target_position_L'],
        args['target_orientation_L'],
        args['target_force_L'],
        args['target_torque_L'],
        args['duration']
    )
    args['target_position_R'] = [trans_box[0]+0.145, trans_box[1], trans_box[2]]
    args['target_position_L'] = [trans_box[0]-0.145, trans_box[1], trans_box[2]]

    # Initialize node class
    args['duration']=5.0
    cmd_pose_client = CmdPoseClient('client', client1, client2,
        args['m_box'],
        args['target_position_Donkey'],
        args['target_orientation_Donkey'],
        args['target_position_R'],
        args['target_orientation_R'],
        args['target_force_R'],
        args['target_torque_R'],
        args['target_position_L'],
        args['target_orientation_L'],
        args['target_force_L'],
        args['target_torque_L'],
        args['duration']
    )

    args['m_box'] = m_box       # unit kg
    args['target_force_R'] = [0, 0, force]
    args['target_force_L'] = [0, 0, force]
    args['target_position_R'] = [trans_box[0]+0.145, trans_box[1], trans_box[2]]
    args['target_position_L'] = [trans_box[0]-0.145, trans_box[1], trans_box[2]]

    # Initialize node class
    args['duration']=5.0
    cmd_pose_client = CmdPoseClient('client', client1, client2,
        args['m_box'],
        args['target_position_Donkey'],
        args['target_orientation_Donkey'],
        args['target_position_R'],
        args['target_orientation_R'],
        args['target_force_R'],
        args['target_torque_R'],
        args['target_position_L'],
        args['target_orientation_L'],
        args['target_force_L'],
        args['target_torque_L'],
        args['duration']
    )

    args['target_force_R'] = [0, 0, force]
    args['target_force_L'] = [0, 0, force]
    args['target_position_R'] = [trans_box[0]+0.145, trans_box[1], trans_box[2]+0.15]
    args['target_position_L'] = [trans_box[0]-0.145, trans_box[1], trans_box[2]+0.15]

    # Initialize node class
    args['duration']=5.0
    cmd_pose_client = CmdPoseClient('client', client1, client2,
        args['m_box'],
        args['target_position_Donkey'],
        args['target_orientation_Donkey'],
        args['target_position_R'],
        args['target_orientation_R'],
        args['target_force_R'],
        args['target_torque_R'],
        args['target_position_L'],
        args['target_orientation_L'],
        args['target_force_L'],
        args['target_torque_L'],
        args['duration']
    )

#    args['target_force_R'] = [0, 0, force]
#    args['target_force_L'] = [0, 0, force]
#    args['target_position_R'] = [2.3, 1.7-0.145, trans_box[2]+0.15]
#    args['target_position_L'] = [2.3, 1.7+0.145, trans_box[2]+0.15]
#    ori_R = optas.spatialmath.Quaternion.fromrpy([np.pi+np.pi/2,    np.pi/2 - np.pi/3,    0]).getquat()
#    args['target_orientation_R'] = [ori_R[0], ori_R[1], ori_R[2], ori_R[3]]
#    ori_L = optas.spatialmath.Quaternion.fromrpy([np.pi-np.pi/2,    np.pi/2 - np.pi/3,    0]).getquat()
#    args['target_orientation_L'] = [ori_L[0], ori_L[1], ori_L[2], ori_L[3]]

#    # Initialize node class
#    args['duration']=4
#    cmd_pose_client = CmdPoseClient('client', client1, client2,
#        args['m_box'],
#        args['target_position_Donkey'],
#        args['target_orientation_Donkey'],
#        args['target_position_R'],
#        args['target_orientation_R'],
#        args['target_force_R'],
#        args['target_torque_R'],
#        args['target_position_L'],
#        args['target_orientation_L'],
#        args['target_force_L'],
#        args['target_torque_L'],
#        args['duration']
#    )

    args['target_force_R'] = [0, 0, force]
    args['target_force_L'] = [0, 0, force]
    args['target_position_R'] = [5.5, -0.145, trans_box[2]+0.15]
    args['target_position_L'] = [5.5, +0.145, trans_box[2]+0.15]
    ori_R = optas.spatialmath.Quaternion.fromrpy([np.pi+np.pi/2,    np.pi/2 - np.pi/3,    0]).getquat()
    args['target_orientation_R'] = [ori_R[0], ori_R[1], ori_R[2], ori_R[3]]
    ori_L = optas.spatialmath.Quaternion.fromrpy([np.pi-np.pi/2,    np.pi/2 - np.pi/3,    0]).getquat()
    args['target_orientation_L'] = [ori_L[0], ori_L[1], ori_L[2], ori_L[3]]

    # Initialize node class
    args['duration']=12
    cmd_pose_client = CmdPoseClient('client', client1, client2,
        args['m_box'],
        args['target_position_Donkey'],
        args['target_orientation_Donkey'],
        args['target_position_R'],
        args['target_orientation_R'],
        args['target_force_R'],
        args['target_torque_R'],
        args['target_position_L'],
        args['target_orientation_L'],
        args['target_force_L'],
        args['target_torque_L'],
        args['duration']
    )

    args['target_force_R'] = [0, 0, force]
    args['target_force_L'] = [0, 0, force]
    args['target_position_R'] = [6.18, -0.145, trans_box[2]+0.15]
    args['target_position_L'] = [6.18, +0.145, trans_box[2]+0.15]
    args['duration']=3
    cmd_pose_client = CmdPoseClient('client', client1, client2,
        args['m_box'],
        args['target_position_Donkey'],
        args['target_orientation_Donkey'],
        args['target_position_R'],
        args['target_orientation_R'],
        args['target_force_R'],
        args['target_torque_R'],
        args['target_position_L'],
        args['target_orientation_L'],
        args['target_force_L'],
        args['target_torque_L'],
        args['duration']
    )

    args['target_force_R'] = [0, 0, force]
    args['target_force_L'] = [0, 0, force]
    args['target_position_R'] = [6.18, -0.145, trans_box[2]]
    args['target_position_L'] = [6.18, +0.145, trans_box[2]]
    # Initialize node class
    args['duration']=5
    cmd_pose_client = CmdPoseClient('client', client1, client2,
        args['m_box'],
        args['target_position_Donkey'],
        args['target_orientation_Donkey'],
        args['target_position_R'],
        args['target_orientation_R'],
        args['target_force_R'],
        args['target_torque_R'],
        args['target_position_L'],
        args['target_orientation_L'],
        args['target_force_L'],
        args['target_torque_L'],
        args['duration']
    )

    args['m_box'] = 0
    args['target_force_R'] = [0, 0, 0]
    args['target_force_L'] = [0, 0, 0]
    args['target_position_R'] = [6.18, -0.145, trans_box[2]]
    args['target_position_L'] = [6.18, +0.145, trans_box[2]]
    # Initialize node class
    args['duration']=5
    cmd_pose_client = CmdPoseClient('client', client1, client2,
        args['m_box'],
        args['target_position_Donkey'],
        args['target_orientation_Donkey'],
        args['target_position_R'],
        args['target_orientation_R'],
        args['target_force_R'],
        args['target_torque_R'],
        args['target_position_L'],
        args['target_orientation_L'],
        args['target_force_L'],
        args['target_torque_L'],
        args['duration']
    )

    args['target_force_R'] = [0, 0, 0]
    args['target_force_L'] = [0, 0, 0]
    args['target_position_R'] = [6.18, -0.25, trans_box[2]]
    args['target_position_L'] = [6.18, +0.25, trans_box[2]]
    # Initialize node class
    args['duration']=5
    cmd_pose_client = CmdPoseClient('client', client1, client2,
        args['m_box'],
        args['target_position_Donkey'],
        args['target_orientation_Donkey'],
        args['target_position_R'],
        args['target_orientation_R'],
        args['target_force_R'],
        args['target_torque_R'],
        args['target_position_L'],
        args['target_orientation_L'],
        args['target_force_L'],
        args['target_torque_L'],
        args['duration']
    )

    args['target_force_R'] = [0, 0, 0]
    args['target_force_L'] = [0, 0, 0]
    args['target_position_R'] = [5, -0.25, trans_box[2]]
    args['target_position_L'] = [5, +0.25, trans_box[2]]
    # Initialize node class
    args['duration']=5
    cmd_pose_client = CmdPoseClient('client', client1, client2,
        args['m_box'],
        args['target_position_Donkey'],
        args['target_orientation_Donkey'],
        args['target_position_R'],
        args['target_orientation_R'],
        args['target_force_R'],
        args['target_torque_R'],
        args['target_position_L'],
        args['target_orientation_L'],
        args['target_force_L'],
        args['target_torque_L'],
        args['duration']
    )



    args['target_position_R'] = [0.865, -0.12, 0.92]
    ori_R = optas.spatialmath.Quaternion.fromrpy([np.pi+np.pi/2,    np.pi/2,    0]).getquat()
    args['target_orientation_R'] = [ori_R[0], ori_R[1], ori_R[2], ori_R[3]]
    args['target_position_L'] = [0.865, 0.12, 0.92]
    ori_L = optas.spatialmath.Quaternion.fromrpy([np.pi-np.pi/2,    np.pi/2,    0]).getquat()
    args['target_orientation_L'] = [ori_L[0], ori_L[1], ori_L[2], ori_L[3]]

    # Initialize node class
    args['duration']=12.0
    cmd_pose_client = CmdPoseClient('client', client1, client2,
        args['m_box'],
        args['target_position_Donkey'],
        args['target_orientation_Donkey'],
        args['target_position_R'],
        args['target_orientation_R'],
        args['target_force_R'],
        args['target_torque_R'],
        args['target_position_L'],
        args['target_orientation_L'],
        args['target_force_L'],
        args['target_torque_L'],
        args['duration']
    )

    # Initialize node class
    args['duration']=2.0
    cmd_pose_client = CmdPoseClient('client', client1, client2,
        args['m_box'],
        args['target_position_Donkey'],
        args['target_orientation_Donkey'],
        args['target_position_R'],
        args['target_orientation_R'],
        args['target_force_R'],
        args['target_torque_R'],
        args['target_position_L'],
        args['target_orientation_L'],
        args['target_force_L'],
        args['target_torque_L'],
        args['duration']
    )


    # execute node
#    rospy.spin()
