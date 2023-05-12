#! /usr/bin/env python3
import argparse

import rospy
import actionlib
from pushing_msgs.msg import CmdDualPoseAction, CmdDualPoseGoal

class CmdPoseClient(object):
    """docstring for CmdPoseClient."""

    def __init__(self, name, target_pos_R, target_quat_R, target_pos_L, target_quat_L, duration) -> None:
        # initialization message
        self._name = name
        self._target_pos_R = target_pos_R
        self._target_quat_R = target_quat_R
        self._target_pos_L = target_pos_L
        self._target_quat_L = target_quat_L
        self._duration = duration
        rospy.loginfo("%s: Initialized action client class.", self._name)
        # create actionlib client
        self._action_client = actionlib.SimpleActionClient('/chonk/cmd_pose', CmdDualPoseAction)
        # wait until actionlib server starts
        rospy.loginfo("%s: Waiting for action server to start.", self._name)
        self._action_client.wait_for_server()
        rospy.loginfo("%s: Action server started, sending goal.", self._name)
        # creates goal and sends to the action server
        goal = CmdDualPoseGoal()
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
            done_cb=self.done_cb,
            active_cb=self.active_cb,
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
    parser = argparse.ArgumentParser(description='Client node to command robot end-effector pose.')
    parser.add_argument('--target_position_R', nargs=3,
        help="Give target position of the robot in meters.",
        type=float, default=[0.722, -0.1, 0.955],
        metavar=('POS_X', 'POS_Y', 'POS_Z')
    )
    parser.add_argument('--target_orientation_R', nargs=4,
        help="Give target orientation as a quaternion.",
        type=float, default=[1., 0., 0., 0.],
        metavar=('QUAT_X','QUAT_Y','QUAT_Z','QUAT_W')
    )
    parser.add_argument('--target_position_L', nargs=3,
        help="Give target position of the robot in meters.",
        type=float, default=[0.722, 0.1, 0.955],
        metavar=('POS_X', 'POS_Y', 'POS_Z')
    )
    parser.add_argument('--target_orientation_L', nargs=4,
        help="Give target orientation as a quaternion.",
        type=float, default=[1., 0., 0., 0.],
        metavar=('QUAT_X','QUAT_Y','QUAT_Z','QUAT_W')
    )
    parser.add_argument("--duration", help="Give duration of motion in seconds.", type=float, default=5.0)
    args = vars(parser.parse_args())

    # Initialize node
    rospy.init_node('cmd_pose_client', anonymous=True)
    # Initialize node class
    cmd_pose_client = CmdPoseClient(rospy.get_name(),
        args['target_position_R'],
        args['target_orientation_R'],
        args['target_position_L'],
        args['target_orientation_L'],
        args['duration']
    )
    # execute node
    rospy.spin()
