#!/usr/bin/env python3
import argparse
from numpy import deg2rad

import rospy
import actionlib
from pushing_msgs.msg import CmdConfigAction, CmdConfigGoal
import numpy as np


class CmdConfigClient(object):
    """docstring for CmdConfigClient."""

    def __init__(self, name, target_config, duration) -> None:
        # initialization message
        self._name = name
        self._target_config = target_config # in radians
        self._duration = duration # in seconds
        rospy.loginfo("%s: Initialized action client class.", self._name)
        # create actionlib client
        self._action_client = actionlib.SimpleActionClient('/chonk/cmd_config', CmdConfigAction)
        # wait until actionlib server starts
        rospy.loginfo("%s: Waiting for action server to start.", self._name)
        self._action_client.wait_for_server()
        rospy.loginfo("%s: Action server started, sending goal.", self._name)
        # creates goal and sends to the action server
        goal = CmdConfigGoal()
        goal.positions = self._target_config
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
    parser = argparse.ArgumentParser(description='Client node to command robot joint configuration.')
#    parser.add_argument("--target_config", nargs=23,
#        help="Give target configuration of the robot in degrees.",
#        type=float, default=[0., 0., 0., 0., 0., -20., 0., 0., 0., 0., 0., -20., 0., 0., 0., 1000., 1000., 1000., 1000., 1000., 1000., 1000., 1000.],
#        metavar=('CHEST_JOINT0', 'HEAD_JOINT0', 'HEAD_JOINT1', 'LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2', 'LARM_JOINT3', 'LARM_JOINT4', 'LARM_JOINT5', 'RARM_JOINT0', 'RARM_JOINT1', 'RARM_JOINT2', 'RARM_JOINT3', 'RARM_JOINT4', 'RARM_JOINT5', 'fl_wheel_joint', 'fl_wheel_link_passive_joint', 'fr_wheel_joint', 'fr_wheel_link_passive_joint', 'rl_wheel_joint', 'rl_wheel_link_passive_joint', 'rr_wheel_joint', 'rr_wheel_link_passive_joint')
#    )
    parser.add_argument("--target_config", nargs=15,
        help="Give target configuration of the robot in degrees.",
        type=float, default=[0., 0., 0., 0., 0., -30., 0., 30., 90., 0., 0., -30., 0., 30., -90],
        metavar=('CHEST_JOINT0', 'HEAD_JOINT0', 'HEAD_JOINT1', 'LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2', 'LARM_JOINT3', 'LARM_JOINT4', 'LARM_JOINT5', 'RARM_JOINT0', 'RARM_JOINT1', 'RARM_JOINT2', 'RARM_JOINT3', 'RARM_JOINT4', 'RARM_JOINT5' )
    )
    parser.add_argument("--duration", help="Give duration of motion in seconds.", type=float, default=4.0)
    args = vars(parser.parse_args())
    # get arguments
    target_config = [deg2rad(q) for q in args['target_config']]
    duration = args['duration']

    # Initialize node
    rospy.init_node('cmd_config_client', anonymous=True)
    # Initialize node class
    cmd_config_client = CmdConfigClient(rospy.get_name(), target_config, duration)
    # execute node
    rospy.spin()
