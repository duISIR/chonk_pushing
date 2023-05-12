#! /usr/bin/env python3

# Copyright (C) 2022 Statistical Machine Learning and Motor Control Group (SLMC)
# Authors: Joao Moura (maintainer)
# email: joao.moura@ed.ac.uk

# This file is part of iiwa_pushing package.

# iiwa_pushing is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# iiwa_pushing is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
import sys
import numpy as np

import rospy
import actionlib

from urdf_parser_py.urdf import URDF

from sensor_msgs.msg import JointState
# ROS messages types for the real robot
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension
from geometry_msgs.msg import Twist
# ROS messages for command configuration action
from pushing_msgs.msg import CmdConfigAction, CmdConfigFeedback, CmdConfigResult

# For mux controller name
from std_msgs.msg import String
# service for selecting the controller
from topic_tools.srv import MuxSelect

class CmdConfigActionServer(object):
    """docstring for CmdConfigActionServer."""

    def __init__(self, name):
        # initialization message
        self._name = name
        rospy.loginfo("%s: Initializing class", self._name)
        # robot name
        # self._robot_name = rospy.get_param('~robot_name', 'iiwa')
        # control frequency
        self._freq = rospy.get_param('~freq', 100)
        # publishing command node name
        self._pub_cmd_topic_name = rospy.get_param('~cmd_topic_name', '/command')
        # load robot_description
        param_robot_description = '~/robot_description'
        if rospy.has_param(param_robot_description):
            self._urdf = URDF.from_parameter_server(param_robot_description)
        else:
            rospy.logerr("%s: Param %s is unavailable!" % (self._name, param_robot_description))
            rospy.signal_shutdown('Incorrect parameter name.')
        # get joint names and limits
        self.joint_names = [jnt.name for jnt in self._urdf.joints if (jnt.type != 'fixed' and jnt.type !='continuous')]
        self.ndof = len(self.joint_names)
        self.joint_limit_min = [jnt.limit.lower for jnt in self._urdf.joints if (jnt.type != 'fixed' and jnt.type !='continuous')]
        self.joint_limit_max = [jnt.limit.upper for jnt in self._urdf.joints if (jnt.type != 'fixed' and jnt.type !='continuous')]
        # initialize variables
        self.q_curr = np.zeros(self.ndof)
        # declare joint subscriber
        self._joint_sub = rospy.Subscriber(
            "/chonk/joint_states",
            JointState,
            self.read_joint_states_cb
        )
        # declare joint publisher
#        self._joint_pub = rospy.Publisher(
#            self._pub_cmd_topic_name,
#            Float64MultiArray,
#            queue_size=10
#        )
        self._joint_pub = rospy.Publisher(
            "/chonk/streaming_controller/command",
            Float64MultiArray,
            queue_size=10
        )
        # This is for donkey_velocity_controller
        self._joint_pub_velocity = rospy.Publisher(
            "/chonk/donkey_velocity_controller/cmd_vel",
            Twist,
            queue_size=10
        )
        # set mux controller selection as wrong by default
        self._correct_mux_selection = False
        self._mux_selection = "None"
        # declare mux service
        self._srv_mux_sel = rospy.ServiceProxy(rospy.get_namespace() + '/mux_joint_position/select', MuxSelect)
        # declare subscriber for selected controller
        self._sub_selected_controller = rospy.Subscriber(
            "/mux_selected",
            String,
            self.read_mux_selection
        )
        # initialize action messages
        self._feedback = CmdConfigFeedback()
        self._result = CmdConfigResult()
        # declare action server
        self._action_server = actionlib.SimpleActionServer(
            'cmd_config', 
            CmdConfigAction, 
            execute_cb=None,
            auto_start=False
        )
        # register the preempt callback
        self._action_server.register_goal_callback(self.goal_cb)
        self._action_server.register_preempt_callback(self.preempt_cb)
        # start action server
        self._action_server.start()

    def goal_cb(self):
        # activate publishing command
        self._srv_mux_sel(self._pub_cmd_topic_name)
        # accept the new goal request
        acceped_goal = self._action_server.accept_new_goal()
        qT_array = acceped_goal.positions
        T = acceped_goal.duration
        # check size of the target vector
        goal_ndof = len(qT_array)
        if  goal_ndof != self.ndof:
            rospy.logwarn("%s: Request aborted. Goal has %d positions instead of %d." % (self._name, goal_ndof, self.ndof))
            self._result.reached_goal = False
            self._action_server.set_aborted(self._result)
            return
        # print goal request
        rospy.loginfo("%s: Request to send robot joints to %s in %.1f seconds." % (self._name, qT_array, T))
        # helper variables
        self._steps = int(T * self._freq)
        self._idx = 0
        q0 = self.q_curr
        qT = np.clip(
            qT_array,
            a_min=self.joint_limit_min,
            a_max=self.joint_limit_max
        ) # make sure that request is within joint limits
        Dq = qT - q0
        # interpolate between current and target configuration 
        # polynomial obtained for zero speed (3rd order) and acceleratin (5th order)
        # at the initial and final time
        # self._q = lambda t: q0 + (3.*((t/T)**2) - 2.*((t/T)**3))*Dq # 3rd order
        self._q = lambda t: q0 + (10.*((t/T)**3) - 15.*((t/T)**4) + 6.*((t/T)**5))*Dq # 5th order
        self._t = np.linspace(0., T, self._steps + 1)
        # initialize the message
        self._msg = Float64MultiArray()
        self._msg.layout = MultiArrayLayout()
        self._msg.layout.data_offset = 0
        self._msg.layout.dim.append(MultiArrayDimension())
        self._msg.layout.dim[0].label = "columns"
        self._msg.layout.dim[0].size = self.ndof

        self._msg_velocity = Twist()
        self._msg_velocity.linear.x = 0
        self._msg_velocity.linear.y = 0
        self._msg_velocity.linear.z = 0
        self._msg_velocity.angular.x = 0
        self._msg_velocity.angular.y = 0
        self._msg_velocity.angular.z = 0

        # create timer
        dur = rospy.Duration(1.0/self._freq)
        self._timer = rospy.Timer(dur, self.timer_cb)

    def timer_cb(self, event):
        """ Publish the robot configuration """

        # make sure that the action is active
        if(not self._action_server.is_active()):
            rospy.logwarn("%s: The action server is NOT active!")
            self._timer.shutdown()
            return

        # main execution
        if(self._idx < self._steps):
            if(self._correct_mux_selection):
                # increment idx (in here counts starts with 1)
                self._idx += 1
                # compute next configuration with lambda function
                q_next = self._q(self._t[self._idx])
                # update message
                self._msg.data = q_next
                # publish message
                self._joint_pub.publish(self._msg)
                self._joint_pub_velocity.publish(self._msg_velocity)  # This is for donkey_velocity_controller
                # compute progress
                self._feedback.progress = (self._idx*100)/self._steps
                # publish feedback
                self._action_server.publish_feedback(self._feedback)
            else:
                # shutdown this timer
                self._timer.shutdown()
                rospy.logwarn("%s: Request aborted. The controller selection changed!" % (self._name))
                self._result.reached_goal = False
                self._action_server.set_aborted(self._result)
                return
        else:
            # shutdown this timer
            self._timer.shutdown()
            # set the action state to succeeded
            rospy.loginfo("%s: Succeeded" % self._name)
            self._result.reached_goal = True
            self._action_server.set_succeeded(self._result)
            return

    def read_joint_states_cb(self, msg):
        self.q_curr = np.asarray(list(msg.position)[:self.ndof])

    def read_mux_selection(self, msg):
        self._mux_selection = msg.data
        self._correct_mux_selection = (msg.data == self._pub_cmd_topic_name)

    def preempt_cb(self):
        rospy.loginfo("%s: Preempted.", self._name)
        # set the action state to preempted
        self._action_server.set_preempted()

if __name__=="__main__":
    # Initialize node
    rospy.init_node("cmd_config_server", anonymous=True)
    # Initialize node class
    cmd_config_server = CmdConfigActionServer(rospy.get_name())
    # executing node
    rospy.spin()
