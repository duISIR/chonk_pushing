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
np.set_printoptions(suppress=True)

import rospy
import actionlib
import optas

from scipy.spatial.transform import Rotation as R

from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
# ROS messages types for the real robot
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension
# ROS messages for command configuration action
from pushing_msgs.msg import TriggerCmdAction, TriggerCmdFeedback, TriggerCmdResult

# For mux controller name
from std_msgs.msg import String
# service for selecting the controller
from topic_tools.srv import MuxSelect

class TeleOp2DActionServer(object):
    """docstring for TeleOp2DActionServer."""

    def __init__(self, name):
        # initialization message
        self.name = name
        rospy.loginfo("%s: Initializing class", self.name)
        #################################################################################
        ## get parameters:
        # workspace limit boundaries
        self._x_min = rospy.get_param('~x_min', 0.2)
        self._x_max = rospy.get_param('~x_max', 0.8)
        self._y_min = rospy.get_param('~y_min', -0.5)
        self._y_max = rospy.get_param('~y_max', 0.5)
        # end-effector orientation limits
        self._roll_min = np.rad2deg(rospy.get_param('~roll_min', 0.0))
        self._roll_max = np.rad2deg(rospy.get_param('~roll_max', 0.0))
        self._pitch_min = np.rad2deg(rospy.get_param('~pitch_min', 0.0))
        self._pitch_max = np.rad2deg(rospy.get_param('~pitch_max', 0.0))
        self._yaw_min = np.rad2deg(rospy.get_param('~yaw_min', -180.0))
        self._yaw_max = np.rad2deg(rospy.get_param('~yaw_max', 180.0))
        # safety gains on joint position and velocity limits
        self._K_safety_lim_q = rospy.get_param('~K_safety_lim_q', 0.99)
        self._K_safety_lim_q_dot = rospy.get_param('~K_safety_lim_q', 0.5)
        # gains on the task cost optimization
        self._K_pos = rospy.get_param('~K_pos', 100.)
        self._K_ori = rospy.get_param('~K_ori', 1.)
        # end-effector frame
        self._link_ee_right = rospy.get_param('~link_ee_right', 'link_ee_right')
        self._link_ee_left = rospy.get_param('~link_ee_left', 'link_ee_left')
        self._link_head = rospy.get_param('~link_head', 'link_head')
        self._link_ref = rospy.get_param('~link_ref', 'link_ref')
        self._link_gaze = rospy.get_param('~link_gaze', 'link_gaze')
        # control frequency
        self._freq = rospy.get_param('~freq', 100)
        self.dt = 1./self._freq
        # publishing command node name
        self._pub_cmd_topic_name = rospy.get_param('~cmd_topic_name', '/command')
        # teleop commands in 2d isometric
        self.joy_max = rospy.get_param('~joy_max', 0.68359375)
        # velocity scalling from the normalized value
        self.K_v_xy = rospy.get_param('~K_v_xy', 0.2)
        self.K_w_z = rospy.get_param('~K_w_z', 0.5)
        #################################################################################
        # load robot_description
        param_robot_description = '~/robot_description'
        if rospy.has_param(param_robot_description):
            self._robot_description = rospy.get_param(param_robot_description)
        else:
            rospy.logerr("%s: Param %s is unavailable!" % (self.name, param_robot_description))
            rospy.signal_shutdown('Incorrect parameter name.')
        #################################################################################
        ### optas
        # set up robot
        right_arm = optas.RobotModel(
            urdf_string=self._robot_description,
            time_derivs=[1],
            param_joints=['HEAD_JOINT0', 'HEAD_JOINT1', 'LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2', 'LARM_JOINT3', 'LARM_JOINT4', 'LARM_JOINT5'],
            name='chonk_right_arm'
        )
        self.right_arm_name = right_arm.get_name()
        self.ndof = right_arm.ndof
        # get right_arm limits
        self.q_min_right_arm = self._K_safety_lim_q * right_arm.dlim[0][0]
        self.q_max_right_arm = self._K_safety_lim_q * right_arm.dlim[0][1]
        self.dq_min_right_arm = self.dt * self._K_safety_lim_q_dot * right_arm.dlim[1][0]
        self.dq_max_right_arm = self.dt * self._K_safety_lim_q_dot * right_arm.dlim[1][1]
        # nominal right_arm configuration
        self.dq_nom = optas.DM.zeros(self.ndof)
        self.right_arm_opt_idx = right_arm.optimized_joint_indexes
        self.n_right_arm_opt = len(self.right_arm_opt_idx)
        self.right_arm_param_idx = right_arm.parameter_joint_indexes
        # set up optimization builder
        builder_right_arm = optas.OptimizationBuilder(T=1, robots=[right_arm], derivs_align=True)
        # get right_arm joint variables
        dq_var = builder_right_arm.get_robot_states_and_parameters(self.right_arm_name, time_deriv=1)
        dq_opt = builder_right_arm.get_model_states(self.right_arm_name, time_deriv=1)
        # set parameters
        q_var = builder_right_arm.add_parameter('q', self.ndof)
        q_opt = q_var[self.right_arm_opt_idx]
        dx = builder_right_arm.add_decision_variables('dx', 3)
        dx_target = builder_right_arm.add_parameter('dx_target', 3)
        z_target = builder_right_arm.add_parameter('z_target', 1)
        # forward differential kinematics
        self.fk = right_arm.get_global_link_position_function(link=self._link_ee_right)
        self.J_pos = right_arm.get_global_link_linear_jacobian_function(link=self._link_ee_right)
        self.quat = right_arm.get_link_quaternion_function(link=self._link_ee_right, base_link=self._link_ref)
        # self.R = right_arm.get_global_link_rotation_function(link=self._link_ee_right)
        self.R = right_arm.get_link_rotation_function(link=self._link_ee_right, base_link=self._link_ref)
        self.J = right_arm.get_global_link_geometric_jacobian_function(link=self._link_ee_right)
        self.rpy = right_arm.get_link_rpy_function(link=self._link_ee_right, base_link=self._link_ref)
        self.J_rpy = right_arm.get_link_angular_analytical_jacobian_function(link=self._link_ee_right, base_link=self._link_ref)
        # cost term
        builder_right_arm.add_cost_term('cost_q', 1.0*optas.sumsqr(dq_opt))
        builder_right_arm.add_cost_term('cost_pos', self._K_pos*optas.sumsqr(dx_target[0:2]-dx[0:2]))
        builder_right_arm.add_cost_term('cost_ori', self._K_ori*optas.sumsqr(dx_target[2]-dx[2]))
        # forward differential kinematics
        builder_right_arm.add_equality_constraint('FDK_pos', (self.J(q_var)[0:2,:])@dq_var, dx[0:2])
        builder_right_arm.add_equality_constraint('FDK_ori', (self.J_rpy(q_var)[2,:])@dq_var, dx[2])
        # add joint position limits
        builder_right_arm.add_bound_inequality_constraint('joint_pos_lim', self.q_min_right_arm, q_opt+dq_opt, self.q_max_right_arm)
        # add joint velocity limitis
        builder_right_arm.add_bound_inequality_constraint('joint_vel_lim', self.dq_min_right_arm, dq_opt, self.dq_max_right_arm)
        # add height constraint
        builder_right_arm.add_equality_constraint('height', self.fk(q_var)[2] + (self.J(q_var)[2,:])@dq_var, z_target)
        # add end-effector yaw-pitch-yaw limits
        builder_right_arm.add_bound_inequality_constraint('roll', self._roll_min, self.rpy(q_var)[0] + self.J_rpy(q_var)[0,:]@dq_var, self._roll_max)
        builder_right_arm.add_bound_inequality_constraint('pitch', self._pitch_min, self.rpy(q_var)[1] + self.J_rpy(q_var)[1,:]@dq_var, self._pitch_max)
        builder_right_arm.add_bound_inequality_constraint('yaw', self._yaw_min, self.rpy(q_var)[2] + self.J_rpy(q_var)[2,:]@dq_var, self._yaw_max)
        # add workspace limits
        builder_right_arm.add_bound_inequality_constraint('x_lim', self._x_min, self.fk(q_var)[0] + self.J(q_var)[0,:]@dq_var, self._x_max)
        builder_right_arm.add_bound_inequality_constraint('y_lim', self._y_min, self.fk(q_var)[1] + self.J(q_var)[1,:]@dq_var, self._y_max)
       # setup solver
        self.solver_right_arm = optas.CasADiSolver(builder_right_arm.build()).setup(
            solver_name='qpoases',
            solver_options={'error_on_fail': False}
        )
        ### ---------------------------------------------------------
        # set up head optimization
        head = optas.RobotModel(
            urdf_string=self._robot_description,
            time_derivs=[1],
            param_joints=['CHEST_JOINT0', 'LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2', 'LARM_JOINT3', 'LARM_JOINT4', 'LARM_JOINT5', 'RARM_JOINT0', 'RARM_JOINT1', 'RARM_JOINT2', 'RARM_JOINT3', 'RARM_JOINT4', 'RARM_JOINT5'],
            name='chonk_head'
        )
        self.head_name = head.get_name()
        self.head_opt_idx = head.optimized_joint_indexes
        self.n_head_opt = len(self.head_opt_idx)
        self.head_param_idx = head.parameter_joint_indexes
        # get head limits
        self.q_min_head = self._K_safety_lim_q * head.dlim[0][0]
        self.q_max_head = self._K_safety_lim_q * head.dlim[0][1]
        self.dq_min_head = self.dt * 0.5 * self._K_safety_lim_q_dot * head.dlim[1][0]
        self.dq_max_head = self.dt * 0.5 * self._K_safety_lim_q_dot * head.dlim[1][1]
        # get head heading
        self.pos_head_fnc = head.get_global_link_position_function(link=self._link_head)
        # get head gaze position
        self.pos_gaze_fnc = head.get_global_link_position_function(link=self._link_gaze)
        J_gaze = head.get_global_link_linear_jacobian_function(link=self._link_gaze)
        # set up optimization builder
        builder_head = optas.OptimizationBuilder(T=1, robots=[head], derivs_align=True)
        # get head joint variables
        dq_var = builder_head.get_robot_states_and_parameters(self.head_name, time_deriv=1)
        dq_opt = builder_head.get_model_states(self.head_name, time_deriv=1)
        # get end-effector pose as parameters
        pos_head = builder_head.add_parameter('pos_head', 3)
        pos_gaze = builder_head.add_parameter('pos_gaze', 3)
        # set parameters
        q_var = builder_head.add_parameter('q', self.ndof)
        q_opt = q_var[self.head_opt_idx]
        # add joint position limits
        builder_head.add_bound_inequality_constraint('joint_pos_lim', self.q_min_head, q_opt+dq_opt, self.q_max_head)
        # add joint velocity limitis
        builder_head.add_bound_inequality_constraint('joint_vel_lim', self.dq_min_head, dq_opt, self.dq_max_head)
        # optimization cost: close to nominal config
        # builder
        builder_head.add_cost_term('cost_dq', optas.norm_2(pos_head-pos_gaze+J_gaze(q_var)@dq_var))
       # setup solver
        self.solver_head = optas.CasADiSolver(builder_head.build()).setup(
            solver_name='qpoases',
            solver_options={'error_on_fail': False}
        )
        #################################################################################
        # initialize variables
        self.q_read = np.zeros(self.ndof)
        self.q_cmd = np.zeros(self.ndof)
        #################################################################################
        # declare joint subscriber
        self._joint_sub = rospy.Subscriber(
            "/joint_states",
            JointState,
            self.read_joint_states_cb
        )
        # declare joystick subscriver
        self._joy_sub = rospy.Subscriber(
            "/spacenav/twist",
            Twist,
            self.read_twist_cb
        )
        # declare joint publisher
        self._joint_pub = rospy.Publisher(
            self._pub_cmd_topic_name,
            Float64MultiArray,
            queue_size=10
        )
        #################################################################################
        # set mux controller selection as wrong by default
        self._correct_mux_selection = False
        # declare mux service
        self._srv_mux_sel = rospy.ServiceProxy(rospy.get_namespace() + '/mux_joint_position/select', MuxSelect)
        # declare subscriber for selected controller
        self._sub_selected_controller = rospy.Subscriber(
            "/mux_selected",
            String,
            self.read_mux_selection
        )
        #################################################################################
        # initialize action messages
        self._feedback = TriggerCmdFeedback()
        self._result = TriggerCmdResult()
        # declare action server
        self._action_server = actionlib.SimpleActionServer(
            'teleop_2d', 
            TriggerCmdAction, 
            execute_cb=None,
            auto_start=False
        )
        # register the preempt callback
        self._action_server.register_goal_callback(self.goal_cb)
        self._action_server.register_preempt_callback(self.preempt_cb)
        # start action server
        self._action_server.start()
        #################################################################################

    def goal_cb(self):
        # activate publishing command
        self._srv_mux_sel(self._pub_cmd_topic_name)
        # accept the new goal request
        acceped_goal = self._action_server.accept_new_goal()
        self.target_pos_z = acceped_goal.pos_z
        # read current robot joint positions for memory
        self.q_cmd = self.q_read
        ### optas
        ### ---------------------------------------------------------
        # initialize the message
        self._msg = Float64MultiArray()
        self._msg.layout = MultiArrayLayout()
        self._msg.layout.data_offset = 0
        self._msg.layout.dim.append(MultiArrayDimension())
        self._msg.layout.dim[0].label = "columns"
        self._msg.layout.dim[0].size = self.ndof
        # create timer
        dur = rospy.Duration(1.0/self._freq)
        self._timer = rospy.Timer(dur, self.timer_cb)

    def timer_cb(self, event):
        """ Publish the robot configuration """

        # make sure that the action is active
        if(not self._action_server.is_active()):
            self._timer.shutdown()
            rospy.logwarn("%s: The action server is NOT active!")
            self._result.trigger_off = False
            self._action_server.set_aborted(self._result)
            return

        # main execution
        if(self._correct_mux_selection):
            # current config
            q_curr = self.q_cmd
            # target displacement
            dx_target = [
                self.dt * self.v_x,
                self.dt * self.v_y,
                self.dt * self.w_z
            ]
            dq = np.zeros(self.ndof)
            ### ---------------------------------------------------------
            ### right hand problem
            self.solver_right_arm.reset_initial_seed({f'{self.right_arm_name}/dq': self.dq_nom[self.right_arm_opt_idx]})
            self.solver_right_arm.reset_parameters({
                f'{self.right_arm_name}/dP': self.dq_nom[self.right_arm_param_idx],
                'q': q_curr,
                'dx_target': dx_target,
                'z_target': self.target_pos_z
            })
            # solve problem
            solution = self.solver_right_arm.solve()
            if self.solver_right_arm.did_solve():
                dq_array = np.asarray(solution[f'{self.right_arm_name}/dq']).T[0]
            else:
                rospy.logwarn("%s: Right arm QP fail to find a solution!" % self.name)
                dq_array = np.zeros(self.n_right_arm_opt)
            dq[self.right_arm_opt_idx] = dq_array
            ### ---------------------------------------------------------
            ### head problem
            self.solver_head.reset_initial_seed({f'{self.head_name}/dq': self.dq_nom[self.head_opt_idx]})
            self.solver_head.reset_parameters({
                f'{self.head_name}/dP': self.dq_nom[self.head_param_idx],
                'pos_head': self.pos_head_fnc(q_curr),
                'pos_gaze': self.pos_gaze_fnc(q_curr)
            })
            # solve problem
            solution = self.solver_head.solve()
            if self.solver_head.did_solve():
                dq_array = np.asarray(solution[f'{self.head_name}/dq']).T[0]
            else:
                rospy.logwarn("%s: Head QP fail to find a solution!" % self.name)
                dq_array = np.zeros(self.n_head_opt)
            dq[self.head_opt_idx] = dq_array
            ### ---------------------------------------------------------
            # integrate solution
            self.q_cmd = q_curr + dq
            # update message
            self._msg.data = self.q_cmd
            # publish message
            self._joint_pub.publish(self._msg)
            # compute progress
            self._feedback.is_active = True
            # publish feedback
            self._action_server.publish_feedback(self._feedback)
        else:
            # shutdown this timer
            self._timer.shutdown()
            rospy.logwarn("%s: Request aborted. The controller selection changed!" % (self.name))
            self._result.trigger_off = False
            self._action_server.set_aborted(self._result)
            return

    def read_twist_cb(self, msg):
        # read planar part of the twist
        v_x = msg.linear.x
        v_y = msg.linear.y
        w_z = msg.angular.z
        # make v_x and v_y isometric
        v_ang = np.arctan2(v_y, v_x)
        c_ang = np.cos(v_ang)
        s_ang = np.sin(v_ang)
        if v_ang > np.deg2rad(-45.) and v_ang < np.deg2rad(45.):
            norm_max = self.joy_max/c_ang
        elif v_ang > np.deg2rad(45.) and v_ang < np.deg2rad(135.):
            norm_max = self.joy_max/s_ang
        elif v_ang < np.deg2rad(-45.) and v_ang > np.deg2rad(-135.):
            norm_max = -self.joy_max/s_ang
        else:
            norm_max = -self.joy_max/c_ang
        # save values
        self.v_x =   self.K_v_xy * (v_x / norm_max)
        self.v_y =   self.K_v_xy * (v_y / norm_max)
        self.w_z = - self.K_w_z * (w_z / self.joy_max)

    def read_joint_states_cb(self, msg):
        self.q_read = np.asarray(list(msg.position))

    def read_mux_selection(self, msg):
        self._correct_mux_selection = (msg.data == self._pub_cmd_topic_name)

    def preempt_cb(self):
        self._timer.shutdown()
        rospy.loginfo("%s: Client preempted this action.", self.name)
        self._result.trigger_off = True
        # set the action state to preempted
        self._action_server.set_preempted(self._result)

if __name__=="__main__":
    # Initialize node
    rospy.init_node("teleop_2d_server", anonymous=True)
    # Initialize node class
    teleop_2d_server = TeleOp2DActionServer(rospy.get_name())
    # executing node
    rospy.spin()
