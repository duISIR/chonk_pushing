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
import casadi
import argparse

from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import Wrench
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension
from pushing_msgs.msg import CmdChonkPoseForceAction, CmdChonkPoseForceFeedback, CmdChonkPoseForceResult
from pushing_msgs.msg import CmdChonkForceAction, CmdChonkForceGoal

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from urdf_parser_py.urdf import URDF
import tf
from X_fromRandP import X_fromRandP, X_fromRandP_different

# For mux controller name
from std_msgs.msg import String
# service for selecting the controller
from topic_tools.srv import MuxSelect
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class CmdForceClient(object):
    """docstring for CmdForceClient."""

    def __init__(self, name, client, m_box, target_force_R, target_torque_R, target_force_L, target_torque_L, duration) -> None:
        # initialization message
        self._name = name
        self._m_box = m_box
        self._target_force_R = target_force_R
        self._target_torque_R = target_torque_R
        self._target_force_L = target_force_L
        self._target_torque_L = target_torque_L
        self._duration = duration
        rospy.loginfo("%s: Initialized action client class.", self._name)
        # create actionlib client
        self._action_client = client
        # wait until actionlib server starts
        rospy.loginfo("%s: Waiting for action server to start.", self._name)
        self._action_client.wait_for_server()
        rospy.loginfo("%s: Action server started, sending goal.", self._name)
        # creates goal and sends to the action server
        goal = CmdChonkForceGoal()
        goal.m_box = self._m_box
        goal.ForceTorqueR.force.x = self._target_force_R[0]
        goal.ForceTorqueR.force.y = self._target_force_R[1]
        goal.ForceTorqueR.force.z = self._target_force_R[2]
        goal.ForceTorqueR.torque.x = self._target_torque_R[0]
        goal.ForceTorqueR.torque.y = self._target_torque_R[1]
        goal.ForceTorqueR.torque.z = self._target_torque_R[2]
        goal.ForceTorqueL.force.x = self._target_force_L[0]
        goal.ForceTorqueL.force.y = self._target_force_L[1]
        goal.ForceTorqueL.force.z = self._target_force_L[2]
        goal.ForceTorqueL.torque.x = self._target_torque_L[0]
        goal.ForceTorqueL.torque.y = self._target_torque_L[1]
        goal.ForceTorqueL.torque.z = self._target_torque_L[2]
        goal.duration = self._duration
        # sends the goal to the action server
        rospy.loginfo("%s: Send goal request to action server.", self._name)
        self._action_client.send_goal(
            goal,
#            done_cb=self.done_cb,
#            active_cb=self.active_cb,
#            feedback_cb=self.feedback_cb
        )

class CmdPoseActionServer(object):
    """docstring for CmdPoseActionServer."""

    def __init__(self, name):
        # initialization message
        self._name = name
        rospy.loginfo("%s: Initializing class", self._name)
        ## get parameters:
        ## --------------------------------------
        # workspace limit boundaries
        self._x_min = rospy.get_param('~x_min', -10000)
        self._x_max = rospy.get_param('~x_max',  10000)
        self._y_min = rospy.get_param('~y_min', -10000)
        self._y_max = rospy.get_param('~y_max',  10000)
        self._z_min = rospy.get_param('~z_min', -10000)
        self._z_max = rospy.get_param('~z_max',  10000)
        self._pos_min = np.asarray([self._x_min, self._y_min, self._z_min])
        self._pos_max = np.asarray([self._x_max, self._y_max, self._z_max])
        # robot name
        # donkey_base frame
        self._link_donkey = rospy.get_param('~link_donkey', 'link_donkey')
        # end-effector frame
        self._link_ee_right = rospy.get_param('~link_ee_right', 'link_ee_right')
        self._link_ee_left = rospy.get_param('~link_ee_left', 'link_ee_left')
#        self._link_sensor_right = rospy.get_param('~link_sensor_right', 'link_sensor_right')
#        self._link_sensor_left = rospy.get_param('~link_sensor_left', 'link_sensor_left')
        self._link_head = rospy.get_param('~link_head', 'link_head')
        self._link_gaze = rospy.get_param('~link_gaze', 'link_gaze')
        # control frequency
        self._freq = rospy.get_param('~freq', 50)
        # publishing command node name
        self._pub_cmd_topic_name = rospy.get_param('~cmd_topic_name', '/command')
        # load robot_description
        param_robot_description = '~/robot_description_wholebody'
        if rospy.has_param(param_robot_description):
            self._robot_description = rospy.get_param(param_robot_description)
            self._urdf = URDF.from_parameter_server(param_robot_description)
        else:
            rospy.logerr("%s: Param %s is unavailable!" % (self._name, param_robot_description))
            rospy.signal_shutdown('Incorrect parameter name.')
        self.joint_names = [jnt.name for jnt in self._urdf.joints if (jnt.type != 'fixed')]
        self.ndof_base = 3
        self.ndof_position_control = len(self.joint_names) - self.ndof_base
        self.ndof = self.ndof_base + self.ndof_position_control
        ### ---------------------------------------------------------
        # initialize variables for planner
        self.q_curr = np.zeros(self.ndof)
        self.q_curr_joint = np.zeros(self.ndof_position_control)
        self.q_curr_base = np.zeros(self.ndof_base)
        self.dq_curr = np.zeros(self.ndof)
        self.dq_curr_joint = np.zeros(self.ndof_position_control)
        self.dq_curr_base = np.zeros(self.ndof_base)
        self.joint_names_position = []
        self.joint_names_base = ['base_joint_1', 'base_joint_2', 'base_joint_3']
        self.donkey_R = np.zeros((3, 3))
        self.donkey_position = np.zeros(3)
        self.donkey_velocity = np.zeros(3)
        self.donkey_angular_velocity = np.zeros(3)
        ### optas
        ### ---------------------------------------------------------
        # set up whole-body MPC planner in real time
        self.wholebodyMPC_planner= optas.RobotModel(
            urdf_string=self._robot_description,
            time_derivs=[0],
            param_joints=['base_joint_1', 'base_joint_2', 'base_joint_3', 'CHEST_JOINT0', 'HEAD_JOINT0', 'HEAD_JOINT1', 'LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2', 'LARM_JOINT3', 'LARM_JOINT4', 'LARM_JOINT5', 'RARM_JOINT0', 'RARM_JOINT1', 'RARM_JOINT2', 'RARM_JOINT3', 'RARM_JOINT4', 'RARM_JOINT5'],
            name='chonk_wholebodyMPC_planner'
        )
        self.wholebodyMPC_planner_name = self.wholebodyMPC_planner.get_name()
#        self.dt_MPC_planner = 0.1 # time step
        self.T_MPC_planner = 10 # T is number of time steps
        self.T_MPC = 6
        self.dt_MPC = 0.15
#        self.duration_MPC_planner = float(self.T_MPC_planner-1)*self.dt_MPC_planner
        # nominal robot configuration
        self.wholebodyMPC_planner_opt_idx = self.wholebodyMPC_planner.optimized_joint_indexes
        self.wholebodyMPC_planner_param_idx = self.wholebodyMPC_planner.parameter_joint_indexes
        # set up optimization builder.
        builder_wholebodyMPC_planner = optas.OptimizationBuilder(T=1, robots=[self.wholebodyMPC_planner])
        builder_wholebodyMPC_planner._decision_variables = optas.sx_container.SXContainer()
        builder_wholebodyMPC_planner._parameters = optas.sx_container.SXContainer()
        builder_wholebodyMPC_planner._lin_eq_constraints = optas.sx_container.SXContainer()
        builder_wholebodyMPC_planner._lin_ineq_constraints = optas.sx_container.SXContainer()
        builder_wholebodyMPC_planner._ineq_constraints = optas.sx_container.SXContainer()
        builder_wholebodyMPC_planner._eq_constraints = optas.sx_container.SXContainer()
        # get robot state variables, get velocity state variables
        R_pos_Right = builder_wholebodyMPC_planner.add_decision_variables('R_pos_Right', 3, self.T_MPC_planner)
        R_pos_Left = builder_wholebodyMPC_planner.add_decision_variables('R_pos_Left', 3, self.T_MPC_planner)
        R_ori_Right = builder_wholebodyMPC_planner.add_decision_variables('R_ori_Right', 4, self.T_MPC_planner)
        R_ori_Left = builder_wholebodyMPC_planner.add_decision_variables('R_ori_Left', 4, self.T_MPC_planner)
#        R_middle = builder_wholebodyMPC_planner.add_decision_variables('R_middle', 3, self.T_MPC_planner)
#        r_ep = builder_wholebodyMPC_planner.add_decision_variables('r_ep', self.T_MPC_planner)
#        r_ep_start = builder_wholebodyMPC_planner.add_decision_variables('r_ep_start', 12)
#        r_ori_ep = builder_wholebodyMPC_planner.add_decision_variables('r_ori_ep', 2, self.T_MPC_planner)


        duration_MPC_planner = builder_wholebodyMPC_planner.add_parameter('duration_MPC_planner', 1)

        self.pos_fnc_Right_planner = self.wholebodyMPC_planner.get_global_link_position_function(link=self._link_ee_right)
        self.pos_fnc_Left_planner = self.wholebodyMPC_planner.get_global_link_position_function(link=self._link_ee_left)
        self.ori_fnc_Right_planner = self.wholebodyMPC_planner.get_global_link_quaternion_function(link=self._link_ee_right)
        self.ori_fnc_Left_planner = self.wholebodyMPC_planner.get_global_link_quaternion_function(link=self._link_ee_left)

        self.pos_Jac_fnc_Right_planner = self.wholebodyMPC_planner.get_global_link_linear_jacobian_function(link=self._link_ee_right)
        self.pos_Jac_fnc_Left_planner = self.wholebodyMPC_planner.get_global_link_linear_jacobian_function(link=self._link_ee_left)
        self.ori_Jac_fnc_Right_planner = self.wholebodyMPC_planner.get_global_link_angular_geometric_jacobian_function(link=self._link_ee_right)
        self.ori_Jac_fnc_Left_planner = self.wholebodyMPC_planner.get_global_link_angular_geometric_jacobian_function(link=self._link_ee_left)

        t = np.linspace(0., 1., self.T_MPC_planner)
        self.n_planner = self.T_MPC_planner -1 # N in Bezier curve
        # Add parameters
#        init_r_position_middle = builder_wholebodyMPC_planner.add_parameter('init_r_position_middle', 2)
        init_r_position_Right = builder_wholebodyMPC_planner.add_parameter('init_r_position_Right', 3)
        init_r_position_Left = builder_wholebodyMPC_planner.add_parameter('init_r_position_Left', 3)
        init_r_orientation_Right = builder_wholebodyMPC_planner.add_parameter('init_r_orientation_Right', 4)
        init_r_orientation_Left = builder_wholebodyMPC_planner.add_parameter('init_r_orientation_Left', 4)

        init_dr_position_Right = builder_wholebodyMPC_planner.add_parameter('init_dr_position_Right', 3)
        init_dr_position_Left = builder_wholebodyMPC_planner.add_parameter('init_dr_position_Left', 3)
        init_dr_orientation_Right = builder_wholebodyMPC_planner.add_parameter('init_dr_orientation_Right', 4)
        init_dr_orientation_Left = builder_wholebodyMPC_planner.add_parameter('init_dr_orientation_Left', 4)
        quaternion_fixed180 = np.array([1, 0, 0, 0])

        # get end-effector pose as parameters
        pos_R = builder_wholebodyMPC_planner.add_parameter('pos_R', 3)
        pos_L = builder_wholebodyMPC_planner.add_parameter('pos_L', 3)
        ori_R = builder_wholebodyMPC_planner.add_parameter('ori_R', 4)
        ori_L = builder_wholebodyMPC_planner.add_parameter('ori_L', 4)

        # define q function depending on P
        r_middle_var_MPC = optas.casadi.SX(np.zeros((3, self.T_MPC_planner)))
        r_pos_RARM_var_MPC = optas.casadi.SX(np.zeros((3, self.T_MPC_planner)))
        r_pos_LARM_var_MPC = optas.casadi.SX(np.zeros((3, self.T_MPC_planner)))
        r_ori_RARM_var_MPC = optas.casadi.SX(np.zeros((4, self.T_MPC_planner)))
        r_ori_LARM_var_MPC = optas.casadi.SX(np.zeros((4, self.T_MPC_planner)))

        for i in range(self.T_MPC_planner):
            for j in range(self.T_MPC_planner):
#                r_middle_var_MPC[:, i] += self.BC(self.n_planner, j) * t[i]**j * (1-t[i])**(self.n_planner-j) * R_middle[:, j]
                r_pos_RARM_var_MPC[:, i] += self.BC(self.n_planner, j) * t[i]**j * (1-t[i])**(self.n_planner-j) * R_pos_Right[:, j]
                r_pos_LARM_var_MPC[:, i] += self.BC(self.n_planner, j) * t[i]**j * (1-t[i])**(self.n_planner-j) * R_pos_Left[:, j]
                r_ori_RARM_var_MPC[:, i] += self.BC(self.n_planner, j) * t[i]**j * (1-t[i])**(self.n_planner-j) * R_ori_Right[:, j]
                r_ori_LARM_var_MPC[:, i] += self.BC(self.n_planner, j) * t[i]**j * (1-t[i])**(self.n_planner-j) * R_ori_Left[:, j]
            r_middle_var_MPC[:, i] = 0.5*(r_pos_RARM_var_MPC[:, i]+r_pos_LARM_var_MPC[:, i])
#            builder_wholebodyMPC_planner.add_cost_term('Right_arm_z' + str(i), optas.sumsqr(r_pos_RARM_var_MPC[2, i] - r_middle_var_MPC[2, i]))
#            builder_wholebodyMPC_planner.add_cost_term('Left_arm_z' + str(i), optas.sumsqr(r_pos_LARM_var_MPC[2, i] -  r_middle_var_MPC[2, i]))
#            builder_wholebodyMPC_planner.add_equality_constraint('middle' +str(i), r_middle_var_MPC[:, i], rhs=0.5*(r_pos_RARM_var_MPC[:, i]+r_pos_LARM_var_MPC[:, i]))
#            builder_wholebodyMPC_planner.add_leq_inequality_constraint('quaternion_equality_right' + str(i),  lhs=optas.sumsqr(R_ori_Right[:, i]), rhs=1. + r_ori_ep[0, i])
#            builder_wholebodyMPC_planner.add_leq_inequality_constraint('quaternion_equality_left' + str(i),  lhs=optas.sumsqr(R_ori_Left[:, i]), rhs=1. + r_ori_ep[1, i])
            builder_wholebodyMPC_planner.add_equality_constraint('quaternion_equality_right' + str(i),  lhs=optas.sumsqr(R_ori_Right[:, i]), rhs=1.)
            builder_wholebodyMPC_planner.add_equality_constraint('quaternion_equality_left' + str(i),  lhs=optas.sumsqr(R_ori_Left[:, i]), rhs=1.)
#            builder_wholebodyMPC_planner.add_cost_term('Two_arm orientation parallel' + str(i), 20*optas.sumsqr(r_ori_RARM_var_MPC[:, i] -  self.qaQb( r_ori_LARM_var_MPC[:, i], quaternion_fixed180 )))
            builder_wholebodyMPC_planner.add_cost_term('Two_arm end height same' + str(i), optas.sumsqr(r_pos_RARM_var_MPC[2, i] - r_pos_LARM_var_MPC[2, i]))
#            builder_wholebodyMPC_planner.add_cost_term('Right_arm_align' + str(i), 5*optas.sumsqr( self.skew_optas(self.quatToRotationZ(r_ori_RARM_var_MPC[:, i])) @ (r_pos_RARM_var_MPC[:, i] - r_pos_LARM_var_MPC[:, i])   ))
#            builder_wholebodyMPC_planner.add_cost_term('Left_arm_align' + str(i), 5*optas.sumsqr( self.skew_optas(self.quatToRotationZ(r_ori_LARM_var_MPC[:, i])) @ (r_pos_RARM_var_MPC[:, i] - r_pos_LARM_var_MPC[:, i])   ))
            builder_wholebodyMPC_planner.add_cost_term('Right_arm_align_x' + str(i), 20*optas.sumsqr( self.quatToRotationX(r_ori_RARM_var_MPC[:, i]).T @ (r_pos_RARM_var_MPC[:, i] - r_pos_LARM_var_MPC[:, i])   ))
            builder_wholebodyMPC_planner.add_cost_term('Right_arm_align_y' + str(i), 5*optas.sumsqr( self.quatToRotationY(r_ori_RARM_var_MPC[:, i]).T @ (r_pos_RARM_var_MPC[:, i] - r_pos_LARM_var_MPC[:, i])   ))

            builder_wholebodyMPC_planner.add_cost_term('Left_arm_align_x' + str(i), 20*optas.sumsqr( self.quatToRotationX(r_ori_LARM_var_MPC[:, i]).T @ (r_pos_RARM_var_MPC[:, i] - r_pos_LARM_var_MPC[:, i])   ))
            builder_wholebodyMPC_planner.add_cost_term('Left_arm_align_y' + str(i), 5*optas.sumsqr( self.quatToRotationY(r_ori_LARM_var_MPC[:, i]).T @ (r_pos_RARM_var_MPC[:, i] - r_pos_LARM_var_MPC[:, i])   ))

#            if(i>0):
#                builder_wholebodyMPC_planner.add_equality_constraint('Two_arm orientation parallel' + str(i), lhs=R_ori_Right[:, i], rhs= self.qaQb( R_ori_Left[:, i], quaternion_fixed180 ))


#            builder_wholebodyMPC_planner.add_cost_term('Right_arm_obstacle_x' + str(i),  optas.sumsqr(r_pos_RARM_var_MPC[0, i] - pos_R[0, i]))
#            builder_wholebodyMPC_planner.add_cost_term('Right_arm_obstacle_y' + str(i),  optas.sumsqr(r_pos_RARM_var_MPC[1, i] - pos_R[1, i]))
#            builder_wholebodyMPC_planner.add_cost_term('Left_arm_obstacle_x' + str(i),  optas.sumsqr(r_pos_LARM_var_MPC[0, i] - pos_L[0, i]))
#            builder_wholebodyMPC_planner.add_cost_term('Left_arm_obstacle_y' + str(i),   optas.sumsqr(r_pos_LARM_var_MPC[1, i] - pos_L[1, i]))

#            builder_wholebodyMPC_planner.add_cost_term('mini_two_arm_pos_x_difference' + str(i),  optas.sumsqr(r_pos_RARM_var_MPC[0, i] - r_pos_LARM_var_MPC[0, i]))

#            builder_wholebodyMPC_planner.add_equality_constraint('two_arm_x_position' + str(i),  lhs=r_pos_RARM_var_MPC[0, i], rhs=r_pos_LARM_var_MPC[0, i])

        # optimization cost: close to target
#        builder_wholebodyMPC_planner.add_cost_term('Final_middle_x',  optas.sumsqr(r_middle_var_MPC[0, -1] - 0.5*(pos_R[0] + pos_L[0])))
#        builder_wholebodyMPC_planner.add_cost_term('Final_middle_y',  10*optas.sumsqr(r_middle_var_MPC[1, -1] - 0.5*(pos_R[1] + pos_L[1])))
#        builder_wholebodyMPC_planner.add_equality_constraint('Final_middle', r_middle_var_MPC[:, self.T_MPC_planner-1], rhs=0.5*(pos_R+pos_L))
        builder_wholebodyMPC_planner.add_equality_constraint('Final_Right_arm', r_pos_RARM_var_MPC[:, self.T_MPC_planner-1], rhs=pos_R)
        builder_wholebodyMPC_planner.add_equality_constraint('Final_Left_arm', r_pos_LARM_var_MPC[:, self.T_MPC_planner-1], rhs=pos_L)
        builder_wholebodyMPC_planner.add_equality_constraint('Final_Right_ori_arm', r_ori_RARM_var_MPC[:, self.T_MPC_planner-1], rhs=ori_R)
        builder_wholebodyMPC_planner.add_equality_constraint('Final_Left_ori_arm', r_ori_LARM_var_MPC[:, self.T_MPC_planner-1], rhs=ori_L)



        for i in range(self.T_MPC_planner):
            obstacle_pos = np.asarray([[4.67], [1.63]])
            obstacle_radius = 0.9
#            builder_wholebodyMPC_planner.add_geq_inequality_constraint('middle_obstacle' + str(i), lhs=(r_middle_var_MPC[0:2, i]-obstacle_pos).T @ (r_middle_var_MPC[0:2, i]-obstacle_pos), rhs=obstacle_radius**2 + r_ep[i])
            builder_wholebodyMPC_planner.add_geq_inequality_constraint('middle_obstacle' + str(i), lhs=(r_middle_var_MPC[0:2, i]-obstacle_pos).T @ (r_middle_var_MPC[0:2, i]-obstacle_pos), rhs=obstacle_radius**2)

#        builder_wholebodyMPC_planner.add_cost_term('minimize_r_ep',  10*optas.sumsqr(r_ep))
#        builder_wholebodyMPC_planner.add_cost_term('minimize_r_ori_ep',  10*optas.sumsqr(r_ori_ep))

        # add position constraint at the beginning state
#        builder_wholebodyMPC_planner.add_equality_constraint('init_r_middle', R_middle[:, 0], rhs=0.5*(init_r_position_Right + init_r_position_Left))
        builder_wholebodyMPC_planner.add_equality_constraint('init_r_Right', R_pos_Right[:, 0], rhs=init_r_position_Right)
        builder_wholebodyMPC_planner.add_equality_constraint('init_r_Left', R_pos_Left[:, 0], rhs=init_r_position_Left)
        builder_wholebodyMPC_planner.add_equality_constraint('init_r_ori_Right', R_ori_Right[:, 0], rhs=init_r_orientation_Right)
        builder_wholebodyMPC_planner.add_equality_constraint('init_r_ori_Left', R_ori_Left[:, 0], rhs=init_r_orientation_Left)

        for i in range(self.T_MPC_planner):
            if(i<(self.T_MPC_planner -1)):
#                builder_wholebodyMPC_planner.add_cost_term('Right_distance' + str(i),      50 * optas.sumsqr(R_pos_Right[:, i+1] - R_pos_Right[:, i]))
#                builder_wholebodyMPC_planner.add_cost_term('Left_distance' + str(i),       50 * optas.sumsqr(R_pos_Left[:, i+1]  - R_pos_Left[:, i]))
                builder_wholebodyMPC_planner.add_cost_term('Right_distance_x' + str(i), 50 * optas.sumsqr(R_pos_Right[0, i+1] - R_pos_Right[0, i]))
                builder_wholebodyMPC_planner.add_cost_term('Right_distance_y' + str(i), 50 * optas.sumsqr(R_pos_Right[1, i+1] - R_pos_Right[1, i]))
                builder_wholebodyMPC_planner.add_cost_term('Right_distance_z' + str(i), 50 * optas.sumsqr(R_pos_Right[2, i+1] - R_pos_Right[2, i]))

                builder_wholebodyMPC_planner.add_cost_term('Left_distance_x' + str(i), 50 * optas.sumsqr(R_pos_Left[0, i+1] - R_pos_Left[0, i]))
                builder_wholebodyMPC_planner.add_cost_term('Left_distance_y' + str(i), 50 * optas.sumsqr(R_pos_Left[1, i+1] - R_pos_Left[1, i]))
                builder_wholebodyMPC_planner.add_cost_term('Left_distance_z' + str(i), 50 * optas.sumsqr(R_pos_Left[2, i+1] - R_pos_Left[2, i]))

                builder_wholebodyMPC_planner.add_cost_term('Right_ori_distance' + str(i),  50 * optas.sumsqr(R_ori_Right[:, i+1] - R_ori_Right[:, i]))
                builder_wholebodyMPC_planner.add_cost_term('Left_ori_distance' + str(i),   50 * optas.sumsqr(R_ori_Left[:, i+1]  - R_ori_Left[:, i]))
            if(i<(self.T_MPC_planner -2)):
#                builder_wholebodyMPC_planner.add_cost_term('dRight_distance' + str(i),     50 * optas.sumsqr(R_pos_Right[:, i+2]-2*R_pos_Right[:, i+1] + R_pos_Right[:, i]))
#                builder_wholebodyMPC_planner.add_cost_term('dLeft_distance' + str(i),      50 * optas.sumsqr(R_pos_Left[:, i+2] -2*R_pos_Left[:, i+1]  + R_pos_Left[:, i]))
                builder_wholebodyMPC_planner.add_cost_term('dRight_distance_x' + str(i), 50 * optas.sumsqr(R_pos_Right[0, i+2]-2*R_pos_Right[0, i+1] + R_pos_Right[0, i]))
                builder_wholebodyMPC_planner.add_cost_term('dRight_distance_y' + str(i), 50 * optas.sumsqr(R_pos_Right[1, i+2]-2*R_pos_Right[1, i+1] + R_pos_Right[1, i]))
                builder_wholebodyMPC_planner.add_cost_term('dRight_distance_z' + str(i), 50 * optas.sumsqr(R_pos_Right[2, i+2]-2*R_pos_Right[2, i+1] + R_pos_Right[2, i]))

                builder_wholebodyMPC_planner.add_cost_term('dLeft_distance_x' + str(i),  50 * optas.sumsqr(R_pos_Left[0, i+2]-2*R_pos_Left[0, i+1] + R_pos_Left[0, i]))
                builder_wholebodyMPC_planner.add_cost_term('dLeft_distance_y' + str(i),  50 * optas.sumsqr(R_pos_Left[1, i+2]-2*R_pos_Left[1, i+1] + R_pos_Left[1, i]))
                builder_wholebodyMPC_planner.add_cost_term('dLeft_distance_z' + str(i),  50 * optas.sumsqr(R_pos_Left[2, i+2]-2*R_pos_Left[2, i+1] + R_pos_Left[2, i]))

                builder_wholebodyMPC_planner.add_cost_term('dRight_ori_distance' + str(i), 50 * optas.sumsqr(R_ori_Right[:, i+2]-2*R_ori_Right[:, i+1] + R_ori_Right[:, i]))
                builder_wholebodyMPC_planner.add_cost_term('dLeft_ori_distance' + str(i),  50 * optas.sumsqr(R_ori_Left[:, i+2] -2*R_ori_Left[:, i+1]  + R_ori_Left[:, i]))

#        dr_middle_var_MPC = optas.casadi.SX(np.zeros((3, self.T_MPC_planner)))
        dr_pos_RARM_var_MPC = optas.casadi.SX(np.zeros((3, self.T_MPC_planner)))
        dr_pos_LARM_var_MPC = optas.casadi.SX(np.zeros((3, self.T_MPC_planner)))
        dr_ori_RARM_var_MPC = optas.casadi.SX(np.zeros((4, self.T_MPC_planner)))
        dr_ori_LARM_var_MPC = optas.casadi.SX(np.zeros((4, self.T_MPC_planner)))

        w_dr = duration_MPC_planner**2 * 0.0005/float(self.T_MPC_planner)
        for i in range(self.T_MPC_planner):
            for j in range(self.T_MPC_planner-1):
#                dr_middle_var_MPC[:, i] += self.BC(self.n_planner-1, j) * t[i]**j * (1-t[i])**(self.n_planner-1-j) * self.n_planner * (R_middle[:, j+1] -  R_middle[:, j])
                dr_pos_RARM_var_MPC[:, i] += (1./duration_MPC_planner) * self.BC(self.n_planner-1, j) * t[i]**j * (1-t[i])**(self.n_planner-1-j) * self.n_planner * (R_pos_Right[:, j+1] -  R_pos_Right[:, j])
                dr_pos_LARM_var_MPC[:, i] += (1./duration_MPC_planner) * self.BC(self.n_planner-1, j) * t[i]**j * (1-t[i])**(self.n_planner-1-j) * self.n_planner * (R_pos_Left[:, j+1] -  R_pos_Left[:, j])
                dr_ori_RARM_var_MPC[:, i] += (1./duration_MPC_planner) * self.BC(self.n_planner-1, j) * t[i]**j * (1-t[i])**(self.n_planner-1-j) * self.n_planner * (R_ori_Right[:, j+1] -  R_ori_Right[:, j])
                dr_ori_LARM_var_MPC[:, i] += (1./duration_MPC_planner) * self.BC(self.n_planner-1, j) * t[i]**j * (1-t[i])**(self.n_planner-1-j) * self.n_planner * (R_ori_Left[:, j+1] -  R_ori_Left[:, j])
#            builder_wholebodyMPC_planner.add_cost_term('minimize_dr_middle' + str(i), w_dr * optas.sumsqr(dr_middle_var_MPC[:, i]))
#            builder_wholebodyMPC_planner.add_cost_term('minimize_dr_right' + str(i), w_dr * optas.sumsqr(dr_pos_RARM_var_MPC[:, i]))
#            builder_wholebodyMPC_planner.add_cost_term('minimize_dr_left' + str(i), w_dr * optas.sumsqr(dr_pos_LARM_var_MPC[:, i]))
#            builder_wholebodyMPC_planner.add_cost_term('minimize_dr_ori_right' + str(i), w_dr * optas.sumsqr(dr_ori_RARM_var_MPC[:, i]))
#            builder_wholebodyMPC_planner.add_cost_term('minimize_dr_ori_left' + str(i), w_dr * optas.sumsqr(dr_ori_LARM_var_MPC[:, i]))


#        builder_wholebodyMPC_planner.add_equality_constraint('init_dr_middle', dr_middle_var_MPC[:, 0], rhs=0.5*(init_dr_position_Right + init_dr_position_Left))
        builder_wholebodyMPC_planner.add_equality_constraint('init_dr_Right', dr_pos_RARM_var_MPC[0:3, 0], rhs=init_dr_position_Right[0:3])
        builder_wholebodyMPC_planner.add_equality_constraint('init_dr_Left', dr_pos_LARM_var_MPC[0:3, 0], rhs=init_dr_position_Left[0:3])
#        builder_wholebodyMPC_planner.add_equality_constraint('init_dr_ori_Right', dr_ori_RARM_var_MPC[:, 0], rhs=init_dr_orientation_Right)
#        builder_wholebodyMPC_planner.add_equality_constraint('init_dr_ori_Left', dr_ori_LARM_var_MPC[:, 0], rhs=init_dr_orientation_Left)
#        builder_wholebodyMPC_planner.add_equality_constraint('final_dr_middle', dr_middle_var_MPC[:,-1], rhs=np.zeros(3))
        builder_wholebodyMPC_planner.add_equality_constraint('final_dr_right', dr_pos_RARM_var_MPC[:,self.T_MPC_planner-1], rhs=np.zeros(3))
        builder_wholebodyMPC_planner.add_equality_constraint('final_dr_left', dr_pos_LARM_var_MPC[:,self.T_MPC_planner-1], rhs=np.zeros(3))
        builder_wholebodyMPC_planner.add_equality_constraint('final_dr_ori_right', dr_ori_RARM_var_MPC[:,self.T_MPC_planner-1], rhs=np.zeros(4))
        builder_wholebodyMPC_planner.add_equality_constraint('final_dr_ori_left', dr_ori_LARM_var_MPC[:,self.T_MPC_planner-1], rhs=np.zeros(4))

        lower_velocity = np.array([-0.9, -0.9, -0.5]); upper_velocity = np.array([0.9, 0.9, 0.5]);
        for i in range(self.T_MPC_planner-1):
#            builder_wholebodyMPC_planner.add_bound_inequality_constraint('dr_vel_limit_right' + str(i), lhs=lower_velocity-r_ep_start[0:3], mid=(1./duration_MPC_planner) * self.n_planner * (R_pos_Right[:, i+1] -  R_pos_Right[:, i]), rhs=upper_velocity+r_ep_start[0:3])
#            builder_wholebodyMPC_planner.add_bound_inequality_constraint('dr_vel_limit_left' + str(i), lhs=lower_velocity-r_ep_start[3:6], mid=(1./duration_MPC_planner) * self.n_planner * (R_pos_Left[:, i+1] -  R_pos_Left[:, i]), rhs=upper_velocity+r_ep_start[3:6])
            builder_wholebodyMPC_planner.add_bound_inequality_constraint('dr_vel_limit_right' + str(i), lhs=lower_velocity, mid=(1./duration_MPC_planner) * self.n_planner * (R_pos_Right[:, i+1] -  R_pos_Right[:, i]), rhs=upper_velocity)
            builder_wholebodyMPC_planner.add_bound_inequality_constraint('dr_vel_limit_left' + str(i), lhs=lower_velocity, mid=(1./duration_MPC_planner) * self.n_planner * (R_pos_Left[:, i+1] -  R_pos_Left[:, i]), rhs=upper_velocity)

#        ddr_middle_var_MPC = optas.casadi.SX(np.zeros((3, self.T_MPC_planner)))
        ddr_pos_RARM_var_MPC = optas.casadi.SX(np.zeros((3, self.T_MPC_planner)))
        ddr_pos_LARM_var_MPC = optas.casadi.SX(np.zeros((3, self.T_MPC_planner)))
        ddr_ori_RARM_var_MPC = optas.casadi.SX(np.zeros((4, self.T_MPC_planner)))
        ddr_ori_LARM_var_MPC = optas.casadi.SX(np.zeros((4, self.T_MPC_planner)))

        w_ddr = duration_MPC_planner**4 * 0.0005/float(self.T_MPC_planner)
        for i in range(self.T_MPC_planner):
            for j in range(self.T_MPC_planner-2):
#                ddr_middle_var_MPC[:, i] += self.BC(self.n_planner-2, j) * t[i]**j * (1-t[i])**(self.n_planner-2-j) * self.n_planner * (self.n_planner-1)* (R_middle[:, j+2] -  2*R_middle[:, j+1] + R_middle[:, j])
                ddr_pos_RARM_var_MPC[:, i] += (1./duration_MPC_planner)**2 * self.BC(self.n_planner-2, j) * t[i]**j * (1-t[i])**(self.n_planner-2-j) * self.n_planner * (self.n_planner-1)* (R_pos_Right[:, j+2] -  2*R_pos_Right[:, j+1] + R_pos_Right[:, j])
                ddr_pos_LARM_var_MPC[:, i] += (1./duration_MPC_planner)**2 * self.BC(self.n_planner-2, j) * t[i]**j * (1-t[i])**(self.n_planner-2-j) * self.n_planner * (self.n_planner-1)* (R_pos_Left[:, j+2] -  2*R_pos_Left[:, j+1] + R_pos_Left[:, j])
                ddr_ori_RARM_var_MPC[:, i] += (1./duration_MPC_planner)**2 * self.BC(self.n_planner-2, j) * t[i]**j * (1-t[i])**(self.n_planner-2-j) * self.n_planner * (self.n_planner-1)* (R_ori_Right[:, j+2] -  2*R_ori_Right[:, j+1] + R_ori_Right[:, j])
                ddr_ori_LARM_var_MPC[:, i] += (1./duration_MPC_planner)**2 * self.BC(self.n_planner-2, j) * t[i]**j * (1-t[i])**(self.n_planner-2-j) * self.n_planner * (self.n_planner-1)* (R_ori_Left[:, j+2] -  2*R_ori_Left[:, j+1] + R_ori_Left[:, j])
#            builder_wholebodyMPC_planner.add_cost_term('minimize_ddr_middle' + str(i), w_ddr * optas.sumsqr(ddr_middle_var_MPC[:, i]))
#            builder_wholebodyMPC_planner.add_cost_term('minimize_ddr_right' + str(i), w_ddr * optas.sumsqr(ddr_pos_RARM_var_MPC[:, i]))
#            builder_wholebodyMPC_planner.add_cost_term('minimize_ddr_left' + str(i), w_ddr * optas.sumsqr(ddr_pos_LARM_var_MPC[:, i]))
#            builder_wholebodyMPC_planner.add_cost_term('minimize_ddr_ori_right' + str(i), w_ddr * optas.sumsqr(ddr_ori_RARM_var_MPC[:, i]))
#            builder_wholebodyMPC_planner.add_cost_term('minimize_ddr_ori_left' + str(i), w_ddr * optas.sumsqr(ddr_ori_LARM_var_MPC[:, i]))

#            builder_wholebodyMPC_planner.add_cost_term('minimize_ddr_two arm equal' + str(i), w_ddr * 10 * optas.sumsqr(ddr_pos_RARM_var_MPC[:, i] - ddr_pos_LARM_var_MPC[:, i]))


#        builder_wholebodyMPC_planner.add_equality_constraint('final_ddr_middle', ddr_middle_var_MPC[:,-1], rhs=np.zeros(3))
        builder_wholebodyMPC_planner.add_equality_constraint('final_ddr_right', ddr_pos_RARM_var_MPC[:,self.T_MPC_planner-1], rhs=np.zeros(3))
        builder_wholebodyMPC_planner.add_equality_constraint('final_ddr_left', ddr_pos_LARM_var_MPC[:,self.T_MPC_planner-1], rhs=np.zeros(3))
        builder_wholebodyMPC_planner.add_equality_constraint('final_ddr_ori_right', ddr_ori_RARM_var_MPC[:,self.T_MPC_planner-1], rhs=np.zeros(4))
        builder_wholebodyMPC_planner.add_equality_constraint('final_ddr_ori_left', ddr_ori_LARM_var_MPC[:,self.T_MPC_planner-1], rhs=np.zeros(4))

        lower_acceleration = np.array([-0.9, -0.9, -0.5]); upper_acceleration = np.array([0.9, 0.9, 0.5]);
        for i in range(self.T_MPC_planner-2):
#            builder_wholebodyMPC_planner.add_bound_inequality_constraint('ddr_acc_limit_right' + str(i), lhs=lower_acceleration-r_ep_start[6:9], mid=(1./duration_MPC_planner)**2 * self.n_planner * (self.n_planner - 1) * (R_pos_Right[:, i+2] - 2 * R_pos_Right[:, i+1] + R_pos_Right[:, i]), rhs=upper_acceleration+r_ep_start[6:9])
#            builder_wholebodyMPC_planner.add_bound_inequality_constraint('ddr_acc_limit_left' + str(i), lhs=lower_acceleration-r_ep_start[9:12], mid=(1./duration_MPC_planner)**2 * self.n_planner * (self.n_planner - 1) * (R_pos_Left[:, i+2] - 2 * R_pos_Left[:, i+1] +  R_pos_Left[:, i]), rhs=upper_acceleration+r_ep_start[9:12])
            builder_wholebodyMPC_planner.add_bound_inequality_constraint('ddr_acc_limit_right' + str(i), lhs=lower_acceleration, mid=(1./duration_MPC_planner)**2 * self.n_planner * (self.n_planner - 1) * (R_pos_Right[:, i+2] - 2 * R_pos_Right[:, i+1] + R_pos_Right[:, i]), rhs=upper_acceleration)
            builder_wholebodyMPC_planner.add_bound_inequality_constraint('ddr_acc_limit_left' + str(i), lhs=lower_acceleration, mid=(1./duration_MPC_planner)**2 * self.n_planner * (self.n_planner - 1) * (R_pos_Left[:, i+2] - 2 * R_pos_Left[:, i+1] +  R_pos_Left[:, i]), rhs=upper_acceleration)


#        builder_wholebodyMPC_planner.add_equality_constraint('trajectory_middle_ddr_right1', R_pos_Right[:,8], rhs=R_pos_Right[:,9])
#        builder_wholebodyMPC_planner.add_equality_constraint('trajectory_middle_ddr_right2', R_pos_Right[:,9], rhs=R_pos_Right[:,10])

#        builder_wholebodyMPC_planner.add_equality_constraint('trajectory_middle_ddr_left1', R_pos_Left[:,8], rhs=R_pos_Left[:,9])
#        builder_wholebodyMPC_planner.add_equality_constraint('trajectory_middle_ddr_left2', R_pos_Left[:,9], rhs=R_pos_Left[:,10])

#        builder_wholebodyMPC_planner.add_equality_constraint('final_dddr_right', R_pos_Right[:,self.T_MPC_planner-4], rhs=R_pos_Right[:,self.T_MPC_planner-1])
#        builder_wholebodyMPC_planner.add_equality_constraint('final_dddr_left', R_pos_Left[:,self.T_MPC_planner-4], rhs=R_pos_Left[:,self.T_MPC_planner-1])
#        builder_wholebodyMPC_planner.add_cost_term('minimize_r_ep_start',  100*optas.sumsqr(r_ep_start))

        # setup solver
        self.solver_wholebodyMPC_planner = optas.CasADiSolver(optimization=builder_wholebodyMPC_planner.build()).setup('knitro', solver_options={
                                                                                                       'knitro.OutLev': 0,
                                                                                                       'print_time': 0,
#                                                                                                       'knitro.par_msnumthreads': 14,
                                                                                                       'knitro.act_qpalg': 1,
                                                                                                       'knitro.FeasTol': 1e-4, 'knitro.OptTol': 1e-4, 'knitro.ftol':1e-4,
                                                                                                       'knitro.algorithm':3, 'knitro.linsolver':2,
#                                                                                                       'knitro.maxtime_real': 4.0e-3,
                                                                                                       'knitro.bar_initpt':3, 'knitro.bar_murule':4, 'knitro.bar_penaltycons': 1,
                                                                                                       'knitro.bar_penaltyrule':2, 'knitro.bar_switchrule':2, 'knitro.linesearch': 1
                                                                                                       } )
        self.solution_MPC_planner = None

        ### ---------------------------------------------------------
        # Build client
        # parse arguments from terminal
#        self.parser = argparse.ArgumentParser(description='Client node to command robot end-effector force and torque.')
        # parse donkey arguments
#        self.parser.add_argument("--m_box", help="Give box mass.", type=float, default=0.0,  metavar=('m_box'))
#        self.parser.add_argument('--target_force_R', nargs=3,
#            help="Give target position of the robot in meters.",
#            type=float, default=[0, 0, 0],
#            metavar=('force_R_X', 'force_R_Y', 'force_R_Z')
#        )
#        self.parser.add_argument('--target_torque_R', nargs=3,
#            help="Give target orientation as a quaternion.",
#            type=float, default=[0, 0, 0],
#            metavar=('torque_R_X','torque_R_Y','torque_R_Z')
#        )
        # parse left arm arguments
#        self.parser.add_argument('--target_force_L', nargs=3,
#            help="Give target position of the robot in meters.",
#            type=float, default=[0, 0, 0],
#            metavar=('force_L_X', 'force_L_Y', 'force_L_Z')
#        )
#        self.parser.add_argument('--target_torque_L', nargs=3,
#            help="Give target orientation as a quaternion.",
#            type=float, default=[0, 0, 0],
#            metavar=('torque_L_X','torque_L_Y','torque_L_Z')
#        )
#        self.parser.add_argument("--duration", help="Give duration of motion in seconds.", type=float, default=8.0, metavar=('duration'))
#        print(self.parser.parse_args())
#        self.args = vars(self.parser.parse_args())
        self.client = actionlib.SimpleActionClient('/chonk/cmd_force', CmdChonkForceAction)
        ### ---------------------------------------------------------
        # declare ft_sensor subscriber
#        self._ft_right_sub = rospy.Subscriber(
#            "/ft_right/raw/data",
#            WrenchStamped,
#            self.read_ft_sensor_right_data_cb
#        )
#        self._ft_left_sub = rospy.Subscriber(
#            "/ft_left/raw/data",
#            WrenchStamped,
#            self.read_ft_sensor_left_data_cb
#        )
        # declare joint subscriber
        self._joint_sub = rospy.Subscriber(
            "/chonk/joint_states",
            JointState,
            self.read_joint_states_cb
        )
#        self._joint_sub_base = rospy.Subscriber(
#            "/chonk/donkey_velocity_controller/odom",
#            Odometry,
#            self.read_base_states_cb
#        )
#        self._joint_sub_base = rospy.Subscriber(
#            "/tf",
#            Odometry,
#            self.read_base_states_cb
#        )
        self._joint_sub_base = rospy.Subscriber("/chonk/base_pose_ground_truth", Odometry, self.read_base_states_cb)
        # declare joint publisher
#        self._joint_pub = rospy.Publisher(
#            self._pub_cmd_topic_name,
#            Float64MultiArray,
#            queue_size=10
#        )
        self._motion_reference_pub = rospy.Publisher(
            "/chonk/motion_reference",
            Float64MultiArray,
            queue_size=10
        )
        # initialize the message
        self._msg = Float64MultiArray()
        self._msg.layout = MultiArrayLayout()
        self._msg.layout.data_offset = 0
        self._msg.layout.dim.append(MultiArrayDimension())
        self._msg.layout.dim[0].label = "columns"
        self._msg.layout.dim[0].size = self.T_MPC * (3+3+4+4)
        self.merged_arr = np.zeros(((3+3+4+4),self.T_MPC))

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
        # initialize action messages
        self._feedback = CmdChonkPoseForceFeedback()
        self._result = CmdChonkPoseForceResult()
        # declare action server
        self._action_server = actionlib.SimpleActionServer(
            'cmd_pose',
            CmdChonkPoseForceAction,
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
        # desired end-effector position
        self.pos_Right = np.asarray([acceped_goal.poseR.position.x, acceped_goal.poseR.position.y, acceped_goal.poseR.position.z])
        self.pos_Left = np.asarray([acceped_goal.poseL.position.x, acceped_goal.poseL.position.y, acceped_goal.poseL.position.z])
        self.ori_Right = np.asarray([acceped_goal.poseR.orientation.x, acceped_goal.poseR.orientation.y, acceped_goal.poseR.orientation.z, acceped_goal.poseR.orientation.w])
        self.ori_Left = np.asarray([acceped_goal.poseL.orientation.x, acceped_goal.poseL.orientation.y, acceped_goal.poseL.orientation.z, acceped_goal.poseL.orientation.w])
        # check boundaries of the position
        if (self.pos_Right > self._pos_max).any() or (self.pos_Right < self._pos_min).any():
            rospy.logwarn("%s: Request aborted. Goal position (%.2f, %.2f, %.2f) is outside of the workspace boundaries. Check parameters for this node." % (self._name, self.pos_Right[0], self.pos_Right[1], self.pos_Right[2]))
            self._result.reached_goal = False
            self._action_server.set_aborted(self._result)
            return
        if (self.pos_Left > self._pos_max).any() or (self.pos_Left < self._pos_min).any():
            rospy.logwarn("%s: Lequest aborted. Goal position (%.2f, %.2f, %.2f) is outside of the workspace boundaries. Check parameters for this node." % (self._name, self.pos_Left[0], self.pos_Left[1], self.pos_Left[2]))
            self._result.reached_goal = False
            self._action_server.set_aborted(self._result)
            return
        # print goal request
        rospy.loginfo("%s: Request to send right arm to position (%.2f, %.2f, %.2f) with orientation (%.2f, %.2f, %.2f, %.2f), and left arm to position (%.2f, %.2f, %.2f) with orientation (%.2f, %.2f, %.2f, %.2f) in %.1f seconds." % (
                self._name,
                self.pos_Right[0], self.pos_Right[1], self.pos_Right[2],
                self.ori_Right[0], self.ori_Right[1], self.ori_Right[2], self.ori_Right[3],
                self.pos_Left[0], self.pos_Left[1], self.pos_Left[2],
                self.ori_Left[0], self.ori_Left[1], self.ori_Left[2], self.ori_Left[3],
                acceped_goal.duration
            )
        )
        # read current robot joint positions
        self.q_curr = np.concatenate((self.q_curr_base, self.q_curr_joint), axis=None)
        self.dq_curr = np.concatenate((self.dq_curr_base, self.dq_curr_joint), axis=None)
        qT = np.zeros(self.ndof)
        self.joint_names = self.joint_names_base + self.joint_names_position
        ### optas
        ### ---------------------------------------------------------
        # get two-arm end effector trajectory in the operational space
        q0 = self.q_curr.T
        self.duration = acceped_goal.duration
        self.duration_MPC_planner = acceped_goal.duration
        self._steps = int(self.duration * self._freq)
        self._idx = 0
        # current right and left arm end effector position and quaternion
        start_RARM_quat = np.asarray(self.wholebodyMPC_planner.get_global_link_quaternion(link=self._link_ee_right, q=q0)).T[0]
        start_RARM_pos = np.asarray(self.wholebodyMPC_planner.get_global_link_position(link=self._link_ee_right, q=q0)).T[0]
        start_LARM_quat = np.asarray(self.wholebodyMPC_planner.get_global_link_quaternion(link=self._link_ee_left, q=q0)).T[0]
        start_LARM_pos = np.asarray(self.wholebodyMPC_planner.get_global_link_position(link=self._link_ee_left, q=q0)).T[0]
        # derivation of right and left arm end effector position and quaternion compared with the beginning ee position and quaternion
        Derivation_RARM_Pos = self.pos_Right - start_RARM_pos
        Derivation_RARM_Quat = self.ori_Right - start_RARM_quat
        Derivation_LARM_Pos = self.pos_Left - start_LARM_pos
        Derivation_LARM_Quat = self.ori_Left - start_LARM_quat
        # interpolate between current and target position polynomial obtained for zero speed (3rd order) and acceleratin (5th order) at the initial and final time
        self._RARM_ee_Pos_trajectory = lambda t: start_RARM_pos + (10.*((t/self.duration)**3) - 15.*((t/self.duration)**4) + 6.*((t/self.duration)**5))*Derivation_RARM_Pos # 5th order
        self._DRARM_ee_Pos_trajectory = lambda t: (30.*((t/self.duration)**2) - 60.*((t/self.duration)**3) +30.*((t/self.duration)**4))*(Derivation_RARM_Pos/self.duration)
        self._LARM_ee_Pos_trajectory = lambda t: start_LARM_pos + (10.*((t/self.duration)**3) - 15.*((t/self.duration)**4) + 6.*((t/self.duration)**5))*Derivation_LARM_Pos # 5th order
        self._DLARM_ee_Pos_trajectory = lambda t: (30.*((t/self.duration)**2) - 60.*((t/self.duration)**3) +30.*((t/self.duration)**4))*(Derivation_LARM_Pos/self.duration)
        # interpolate between current and target quaternion polynomial obtained for zero speed (3rd order) and acceleratin (5th order) at the initial and final time
        self._RARM_ee_Quat_trajectory = lambda t: start_RARM_quat + (10.*((t/self.duration)**3) - 15.*((t/self.duration)**4) + 6.*((t/self.duration)**5))*Derivation_RARM_Quat # 5th order
        self._DRARM_ee_Quat_trajectory = lambda t: (30.*((t/self.duration)**2) - 60.*((t/self.duration)**3) +30.*((t/self.duration)**4))*(Derivation_RARM_Quat/self.duration)
        self._LARM_ee_Quat_trajectory = lambda t: start_LARM_quat + (10.*((t/self.duration)**3) - 15.*((t/self.duration)**4) + 6.*((t/self.duration)**5))*Derivation_LARM_Quat # 5th order
        self._DLARM_ee_Quat_trajectory = lambda t: (30.*((t/self.duration)**2) - 60.*((t/self.duration)**3) +30.*((t/self.duration)**4))*(Derivation_LARM_Quat/self.duration)

        self._t = np.linspace(0., self.duration, self._steps + 1)
#        self.args['duration']= acceped_goal.duration
#        self.args['m_box'] = acceped_goal.m_box
#        self.args['target_force_R'] = [acceped_goal.ForceTorqueR.force.x, acceped_goal.ForceTorqueR.force.y, acceped_goal.ForceTorqueR.force.z]
#        self.args['target_torque_R'] = [acceped_goal.ForceTorqueR.torque.x, acceped_goal.ForceTorqueR.torque.y, acceped_goal.ForceTorqueR.torque.z]
#        self.args['target_force_L'] = [acceped_goal.ForceTorqueL.force.x, acceped_goal.ForceTorqueL.force.y, acceped_goal.ForceTorqueL.force.z]
#        self.args['target_torque_L'] = [acceped_goal.ForceTorqueL.torque.x, acceped_goal.ForceTorqueL.torque.y, acceped_goal.ForceTorqueL.torque.z]
#        cmd_force_client = CmdForceClient('client', self.client,
#            self.args['m_box'],
#            self.args['target_force_R'],
#            self.args['target_torque_R'],
#            self.args['target_force_L'],
#            self.args['target_torque_L'],
#            self.args['duration']
#        )
        self.force_Right_goal = np.asarray([acceped_goal.ForceTorqueR.force.x, acceped_goal.ForceTorqueR.force.y, acceped_goal.ForceTorqueR.force.z])
        self.torque_Right_goal = np.asarray([acceped_goal.ForceTorqueR.torque.x, acceped_goal.ForceTorqueR.torque.y, acceped_goal.ForceTorqueR.torque.z])
        self.force_Left_goal = np.asarray([acceped_goal.ForceTorqueL.force.x, acceped_goal.ForceTorqueL.force.y, acceped_goal.ForceTorqueL.force.z])
        self.torque_Left_goal = np.asarray([acceped_goal.ForceTorqueL.torque.x, acceped_goal.ForceTorqueL.torque.y, acceped_goal.ForceTorqueL.torque.z])
        cmd_force_client = CmdForceClient('client', self.client,
                                          acceped_goal.m_box,
                                          self.force_Right_goal,
                                          self.torque_Right_goal,
                                          self.force_Left_goal,
                                          self.torque_Left_goal,
                                          acceped_goal.duration)
        ### ---------------------------------------------------------
        # create timer
        dur = rospy.Duration(1.0/self._freq)
        self._timer = rospy.Timer(dur, self.timer_cb)


    def timer_cb(self, event):
        """ Publish the robot configuration """
        # read current robot joint positions
        self.q_curr = np.concatenate((self.q_curr_base, self.q_curr_joint), axis=None)
        self.dq_curr = np.concatenate((self.dq_curr_base, self.dq_curr_joint), axis=None)
#        """ee mass data in ee frame"""
#        ft_ee_mass_right = X_fromRandP_different(self.rot_ee_right_fnc_global(self.q_curr), self.pos_mass_INee_Right).T @ self.mass_ee_Force
#        ft_ee_mass_left = X_fromRandP_different(self.rot_ee_left_fnc_global(self.q_curr), self.pos_mass_INee_Left).T @ self.mass_ee_Force
#        """Sensor data delete mass influence"""
#        ft_ee_composite_right = ft_ee_right + ft_ee_mass_right
#        ft_ee_composite_left = ft_ee_left + ft_ee_mass_left

        # make sure that the action is active
        if(not self._action_server.is_active()):
            self._timer.shutdown()
            rospy.logwarn("%s: The action server is NOT active!")
            self._result.reached_goal = False
            self._action_server.set_aborted(self._result)
            return

        # main execution
        if(self._idx < self._steps):
            if(self._correct_mux_selection):
                # increment idx (in here counts starts with 1)
                self.ti_MPC_planner = 0 # time index of the MPC
                self._idx += 1

                r_pos_actual_Right = np.array(self.pos_fnc_Right_planner(self.q_curr))[0:3]
                r_pos_actual_Left = np.array(self.pos_fnc_Left_planner(self.q_curr))[0:3]
                dr_pos_actual_Right = np.asarray(self.pos_Jac_fnc_Right_planner(self.q_curr)) @ self.dq_curr
                dr_pos_actual_Left = np.asarray(self.pos_Jac_fnc_Left_planner(self.q_curr)) @ self.dq_curr

                r_ori_actual_Right = np.array(self.ori_fnc_Right_planner(self.q_curr))[0:4]
                r_ori_actual_Left = np.array(self.ori_fnc_Left_planner(self.q_curr))[0:4]
                dr_ori_actual_Right = self.angular_velocity_to_quaternionRate(r_ori_actual_Right[:, 0]) @ np.asarray(self.ori_Jac_fnc_Right_planner(self.q_curr)) @ self.dq_curr
                dr_ori_actual_Left = self.angular_velocity_to_quaternionRate(r_ori_actual_Left[:, 0]) @ np.asarray(self.ori_Jac_fnc_Left_planner(self.q_curr)) @ self.dq_curr

                ### optas. Solve the whole-body MPC planner
                # set initial seed
                if self.solution_MPC_planner is None:
                    pos_R_goal = []; pos_L_goal = []; ori_R_goal = []; ori_L_goal = [];
                    _t = np.asarray(np.linspace(0., self.duration_MPC_planner, self.T_MPC_planner))
                    for i in range(self.T_MPC_planner):
                        try:
                            g_rarm_ee_pos = self._RARM_ee_Pos_trajectory(_t[i]).flatten()
                            pos_R_goal.append(g_rarm_ee_pos.tolist())
                            g_larm_ee_pos = self._LARM_ee_Pos_trajectory(_t[i]).flatten()
                            pos_L_goal.append(g_larm_ee_pos.tolist())
                            g_rarm_ee_ori = self._RARM_ee_Quat_trajectory(_t[i]).flatten()
                            ori_R_goal.append(g_rarm_ee_ori.tolist())
                            g_larm_ee_ori = self._LARM_ee_Quat_trajectory(_t[i]).flatten()
                            ori_L_goal.append(g_larm_ee_ori.tolist())
                        except ValueError:
                            pos_R_goal.append(g_rarm_ee_pos.tolist()) # i.e. previous goal
                            pos_L_goal.append(g_larm_ee_pos.tolist()) # i.e. previous goal
                            ori_R_goal.append(g_rarm_ee_ori.tolist())     # i.e. previous goal
                            ori_L_goal.append(g_larm_ee_ori.tolist())     # i.e. previous goal
                    pos_R_goal = optas.np.array(pos_R_goal).T
                    pos_L_goal = optas.np.array(pos_L_goal).T
                    ori_R_goal = optas.np.array(ori_R_goal).T
                    ori_L_goal = optas.np.array(ori_L_goal).T
                    self.solver_wholebodyMPC_planner.reset_initial_seed({f'R_pos_Right': pos_R_goal, f'R_pos_Left': pos_L_goal,
#                                                                         f'r_ep': np.zeros(self.T_MPC_planner),
#                                                                         f'r_ep_start': np.zeros(12),
                                                                         f'R_ori_Right': ori_R_goal, f'R_ori_Left': ori_L_goal,
#                                                                         f'r_ori_ep': np.zeros((2, self.T_MPC_planner)),
                                                                         })
#                    self.solver_wholebodyMPC_planner.reset_initial_seed({f'R_pos_Right': np.zeros((3, self.T_MPC_planner)),
#                                                                         f'R_pos_Left': np.zeros((3, self.T_MPC_planner)), f'r_ep': np.zeros(self.T_MPC_planner) })
                # set initial seed
                if self.solution_MPC_planner is not None:
                    self.solver_wholebodyMPC_planner.reset_initial_seed({f'R_pos_Right': self.solution_MPC_planner[f'R_pos_Right'],
                                                                         f'R_pos_Left': self.solution_MPC_planner[f'R_pos_Left'],
                                                                         f'R_ori_Right': self.solution_MPC_planner[f'R_ori_Right'],
                                                                         f'R_ori_Left': self.solution_MPC_planner[f'R_ori_Left'],
#                                                                         f'r_ori_ep': self.solution_MPC_planner[f'r_ori_ep'],
#                                                                         f'r_ep': self.solution_MPC_planner[f'r_ep'],
#                                                                         f'r_ep_start': self.solution_MPC_planner[f'r_ep_start']
                                                                         })

                self.solver_wholebodyMPC_planner.reset_parameters({'pos_R': self.pos_Right, 'pos_L': self.pos_Left,
                                                                   'ori_R': self.ori_Right, 'ori_L': self.ori_Left,
                                                                   'init_r_position_Right': r_pos_actual_Right, 'init_r_position_Left': r_pos_actual_Left,
                                                                   'init_dr_position_Right': dr_pos_actual_Right, 'init_dr_position_Left': dr_pos_actual_Left,
                                                                   'init_r_orientation_Right': r_ori_actual_Right, 'init_r_orientation_Left': r_ori_actual_Left,
                                                                   'init_dr_orientation_Right': dr_ori_actual_Right, 'init_dr_orientation_Left': dr_ori_actual_Left,
                                                                   'duration_MPC_planner': self.duration_MPC_planner} )
                # solve problem
                self.solution_MPC_planner = self.solver_wholebodyMPC_planner.opt.decision_variables.vec2dict(self.solver_wholebodyMPC_planner._solve())
                R_pos_Right = np.asarray(self.solution_MPC_planner[f'R_pos_Right'])
                R_pos_Left = np.asarray(self.solution_MPC_planner[f'R_pos_Left'])
                R_ori_Right = np.asarray(self.solution_MPC_planner[f'R_ori_Right'])
                R_ori_Left = np.asarray(self.solution_MPC_planner[f'R_ori_Left'])

                pos_R_reasonal = np.zeros((3, self.T_MPC))
                pos_L_reasonal = np.zeros((3, self.T_MPC))
                ori_R_reasonal = np.zeros((4, self.T_MPC))
                ori_L_reasonal = np.zeros((4, self.T_MPC))

                for i in range(self.T_MPC):
                    self.ti_MPC = self._t[self._idx-1]  + self.dt_MPC*i
#                    if((self.ti_MPC - self._t[self._idx-1]) <= self.duration_MPC_planner and self.duration_MPC_planner>= 2./self._freq):
                    if((self.ti_MPC - self._t[self._idx-1]) <= self.duration_MPC_planner):
                        t_nomalized = (self.ti_MPC - self._t[self._idx-1])/self.duration_MPC_planner
                        for j in range(self.T_MPC_planner):
                            pos_R_reasonal[:, i] += self.BC(self.n_planner, j) * (t_nomalized)**j * (1-t_nomalized)**(self.n_planner-j) * R_pos_Right[:, j]
                            pos_L_reasonal[:, i] += self.BC(self.n_planner, j) * (t_nomalized)**j * (1-t_nomalized)**(self.n_planner-j) * R_pos_Left[:, j]
                            ori_R_reasonal[:, i] += self.BC(self.n_planner, j) * (t_nomalized)**j * (1-t_nomalized)**(self.n_planner-j) * R_ori_Right[:, j]
                            ori_L_reasonal[:, i] += self.BC(self.n_planner, j) * (t_nomalized)**j * (1-t_nomalized)**(self.n_planner-j) * R_ori_Left[:, j]
                    else:
                        t_nomalized = 1.
                        for j in range(self.T_MPC_planner):
                            pos_R_reasonal[:, i] += self.BC(self.n_planner, j) * (t_nomalized)**j * (1-t_nomalized)**(self.n_planner-j) * R_pos_Right[:, j]
                            pos_L_reasonal[:, i] += self.BC(self.n_planner, j) * (t_nomalized)**j * (1-t_nomalized)**(self.n_planner-j) * R_pos_Left[:, j]
                            ori_R_reasonal[:, i] += self.BC(self.n_planner, j) * (t_nomalized)**j * (1-t_nomalized)**(self.n_planner-j) * R_ori_Right[:, j]
                            ori_L_reasonal[:, i] += self.BC(self.n_planner, j) * (t_nomalized)**j * (1-t_nomalized)**(self.n_planner-j) * R_ori_Left[:, j]
                for i in range(self.T_MPC):
                    for j in range(4):
                        ori_R_reasonal[j, i] /= optas.sumsqr(ori_R_reasonal[:, i])
                        ori_L_reasonal[j, i] /= optas.sumsqr(ori_L_reasonal[:, i])


                self.duration_MPC_planner = self.duration - self._idx/self._freq

                # publish message
                self.merged_arr = np.vstack((pos_R_reasonal, pos_L_reasonal, ori_R_reasonal, ori_L_reasonal))
                self._msg.data = self.merged_arr.flatten().tolist()
                self._motion_reference_pub.publish(self._msg)

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
        self.q_curr_joint = np.asarray(list(msg.position)[:self.ndof_position_control])
        self.joint_names_position = msg.name[:self.ndof_position_control]
        self.dq_curr_joint = np.asarray(list(msg.velocity)[:self.ndof_position_control])

#    def read_ft_sensor_right_data_cb(self, msg):
#        """ paranet to child: the force/torque from robot to ee"""
#        self.ft_right = -np.asarray([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z, msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])

#    def read_ft_sensor_left_data_cb(self, msg):
#        """ paranet to child: the force/torque from robot to ee"""
#        self.ft_left = -np.asarray([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z, msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])

    def read_base_states_cb(self, msg):
        base_euler_angle = tf.transformations.euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        self.q_curr_base = [msg.pose.pose.position.x, msg.pose.pose.position.y, base_euler_angle[2]]
        self.donkey_R = optas.spatialmath.rotz(base_euler_angle[2])
        self.donkey_position = np.asarray([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
        self.donkey_velocity = np.asarray([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
        self.donkey_angular_velocity = np.asarray([msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z])
        self.dq_curr_base = [float(msg.twist.twist.linear.x), float(msg.twist.twist.linear.y), float(msg.twist.twist.angular.z)]

#    def read_base_states_cb(self, msg):
#        try:
#            (trans,rot) = self.tf_listener.lookupTransform('/vicon/world', 'vicon/chonk/CHONK', rospy.Time(0))
#        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
#            print("error: cannot find vicon data!!!!")
#        self.base_euler_angle = tf.transformations.euler_from_quaternion([rot[0], rot[1], rot[2], rot[3]])
#        self.q_curr_base = np.asarray([trans[0], trans[1], self.base_euler_angle[2]])
#        self.donkey_R = optas.spatialmath.rotz(self.base_euler_angle[2])

#        self.donkey_position = np.asarray([trans[0], trans[1], trans[2]])
#        self.donkey_velocity = self.donkey_R @ np.asarray([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
#        self.donkey_angular_velocity = self.donkey_R @ np.asarray([msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z/6.05])
#        self.dq_curr_base = np.asarray([self.donkey_velocity[0], self.donkey_velocity[1], self.donkey_angular_velocity[2]])

    def read_mux_selection(self, msg):
        self._correct_mux_selection = (msg.data == self._pub_cmd_topic_name)

    def preempt_cb(self):
        rospy.loginfo("%s: Preempted.", self._name)
        # set the action state to preempted
        self._action_server.set_preempted()

    def BC(self, n, i):
        return np.math.factorial(n)/(np.math.factorial(i) * (np.math.factorial(n-i)))

    def skew(self, vec):
        A = np.asarray([ [0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]])
        return A

    def qaQb(self, a, b):
        Quaternion_result = optas.casadi.SX(np.zeros(4))
        Quaternion_result[0] = a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1]
        Quaternion_result[1] = a[3] * b[1] + a[1] * b[3] + a[2] * b[0] - a[0] * b[2]
        Quaternion_result[2] = a[3] * b[2] + a[2] * b[3] + a[0] * b[1] - a[1] * b[0]
        Quaternion_result[3] = a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]

        return Quaternion_result

    def qaConjugateQb_numpy(self, a, b):
        Quaternion_result = np.zeros(4)
        Quaternion_result[0] = a[3] * b[0] - a[0] * b[3] - a[1] * b[2] + a[2] * b[1]
        Quaternion_result[1] = a[3] * b[1] - a[1] * b[3] - a[2] * b[0] + a[0] * b[2]
        Quaternion_result[2] = a[3] * b[2] - a[2] * b[3] - a[0] * b[1] + a[1] * b[0]
        Quaternion_result[3] = a[3] * b[3] + a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

        return Quaternion_result

    def quatToRotationX(self, quaternion):
        A = optas.casadi.SX(np.zeros(3))
        A[0] = 1 - 2 * (quaternion[1]**2 + quaternion[2]**2)
        A[1] = 2 * (quaternion[0] * quaternion[1] + quaternion[3] * quaternion[2])
        A[2] = 2 * (quaternion[0] * quaternion[2] - quaternion[3] * quaternion[1])
        return A

    def quatToRotationY(self, quaternion):
        A = optas.casadi.SX(np.zeros(3))
        A[0] = 2 * (quaternion[0] * quaternion[1] - quaternion[3] * quaternion[2])
        A[1] = 1 - 2 * (quaternion[0]**2 + quaternion[2]**2)
        A[2] = 2 * (quaternion[1] * quaternion[2] + quaternion[3] * quaternion[0])
        return A

    def quatToRotationZ(self, quaternion):
        A = optas.casadi.SX(np.zeros(3))
        A[0] = 2 * (quaternion[0] * quaternion[2] + quaternion[3] * quaternion[1])
        A[1] = 2 * (quaternion[1] * quaternion[2] - quaternion[3] * quaternion[0])
        A[2] = 1 - 2 * (quaternion[0]**2 + quaternion[1]**2)
        return A

    def qaQb(self, a, b):
        Quaternion_result = optas.casadi.SX(np.zeros(4))
        Quaternion_result[0] = a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1]
        Quaternion_result[1] = a[3] * b[1] + a[1] * b[3] + a[2] * b[0] - a[0] * b[2]
        Quaternion_result[2] = a[3] * b[2] + a[2] * b[3] + a[0] * b[1] - a[1] * b[0]
        Quaternion_result[3] = a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]
        return Quaternion_result

    def qaConjugateQb_numpy(self, a, b):
        Quaternion_result = np.zeros(4)
        Quaternion_result[0] = a[3] * b[0] - a[0] * b[3] - a[1] * b[2] + a[2] * b[1]
        Quaternion_result[1] = a[3] * b[1] - a[1] * b[3] - a[2] * b[0] + a[0] * b[2]
        Quaternion_result[2] = a[3] * b[2] - a[2] * b[3] - a[0] * b[1] + a[1] * b[0]
        Quaternion_result[3] = a[3] * b[3] + a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

        return Quaternion_result

    def angular_velocity_to_quaternionRate(self, quaternion):
        A = 0.5* np.asarray([ [  quaternion[3],  quaternion[2], -quaternion[1] ],
                              [ -quaternion[2],  quaternion[3],  quaternion[0] ],
                              [  quaternion[1], -quaternion[0],  quaternion[3] ],
                              [ -quaternion[0], -quaternion[1], -quaternion[2] ]
                            ])
#        A = optas.casadi.SX(np.zeros((4, 3)))
#        A[0,0] =  quaternion[3]; A[0,1] =  quaternion[2]; A[0,2] = -quaternion[1];
#        A[1,0] = -quaternion[2]; A[1,1] =  quaternion[3]; A[1,2] =  quaternion[0];
#        A[2,0] =  quaternion[1]; A[2,1] = -quaternion[0]; A[2,2] =  quaternion[3];
#        A[3,0] = -quaternion[0]; A[3,1] = -quaternion[1]; A[3,2] = -quaternion[2];
        return A


if __name__=="__main__":
    # Initialize node
    rospy.init_node("cmd_pose_server_MPC_BC_operational", anonymous=True)
    # Initialize node class
    cmd_pose_server = CmdPoseActionServer(rospy.get_name())
    # executing node
    rospy.spin()
