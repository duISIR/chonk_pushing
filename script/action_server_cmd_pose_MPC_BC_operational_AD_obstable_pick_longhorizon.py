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
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import Wrench
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension
from pushing_msgs.msg import CmdChonkPoseForceAction, CmdChonkPoseForceFeedback, CmdChonkPoseForceResult
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from urdf_parser_py.urdf import URDF
import tf
from X_fromRandP import X_fromRandP, X_fromRandP_different

# For mux controller name
from std_msgs.msg import String
# service for selecting the controller
from topic_tools.srv import MuxSelect

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
            name='chonk_wholebodyMPC'
        )
        self.wholebodyMPC_planner_name = self.wholebodyMPC_planner.get_name()
        self.dt_MPC_planner = 0.12 # time step
        self.T_MPC_planner = 10 # T is number of time steps
        self.duration_MPC_planner = float(self.T_MPC_planner-1)*self.dt_MPC_planner
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
        R_Right = builder_wholebodyMPC_planner.add_decision_variables('R_Right', 3, self.T_MPC_planner)
        R_Left = builder_wholebodyMPC_planner.add_decision_variables('R_Left', 3, self.T_MPC_planner)
        R_middle = builder_wholebodyMPC_planner.add_decision_variables('R_middle', 3, self.T_MPC_planner)
        r_ep = builder_wholebodyMPC_planner.add_decision_variables('r_ep', self.T_MPC_planner)

        t = builder_wholebodyMPC_planner.add_parameter('t', self.T_MPC_planner)  # time
        self.n_planner = self.T_MPC_planner -1 # N in Bezier curve
        # Add parameters
#        init_r_position_middle = builder_wholebodyMPC_planner.add_parameter('init_r_position_middle', 2)
        init_r_position_Right = builder_wholebodyMPC_planner.add_parameter('init_r_position_Right', 2)
        init_r_position_Left = builder_wholebodyMPC_planner.add_parameter('init_r_position_Left', 2)

        # get end-effector pose as parameters
        pos_R = builder_wholebodyMPC_planner.add_parameter('pos_R', 3, self.T_MPC_planner)
        pos_L = builder_wholebodyMPC_planner.add_parameter('pos_L', 3, self.T_MPC_planner)

        # define q function depending on P
        r_middle_var_MPC = optas.casadi.SX(np.zeros((3, self.T_MPC_planner)))
        r_RARM_var_MPC = optas.casadi.SX(np.zeros((3, self.T_MPC_planner)))
        r_LARM_var_MPC = optas.casadi.SX(np.zeros((3, self.T_MPC_planner)))

        for i in range(self.T_MPC_planner):
            for j in range(self.T_MPC_planner):
                r_middle_var_MPC[:, i] += self.BC(self.n_planner, j) * t[i]**j * (1-t[i])**(self.n_planner-j) * R_middle[:, j]
                r_RARM_var_MPC[:, i] += self.BC(self.n_planner, j) * t[i]**j * (1-t[i])**(self.n_planner-j) * R_Right[:, j]
                r_LARM_var_MPC[:, i] += self.BC(self.n_planner, j) * t[i]**j * (1-t[i])**(self.n_planner-j) * R_Left[:, j]
            # optimization cost: close to target
            builder_wholebodyMPC_planner.add_cost_term('Right_middle_x' + str(i),  optas.sumsqr(r_middle_var_MPC[0, i] - 0.5*(pos_R[0, i] + pos_L[0, i])))
            builder_wholebodyMPC_planner.add_cost_term('Right_middle_y' + str(i),  10*optas.sumsqr(r_middle_var_MPC[1, i] - 0.5*(pos_R[1, i] + pos_L[1, i])))

#            builder_wholebodyMPC_planner.add_cost_term('Right_arm_obstacle_x' + str(i),  optas.sumsqr(r_RARM_var_MPC[0, i] - pos_R[0, i]))
#            builder_wholebodyMPC_planner.add_cost_term('Right_arm_obstacle_y' + str(i),  optas.sumsqr(r_RARM_var_MPC[1, i] - pos_R[1, i]))
#            builder_wholebodyMPC_planner.add_cost_term('Left_arm_obstacle_x' + str(i),  optas.sumsqr(r_LARM_var_MPC[0, i] - pos_L[0, i]))
#            builder_wholebodyMPC_planner.add_cost_term('Left_arm_obstacle_y' + str(i),   optas.sumsqr(r_LARM_var_MPC[1, i] - pos_L[1, i]))

#            builder_wholebodyMPC_planner.add_cost_term('mini_two_arm_pos_x_difference' + str(i),  optas.sumsqr(r_RARM_var_MPC[0, i] - r_LARM_var_MPC[0, i]))

#            builder_wholebodyMPC_planner.add_equality_constraint('two_arm_x_position' + str(i),  lhs=r_RARM_var_MPC[0, i], rhs=r_LARM_var_MPC[0, i])

        builder_wholebodyMPC_planner.add_equality_constraint('Right_middle_z', r_middle_var_MPC[2, :], rhs=0.5*(pos_R[2, :]+pos_L[2, :]))

        builder_wholebodyMPC_planner.add_equality_constraint('Right_arm_xy', r_RARM_var_MPC[0:2, :], rhs=r_middle_var_MPC[0:2, :] - 0.5*(pos_L[0:2, :]-pos_R[0:2, :]))
        builder_wholebodyMPC_planner.add_equality_constraint('Left_arm_xy', r_LARM_var_MPC[0:2, :], rhs=r_middle_var_MPC[0:2, :] + 0.5*(pos_L[0:2, :]-pos_R[0:2, :]))

        builder_wholebodyMPC_planner.add_equality_constraint('Right_arm_z', r_RARM_var_MPC[2, :], rhs=r_middle_var_MPC[2, :])
        builder_wholebodyMPC_planner.add_equality_constraint('Left_arm_z', r_LARM_var_MPC[2, :], rhs=r_middle_var_MPC[2, :])
#        builder_wholebodyMPC_planner.add_equality_constraint('Right_arm_z', R_Right[2, :], rhs=pos_R[2, :])
#        builder_wholebodyMPC_planner.add_equality_constraint('Left_arm_z', R_Left[2, :], rhs=pos_L[2, :])

        for i in range(self.T_MPC_planner):
            obstacle_pos = np.asarray([[4], [0]])
            obstacle_radius = 0.5 + 1
            builder_wholebodyMPC_planner.add_geq_inequality_constraint('middle_obstacle' + str(i), lhs=(r_middle_var_MPC[0:2, i]-obstacle_pos).T @ (r_middle_var_MPC[0:2, i]-obstacle_pos), rhs=obstacle_radius**2 + r_ep[i])

#            builder_wholebodyMPC_planner.add_geq_inequality_constraint('RARM_ee_obstacle' + str(i), lhs=(r_RARM_var_MPC[0:2, i]-obstacle_pos).T @ (r_RARM_var_MPC[0:2, i]-obstacle_pos), rhs=obstacle_radius_right**2 + r_right_ep[i])
#            builder_wholebodyMPC_planner.add_geq_inequality_constraint('LARM_ee_obstacle' + str(i), lhs=(r_LARM_var_MPC[0:2, i]-obstacle_pos).T @ (r_LARM_var_MPC[0:2, i]-obstacle_pos), rhs=obstacle_radius_left**2 + r_left_ep[i])

        builder_wholebodyMPC_planner.add_cost_term('minimize_r_ep',  optas.sumsqr(r_ep))

#        builder_wholebodyMPC_planner.add_cost_term('minimize_r_right_ep',  optas.sumsqr(r_right_ep))
#        builder_wholebodyMPC_planner.add_cost_term('minimize_r_left_ep',  optas.sumsqr(r_left_ep))

        # add position constraint at the beginning state
        builder_wholebodyMPC_planner.add_equality_constraint('init_r_middle', R_middle[0:2, 0], rhs=0.5*(init_r_position_Right + init_r_position_Left))
#        builder_wholebodyMPC_planner.add_equality_constraint('init_r_Right', R_Right[0:2, 0], rhs=init_r_position_Right)
#        builder_wholebodyMPC_planner.add_equality_constraint('init_r_Left', R_Left[0:2, 0], rhs=init_r_position_Left)


        dr_middle_var_MPC = optas.casadi.SX(np.zeros((3, self.T_MPC_planner)))
#        dr_RARM_var_MPC = optas.casadi.SX(np.zeros((3, self.T_MPC_planner)))
#        dr_LARM_var_MPC = optas.casadi.SX(np.zeros((3, self.T_MPC_planner)))

        w_dr = 0.0001/float(self.T_MPC_planner)
        for i in range(self.T_MPC_planner):
            for j in range(self.T_MPC_planner-1):
                dr_middle_var_MPC[:, i] += self.BC(self.n_planner-1, j) * t[i]**j * (1-t[i])**(self.n_planner-1-j) * self.n_planner * (R_middle[:, j+1] -  R_middle[:, j])
#                dr_RARM_var_MPC[:, i] += self.BC(self.n_planner-1, j) * t[i]**j * (1-t[i])**(self.n_planner-1-j) * self.n_planner * (R_Right[:, j+1] -  R_Right[:, j])
#                dr_LARM_var_MPC[:, i] += self.BC(self.n_planner-1, j) * t[i]**j * (1-t[i])**(self.n_planner-1-j) * self.n_planner * (R_Left[:, j+1] -  R_Left[:, j])
            builder_wholebodyMPC_planner.add_cost_term('minimize_dr_middle' + str(i), w_dr * optas.sumsqr(dr_middle_var_MPC[:, i]))
#            builder_wholebodyMPC_planner.add_cost_term('minimize_dr_right' + str(i), w_dq * optas.sumsqr(dr_RARM_var_MPC[:, i]))
#            builder_wholebodyMPC_planner.add_cost_term('minimize_dr_left' + str(i), w_dq * optas.sumsqr(dr_LARM_var_MPC[:, i]))

        ddr_middle_var_MPC = optas.casadi.SX(np.zeros((3, self.T_MPC_planner)))
        ddr_RARM_var_MPC = optas.casadi.SX(np.zeros((3, self.T_MPC_planner)))
        ddr_LARM_var_MPC = optas.casadi.SX(np.zeros((3, self.T_MPC_planner)))
        w_ddr = 0.0005/float(self.T_MPC_planner)
        for i in range(self.T_MPC_planner):
            for j in range(self.T_MPC_planner-2):
                ddr_middle_var_MPC[:, i] += self.BC(self.n_planner-2, j) * t[i]**j * (1-t[i])**(self.n_planner-2-j) * self.n_planner * (self.n_planner-1)* (R_middle[:, j+2] -  2*R_middle[:, j+1] + R_middle[:, j])
#                ddr_RARM_var_MPC[:, i] += self.BC(self.n_planner-2, j) * t[i]**j * (1-t[i])**(self.n_planner-2-j) * self.n_planner * (self.n_planner-1)* (R_Right[:, j+2] -  2*R_Right[:, j+1] + R_Right[:, j])
#                ddr_LARM_var_MPC[:, i] += self.BC(self.n_planner-2, j) * t[i]**j * (1-t[i])**(self.n_planner-2-j) * self.n_planner * (self.n_planner-1)* (R_Left[:, j+2] -  2*R_Left[:, j+1] + R_Left[:, j])
            builder_wholebodyMPC_planner.add_cost_term('minimize_ddr_middle' + str(i), w_ddr * optas.sumsqr(ddr_middle_var_MPC[:, i]))
#            builder_wholebodyMPC_planner.add_cost_term('minimize_ddr_right' + str(i), w_ddr * optas.sumsqr(ddr_RARM_var_MPC[:, i]))
#            builder_wholebodyMPC_planner.add_cost_term('minimize_ddr_left' + str(i), w_ddr * optas.sumsqr(ddr_LARM_var_MPC[:, i]))

        # setup solver
        self.solver_wholebodyMPC_planner = optas.CasADiSolver(optimization=builder_wholebodyMPC_planner.build()).setup('knitro', solver_options={
                                                                                                       'knitro.OutLev': 0, 'print_time': 0,
                                                                                                       'knitro.FeasTol': 5e-4, 'knitro.OptTol': 5e-4, 'knitro.ftol':1e-6,
                                                                                                       'knitro.algorithm':1, 'knitro.linsolver':2,
#                                                                                                       'knitro.maxtime_real': 4.0e-3,
                                                                                                       'knitro.bar_initpt':3, 'knitro.bar_murule':4, 'knitro.bar_penaltycons': 1,
                                                                                                       'knitro.bar_penaltyrule':2, 'knitro.bar_switchrule':2, 'knitro.linesearch': 1
                                                                                                       } )
        self.solution_MPC_planner = None
        self.time_linspace_planner = np.linspace(0., self.duration_MPC_planner, self.T_MPC_planner)
        self.timebyT_planner = np.asarray(self.time_linspace_planner)/self.duration_MPC_planner
        ### ---------------------------------------------------------
        # set up whole-body MPC
        wholebodyMPC_LIMITS = optas.RobotModel(
            urdf_string=self._robot_description,
            time_derivs=[0, 1],
            param_joints=[],
            name='chonk_wholebodyMPC_LIMITS'
        )
        self.wholebodyMPC = optas.RobotModel(
            urdf_string=self._robot_description,
            time_derivs=[0],
            param_joints=['base_joint_1', 'base_joint_2', 'base_joint_3', 'CHEST_JOINT0', 'HEAD_JOINT0', 'HEAD_JOINT1', 'LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2', 'LARM_JOINT3', 'LARM_JOINT4', 'LARM_JOINT5', 'RARM_JOINT0', 'RARM_JOINT1', 'RARM_JOINT2', 'RARM_JOINT3', 'RARM_JOINT4', 'RARM_JOINT5'],
            name='chonk_wholebodyMPC'
        )
        lower, upper = wholebodyMPC_LIMITS.get_limits(time_deriv=0)
        dlower, dupper = wholebodyMPC_LIMITS.get_limits(time_deriv=1)
        self.wholebodyMPC_name = self.wholebodyMPC.get_name()
        self.dt_MPC = 0.12 # time step
        self.T_MPC = 8 # T is number of time steps
        self.duration_MPC = float(self.T_MPC-1)*self.dt_MPC
        # nominal robot configuration
        self.wholebodyMPC_opt_idx = self.wholebodyMPC.optimized_joint_indexes
        self.wholebodyMPC_param_idx = self.wholebodyMPC.parameter_joint_indexes
        # set up optimization builder.
        builder_wholebodyMPC = optas.OptimizationBuilder(T=1, robots=[self.wholebodyMPC])
        builder_wholebodyMPC._decision_variables = optas.sx_container.SXContainer()
        builder_wholebodyMPC._parameters = optas.sx_container.SXContainer()
        builder_wholebodyMPC._lin_eq_constraints = optas.sx_container.SXContainer()
        builder_wholebodyMPC._lin_ineq_constraints = optas.sx_container.SXContainer()
        builder_wholebodyMPC._ineq_constraints = optas.sx_container.SXContainer()
        builder_wholebodyMPC._eq_constraints = optas.sx_container.SXContainer()
        # get robot state variables, get velocity state variables
        Q = builder_wholebodyMPC.add_decision_variables('Q', self.ndof, self.T_MPC)
        P_Right = builder_wholebodyMPC.add_decision_variables('P_Right', 3, self.T_MPC)
        P_Left = builder_wholebodyMPC.add_decision_variables('P_Left', 3, self.T_MPC)

        t = builder_wholebodyMPC.add_parameter('t', self.T_MPC)  # time
        self.n = self.T_MPC -1 # N in Bezier curve
        # Add parameters
        init_position_MPC = builder_wholebodyMPC.add_parameter('init_position_MPC', self.ndof)  # initial robot position
        init_velocity_MPC = builder_wholebodyMPC.add_parameter('init_velocity_MPC', self.ndof)  # initial robot velocity
        init_Delta_position_Right = builder_wholebodyMPC.add_parameter('init_Delta_position_Right', 3)
        init_Delta_position_Left = builder_wholebodyMPC.add_parameter('init_Delta_position_Left', 3)

        inertia_Right = builder_wholebodyMPC.add_parameter('inertia_Right', 3, 3)  # inertia Right parameter
        inertia_Left = builder_wholebodyMPC.add_parameter('inertia_Left', 3, 3)  # inertia Left parameter

        self.Lambda_ee_Right = np.diag([2, 2, 2])
        self.Lambda_ee_Left = np.diag([2, 2, 2])
        self.Lambda_ee_1DOF_Right = 2
        self.Lambda_ee_1DOF_Left = 2
        self.K_Right = np.diag([1000, 1000, 1000]) # Stiffness Right
        self.K_Left = np.diag([1000, 1000, 1000]) # Stiffness Left
        self.D_Right = np.diag([2 * np.sqrt(2*self.K_Right[0,0]), 2 * np.sqrt(2*self.K_Right[1,1]), 2 * np.sqrt(2*self.K_Right[2,2])]) # Damping Right
        self.D_Left = np.diag([2 * np.sqrt(2*self.K_Left[0,0]), 2 * np.sqrt(2*self.K_Left[1,1]), 2 * np.sqrt(2*self.K_Left[2,2])]) # Damping Left

        # get end-effector pose as parameters
        pos_R_reasonal = optas.casadi.SX(np.zeros((3, self.T_MPC)))
        pos_L_reasonal = optas.casadi.SX(np.zeros((3, self.T_MPC)))
        R_Right_reasonal = builder_wholebodyMPC.add_parameter('R_Right_reasonal', 3, self.T_MPC_planner)
        R_Left_reasonal = builder_wholebodyMPC.add_parameter('R_Left_reasonal', 3, self.T_MPC_planner)
        ori_R_reasonal = builder_wholebodyMPC.add_parameter('ori_R_reasonal', 4, self.T_MPC)
        ori_L_reasonal = builder_wholebodyMPC.add_parameter('ori_L_reasonal', 4, self.T_MPC)
        F_ext_Right_target = builder_wholebodyMPC.add_parameter('F_ext_Right_target', 3, self.T_MPC)
        F_ext_Left_target = builder_wholebodyMPC.add_parameter('F_ext_Left_target', 3, self.T_MPC)

        # functions of right and left arm positions
        pos_fnc_Right = self.wholebodyMPC.get_global_link_position_function(link=self._link_ee_right)
        pos_fnc_Left = self.wholebodyMPC.get_global_link_position_function(link=self._link_ee_left)
        # quaternion functions of two arm end effectors
        ori_fnc_Right = self.wholebodyMPC.get_global_link_quaternion_function(link=self._link_ee_right)
        ori_fnc_Left = self.wholebodyMPC.get_global_link_quaternion_function(link=self._link_ee_left)
        # define q function depending on P
        q_var_MPC = optas.casadi.SX(np.zeros((self.ndof, self.T_MPC)))
        Delta_p_Right_var_MPC = optas.casadi.SX(np.zeros((3, self.T_MPC)))
        Delta_p_Left_var_MPC = optas.casadi.SX(np.zeros((3, self.T_MPC)))
        dDelta_p_Right_var_MPC = optas.casadi.SX(np.zeros((3, self.T_MPC)))
        dDelta_p_Left_var_MPC = optas.casadi.SX(np.zeros((3, self.T_MPC)))
        ddDelta_p_Right_var_MPC = optas.casadi.SX(np.zeros((3, self.T_MPC)))
        ddDelta_p_Left_var_MPC = optas.casadi.SX(np.zeros((3, self.T_MPC)))
        F_ext_Right_var_MPC = optas.casadi.SX(np.zeros((3, self.T_MPC)))
        F_ext_Left_var_MPC = optas.casadi.SX(np.zeros((3, self.T_MPC)))

        pos_ee_INsensorFrame_Right = np.asarray([-0.08, 0, 0.039+0.04])
        pos_ee_INsensorFrame_Left = np.asarray([-0.08, 0, 0.039+0.04])
        rot_ee_INsensorFrame_Right = optas.spatialmath.roty(-np.pi/2)
        rot_ee_INsensorFrame_Left = optas.spatialmath.roty(-np.pi/2)

        self.X_ee_INsensorFrame_Right = X_fromRandP(rot_ee_INsensorFrame_Right, pos_ee_INsensorFrame_Right)
        self.X_ee_INsensorFrame_Left = X_fromRandP(rot_ee_INsensorFrame_Left, pos_ee_INsensorFrame_Left)

        self.rot_ee_right_fnc_global = self.wholebodyMPC.get_global_link_rotation_function(link=self._link_ee_right)
        self.rot_ee_left_fnc_global = self.wholebodyMPC.get_global_link_rotation_function(link=self._link_ee_left)
        self.pos_mass_INee_Right = np.asarray([-0.039-0.04+0.047729, 0, -0.08+0.017732])
        self.pos_mass_INee_Left = np.asarray([-0.039-0.04+0.047729, 0, -0.08+0.017732])

        self.mass_ee_Force = np.asarray([0,0,0,0,0,-0.3113*9.8])

        fac = self.duration_MPC/self.duration_MPC_planner

        for i in range(self.T_MPC):
            for j in range(self.T_MPC_planner):
                pos_R_reasonal[:, i] += self.BC(self.n_planner, j) * (t[i]*fac)**j * (1-t[i]*fac)**(self.n_planner-j) * R_Right_reasonal[:, j]
                pos_L_reasonal[:, i] += self.BC(self.n_planner, j) * (t[i]*fac)**j * (1-t[i]*fac)**(self.n_planner-j) * R_Left_reasonal[:, j]

        for i in range(self.T_MPC):
            for j in range(self.T_MPC):
                q_var_MPC[:, i] += self.BC(self.n, j) * t[i]**j * (1-t[i])**(self.n-j) * Q[:, j]
                Delta_p_Right_var_MPC += self.BC(self.n, j) * t[i]**j * (1-t[i])**(self.n-j) * P_Right[:, j]
                Delta_p_Left_var_MPC += self.BC(self.n, j) * t[i]**j * (1-t[i])**(self.n-j) * P_Left[:, j]
            for j in range(self.T_MPC-1):
                dDelta_p_Right_var_MPC[:, i] += self.BC(self.n-1, j) * t[i]**j * (1-t[i])**(self.n-1-j) * self.n * (P_Right[:, j+1] -  P_Right[:, j])
                dDelta_p_Left_var_MPC[:, i] += self.BC(self.n-1, j) * t[i]**j * (1-t[i])**(self.n-1-j) * self.n * (P_Left[:, j+1] -  P_Left[:, j])
            for j in range(self.T_MPC-2):
                ddDelta_p_Right_var_MPC[:, i] += self.BC(self.n-2, j) * t[i]**j * (1-t[i])**(self.n-2-j) * self.n * (self.n-1)* (P_Right[:, j+2] -  2*P_Right[:, j+1] + P_Right[:, j])
                ddDelta_p_Left_var_MPC[:, i] += self.BC(self.n-2, j) * t[i]**j * (1-t[i])**(self.n-2-j) * self.n * (self.n-1)* (P_Left[:, j+2] -  2*P_Left[:, j+1] + P_Left[:, j])
            F_ext_Right_var_MPC[:, i] = inertia_Right @ ddDelta_p_Right_var_MPC[:, i] + self.K_Right @ Delta_p_Right_var_MPC[:, i] + self.D_Right @ dDelta_p_Right_var_MPC[:, i]
            F_ext_Left_var_MPC[:, i] = inertia_Left @ ddDelta_p_Left_var_MPC[:, i] + self.K_Left @ Delta_p_Left_var_MPC[:, i] + self.D_Left @ dDelta_p_Left_var_MPC[:, i]
            builder_wholebodyMPC.add_bound_inequality_constraint('control_point_' + str(i) + '_bound', lhs=lower, mid=Q[:, i], rhs=upper)


           # optimization cost: close to target
            builder_wholebodyMPC.add_cost_term('Right_arm orientation' + str(i), optas.sumsqr(ori_fnc_Right(q_var_MPC[:, i])-ori_R_reasonal[:, i]))
            builder_wholebodyMPC.add_cost_term('Left_arm orientation' + str(i),  optas.sumsqr(ori_fnc_Left(q_var_MPC[:, i])-ori_L_reasonal[:, i]))
            builder_wholebodyMPC.add_cost_term('Right_arm position AD' + str(i), optas.sumsqr(pos_fnc_Right(q_var_MPC[:, i])-pos_R_reasonal[:, i] - Delta_p_Right_var_MPC[:, i]))
            builder_wholebodyMPC.add_cost_term('Left_arm position AD' + str(i),  optas.sumsqr(pos_fnc_Left(q_var_MPC[:, i])-pos_L_reasonal[:, i] - Delta_p_Left_var_MPC[:, i]))

            builder_wholebodyMPC.add_cost_term('Right_arm Force' + str(i), optas.sumsqr(F_ext_Right_var_MPC[:, i] - F_ext_Right_target[:, i]))
            builder_wholebodyMPC.add_cost_term('Left_arm Force' + str(i),  optas.sumsqr(F_ext_Left_var_MPC[:, i] - F_ext_Left_target[:, i]))

            builder_wholebodyMPC.add_cost_term('twoarm_miniscope' + str(i), 0.01 * optas.sumsqr(q_var_MPC[6, i]+q_var_MPC[12, i]))
            builder_wholebodyMPC.add_cost_term('head_miniscope' + str(i), 0.01 * optas.sumsqr(q_var_MPC[3, i]))
            builder_wholebodyMPC.add_cost_term('arm_joint_miniscope' + str(i), 0.002 * optas.sumsqr(q_var_MPC[6:self.ndof, i]))
            builder_wholebodyMPC.add_cost_term('donkey_yaw_miniscope' + str(i), 0.1 * optas.sumsqr(q_var_MPC[2, i]))

            if(i<(self.T_MPC -1)):
                builder_wholebodyMPC.add_cost_term('distance' + str(i), 0.001 * optas.sumsqr(Q[:, i+1] - Q[:, i]))
        # add position constraint at the beginning state
        builder_wholebodyMPC.add_equality_constraint('init_position', Q[0:4, 0], rhs=init_position_MPC[0:4])
        builder_wholebodyMPC.add_equality_constraint('init_position2', Q[6:self.ndof, 0], rhs=init_position_MPC[6:self.ndof])
        builder_wholebodyMPC.add_equality_constraint('head_miniscope', Q[4:6, :], rhs=np.zeros((2, self.T_MPC)))

        builder_wholebodyMPC.add_equality_constraint('Delta_p_Right_var_MPC_non_motion_direction_x', P_Right[0, :], rhs=np.zeros((1, self.T_MPC)))
        builder_wholebodyMPC.add_equality_constraint('Delta_p_Right_var_MPC_non_motion_direction_z', P_Right[2, :], rhs=np.zeros((1, self.T_MPC)))
        builder_wholebodyMPC.add_equality_constraint('Delta_p_Left_var_MPC_non_motion_direction_x', P_Left[0, :], rhs=np.zeros((1, self.T_MPC)))
        builder_wholebodyMPC.add_equality_constraint('Delta_p_Left_var_MPC_non_motion_direction_z', P_Left[2, :], rhs=np.zeros((1, self.T_MPC)))

        builder_wholebodyMPC.add_equality_constraint('init_Delta_position_Right_constraint', P_Right[:, 0], rhs=init_Delta_position_Right)
        builder_wholebodyMPC.add_equality_constraint('init_Delta_position_Left_constraint', P_Left[:, 0], rhs=init_Delta_position_Right)

        dq_var_MPC = optas.casadi.SX(np.zeros((self.ndof, self.T_MPC)))
        w_dq = 0.0001/float(self.T_MPC)
        for i in range(self.T_MPC):
            for j in range(self.T_MPC-1):
                dq_var_MPC[:, i] += self.BC(self.n-1, j) * t[i]**j * (1-t[i])**(self.n-1-j) * self.n * (Q[:, j+1] -  Q[:, j])
            if(i<(self.T_MPC -1)):
                builder_wholebodyMPC.add_bound_inequality_constraint('control_point_deriv_' + str(i) + '_bound', lhs=dlower, mid=self.n * (Q[:, i+1] -  Q[:, i]), rhs=dupper)
            builder_wholebodyMPC.add_cost_term('minimize_velocity_dq' + str(i), w_dq * optas.sumsqr(dq_var_MPC[:, i]))
            builder_wholebodyMPC.add_cost_term('minimize_dDelta_p_Right' + str(i), w_dq * optas.sumsqr(dDelta_p_Right_var_MPC[:, i]))
            builder_wholebodyMPC.add_cost_term('minimize_dDelta_p_Left' + str(i), w_dq * optas.sumsqr(dDelta_p_Left_var_MPC[:, i]))

        ddq_var_MPC = optas.casadi.SX(np.zeros((self.ndof, self.T_MPC)))
        w_ddq = 0.0005/float(self.T_MPC)
        for i in range(self.T_MPC):
            for j in range(self.T_MPC-2):
                ddq_var_MPC[:, i] += self.BC(self.n-2, j) * t[i]**j * (1-t[i])**(self.n-2-j) * self.n * (self.n-1)* (Q[:, j+2] -  2*Q[:, j+1] + Q[:, j])
            builder_wholebodyMPC.add_cost_term('minimize_acceleration' + str(i), w_ddq * optas.sumsqr(ddq_var_MPC[:, i]))
            builder_wholebodyMPC.add_cost_term('minimize_ddDelta_p_Right' + str(i), w_ddq * optas.sumsqr(ddDelta_p_Right_var_MPC[:, i]))
            builder_wholebodyMPC.add_cost_term('minimize_ddDelta_p_Left' + str(i), w_ddq * optas.sumsqr(ddDelta_p_Left_var_MPC[:, i]))

        # setup solver
        self.solver_wholebodyMPC = optas.CasADiSolver(optimization=builder_wholebodyMPC.build()).setup('knitro', solver_options={
                                                                                                       'knitro.OutLev': 0, 'print_time': 0,
                                                                                                       'knitro.FeasTol': 1e-4, 'knitro.OptTol': 1e-4, 'knitro.ftol':1e-6,
                                                                                                       'knitro.algorithm':1, 'knitro.linsolver':2,
                                                                                                       'knitro.maxtime_real': 1.8e-2,
                                                                                                       'knitro.bar_initpt':3, 'knitro.bar_murule':4, 'knitro.bar_penaltycons': 1,
                                                                                                       'knitro.bar_penaltyrule':2, 'knitro.bar_switchrule':2, 'knitro.linesearch': 1
                                                                                                       } )
        self.solution_MPC = None
        self.time_linspace = np.linspace(0., self.duration_MPC, self.T_MPC)
        self.timebyT = np.asarray(self.time_linspace)/self.duration_MPC

        self.start_RARM_force = np.zeros(3)
#        self.start_RARM_torque = np.zeros(3)
        self.start_LARM_force = np.zeros(3)
#        self.start_LARM_torque = np.zeros(3)
        ### ---------------------------------------------------------
        # declare ft_sensor subscriber
        self._ft_right_sub = rospy.Subscriber("/ft_right/raw/data", WrenchStamped, self.read_ft_sensor_right_data_cb)
        self._ft_left_sub = rospy.Subscriber("/ft_left/raw/data", WrenchStamped, self.read_ft_sensor_left_data_cb)
        # declare joint subscriber
        self._joint_sub = rospy.Subscriber("/chonk/joint_states", JointState, self.read_joint_states_cb)
#        self._joint_sub_base = rospy.Subscriber("/chonk/donkey_velocity_controller/odom", Odometry, self.read_base_states_cb)
        self._joint_sub_base = rospy.Subscriber("/chonk/base_pose_ground_truth", Odometry, self.read_base_states_cb)
        # declare joint publisher
        self._joint_pub = rospy.Publisher("/chonk/streaming_controller/command", Float64MultiArray, queue_size=10)
        # This is for donkey_velocity_controller
        self._joint_pub_velocity = rospy.Publisher("/chonk/donkey_velocity_controller/cmd_vel", Twist, queue_size=10)
        # declare two arm ee grasp inertia publisher
        self._inertia_sub = rospy.Subscriber("/chonk/arm_ee_inertias", Float64MultiArray, self.read_two_ee_grasp_inertia_data_cb)
        self._inertia_1DOF_sub = rospy.Subscriber("/chonk/arm_ee_1DOF_inertias_1DOF", Float64MultiArray, self.read_two_ee_grasp_inertia_1DOF_data_cb)
        # set mux controller selection as wrong by default
        self._correct_mux_selection = False
        # declare mux service
        self._srv_mux_sel = rospy.ServiceProxy(rospy.get_namespace() + '/mux_joint_position/select', MuxSelect)
        # declare subscriber for selected controller
        self._sub_selected_controller = rospy.Subscriber("/mux_selected", String, self.read_mux_selection)
        # initialize action messages
        self._feedback = CmdChonkPoseForceFeedback()
        self._result = CmdChonkPoseForceResult()
        # declare action server
        self._action_server = actionlib.SimpleActionServer('cmd_pose', CmdChonkPoseForceAction, execute_cb=None, auto_start=False)
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
        pos_Right = np.asarray([acceped_goal.poseR.position.x, acceped_goal.poseR.position.y, acceped_goal.poseR.position.z])
        pos_Left = np.asarray([acceped_goal.poseL.position.x, acceped_goal.poseL.position.y, acceped_goal.poseL.position.z])
        ori_Right = np.asarray([acceped_goal.poseR.orientation.x, acceped_goal.poseR.orientation.y, acceped_goal.poseR.orientation.z, acceped_goal.poseR.orientation.w])
        ori_Left = np.asarray([acceped_goal.poseL.orientation.x, acceped_goal.poseL.orientation.y, acceped_goal.poseL.orientation.z, acceped_goal.poseL.orientation.w])
        self.force_Right = np.asarray([acceped_goal.ForceTorqueR.force.x, acceped_goal.ForceTorqueR.force.y, acceped_goal.ForceTorqueR.force.z])
#        torque_Right = np.asarray([acceped_goal.ForceTorqueR.torque.x, acceped_goal.ForceTorqueR.torque.y, acceped_goal.ForceTorqueR.torque.z])
        self.force_Left = np.asarray([acceped_goal.ForceTorqueL.force.x, acceped_goal.ForceTorqueL.force.y, acceped_goal.ForceTorqueL.force.z])
#        torque_Left = np.asarray([acceped_goal.ForceTorqueL.torque.x, acceped_goal.ForceTorqueL.torque.y, acceped_goal.ForceTorqueL.torque.z])
        # print goal request
        rospy.loginfo("%s: Request to send right arm to position (%.2f, %.2f, %.2f) with orientation (%.2f, %.2f, %.2f, %.2f), and left arm to position (%.2f, %.2f, %.2f) with orientation (%.2f, %.2f, %.2f, %.2f) in %.1f seconds." % (
                self._name, pos_Right[0], pos_Right[1], pos_Right[2], ori_Right[0], ori_Right[1], ori_Right[2], ori_Right[3],
                pos_Left[0], pos_Left[1], pos_Left[2], ori_Left[0], ori_Left[1], ori_Left[2], ori_Left[3], acceped_goal.duration))
        # read current robot joint positions
        self.q_curr = np.concatenate((self.q_curr_base, self.q_curr_joint), axis=None)
        self.dq_curr = np.concatenate((self.dq_curr_base, self.dq_curr_joint), axis=None)
        qT = np.zeros(self.ndof)
        self.joint_names = self.joint_names_base + self.joint_names_position
        ### ---------------------------------------------------------
        # get two-arm end effector trajectory in the operational space
        q0 = self.q_curr.T
        self.duration = acceped_goal.duration
        self._steps = int(self.duration * self._freq)
        self._idx = 0
        # current right and left arm end effector position and quaternion
        self.start_RARM_quat = np.asarray(self.wholebodyMPC.get_global_link_quaternion(link=self._link_ee_right, q=q0)).T[0]
        self.start_RARM_pos = np.asarray(self.wholebodyMPC.get_global_link_position(link=self._link_ee_right, q=q0)).T[0]
        self.start_LARM_quat = np.asarray(self.wholebodyMPC.get_global_link_quaternion(link=self._link_ee_left, q=q0)).T[0]
        self.start_LARM_pos = np.asarray(self.wholebodyMPC.get_global_link_position(link=self._link_ee_left, q=q0)).T[0]
        # derivation of right and left arm end effector position and quaternion compared with the beginning ee position and quaternion
        Derivation_RARM_Pos = pos_Right - self.start_RARM_pos
        Derivation_RARM_Quat = ori_Right - self.start_RARM_quat
        Derivation_LARM_Pos = pos_Left - self.start_LARM_pos
        Derivation_LARM_Quat = ori_Left - self.start_LARM_quat
        Derivation_RARM_force = self.force_Right - self.start_RARM_force
#        Derivation_RARM_torque = torque_Right - self.start_RARM_torque
        Derivation_LARM_force = self.force_Left - self.start_LARM_force
#        Derivation_LARM_torque = torque_Left - self.start_LARM_torque
        # interpolate between current and target position polynomial obtained for zero speed (3rd order) and acceleratin (5th order) at the initial and final time
        self._RARM_ee_Pos_trajectory = lambda t: self.start_RARM_pos + (10.*((t/self.duration)**3) - 15.*((t/self.duration)**4) + 6.*((t/self.duration)**5))*Derivation_RARM_Pos # 5th order
        self._LARM_ee_Pos_trajectory = lambda t: self.start_LARM_pos + (10.*((t/self.duration)**3) - 15.*((t/self.duration)**4) + 6.*((t/self.duration)**5))*Derivation_LARM_Pos # 5th order
        # interpolate between current and target quaternion polynomial obtained for zero speed (3rd order) and acceleratin (5th order) at the initial and final time
        self._RARM_ee_Quat_trajectory = lambda t: self.start_RARM_quat + (10.*((t/self.duration)**3) - 15.*((t/self.duration)**4) + 6.*((t/self.duration)**5))*Derivation_RARM_Quat # 5th order
        self._LARM_ee_Quat_trajectory = lambda t: self.start_LARM_quat + (10.*((t/self.duration)**3) - 15.*((t/self.duration)**4) + 6.*((t/self.duration)**5))*Derivation_LARM_Quat # 5th order
        # interpolate between zero and target force polynomail obtained for zero speed (3rd order) and acceleration (5th order) at the initial and final time
        self._RARM_ee_force_trajectory = lambda t: self.start_RARM_force + (10.*((t/self.duration)**3) - 15.*((t/self.duration)**4) + 6.*((t/self.duration)**5))*Derivation_RARM_force # 5th order
#        self._RARM_ee_torque_trajectory = lambda t: self.start_RARM_torque + (10.*((t/self.duration)**3) - 15.*((t/self.duration)**4) + 6.*((t/self.duration)**5))*Derivation_RARM_torque # 5th order
        self._LARM_ee_force_trajectory = lambda t: self.start_LARM_force + (10.*((t/self.duration)**3) - 15.*((t/self.duration)**4) + 6.*((t/self.duration)**5))*Derivation_LARM_force # 5th order
#        self._LARM_ee_torque_trajectory = lambda t: self.start_LARM_torque + (10.*((t/self.duration)**3) - 15.*((t/self.duration)**4) + 6.*((t/self.duration)**5))*Derivation_LARM_torque # 5th order


#        self.start_RARM_force = self._RARM_ee_force_trajectory(self.duration)
#        self.start_RARM_torque = self._RARM_ee_torque_trajectory(self.duration)
#        self.start_LARM_force = self._LARM_ee_force_trajectory(self.duration)
#        self.start_LARM_torque = self._LARM_ee_torque_trajectory(self.duration)

        self._t = np.linspace(0., self.duration, self._steps + 1)
        ### ---------------------------------------------------------
        # initialize the message
        self._msg = Float64MultiArray()
        self._msg.layout = MultiArrayLayout()
        self._msg.layout.data_offset = 0
        self._msg.layout.dim.append(MultiArrayDimension())
        self._msg.layout.dim[0].label = "columns"
        self._msg.layout.dim[0].size = self.ndof_position_control

        self._msg_velocity = Twist()
        self._msg_velocity.linear.x  = 0
        self._msg_velocity.linear.y  = 0
        self._msg_velocity.linear.z  = 0
        self._msg_velocity.angular.x = 0
        self._msg_velocity.angular.y = 0
        self._msg_velocity.angular.z = 0

        # create timer
        dur = rospy.Duration(1.0/self._freq)
        self._timer = rospy.Timer(dur, self.timer_cb)

        self.curr_MPC = np.zeros((self.ndof, self.T_MPC))
        for i in range(self.T_MPC):
            self.curr_MPC[:,i] = self.q_curr

    def timer_cb(self, event):
        """Sensor data in ee frame"""
#        ft_ee_right = self.X_ee_INsensorFrame_Right.T @ self.ft_right
#        ft_ee_left = self.X_ee_INsensorFrame_Left.T @ self.ft_left

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
                pos_R_goal = []
                pos_L_goal = []

                for i in range(self.T_MPC_planner):
                    if(self.ti_MPC_planner <= self.duration):
                        self.ti_MPC_planner = self._t[self._idx-1]  + self.dt_MPC_planner*i
                    if(self.ti_MPC_planner > self.duration):
                        self.ti_MPC_planner = self.duration
                    try:
                        g_rarm_ee_pos = self._RARM_ee_Pos_trajectory(self.ti_MPC_planner).flatten()
                        pos_R_goal.append(g_rarm_ee_pos.tolist())
                        g_larm_ee_pos = self._LARM_ee_Pos_trajectory(self.ti_MPC_planner).flatten()
                        pos_L_goal.append(g_larm_ee_pos.tolist())
                    except ValueError:
                        pos_R_goal.append(g_rarm_ee_pos.tolist()) # i.e. previous goal
                        pos_L_goal.append(g_larm_ee_pos.tolist()) # i.e. previous goal
                pos_R_goal = optas.np.array(pos_R_goal).T
                pos_L_goal = optas.np.array(pos_L_goal).T

                # read current robot joint positions
                self.q_curr = np.concatenate((self.q_curr_base, self.q_curr_joint), axis=None)
                self.dq_curr = np.concatenate((self.dq_curr_base, self.dq_curr_joint), axis=None)

                r_actual_Right = np.array(self.wholebodyMPC.get_global_link_position(link=self._link_ee_right, q=self.q_curr))[0:2]
                r_actual_Left = np.array(self.wholebodyMPC.get_global_link_position(link=self._link_ee_left, q=self.q_curr))[0:2]
                ### -----------------------------------------------------------
                ### optas. Solve the whole-body MPC
                # set initial seed
                if self.solution_MPC_planner is None:
                    self.solver_wholebodyMPC_planner.reset_initial_seed({f'R_middle': 0.5*(pos_R_goal+pos_L_goal), f'R_Right': pos_R_goal,
                                                                         f'R_Left': pos_L_goal, f'r_ep': np.zeros(self.T_MPC_planner) })
                # set initial seed
                if self.solution_MPC_planner is not None:
                    self.solver_wholebodyMPC_planner.reset_initial_seed({f'R_middle': self.solution_MPC_planner[f'R_middle'],
                                                                         f'R_Right': self.solution_MPC_planner[f'R_Right'],
                                                                         f'R_Left': self.solution_MPC_planner[f'R_Left'],
                                                                         f'r_ep': self.solution_MPC_planner[f'r_ep'] })

                self.solver_wholebodyMPC_planner.reset_parameters({'pos_R': pos_R_goal, 'pos_L': pos_L_goal,
                                                                   't': self.timebyT_planner,
                                                                   'init_r_position_Right': r_actual_Right,
                                                                   'init_r_position_Left': r_actual_Left} )
                # solve problem
                self.solution_MPC_planner = self.solver_wholebodyMPC_planner.opt.decision_variables.vec2dict(self.solver_wholebodyMPC_planner._solve())
                R_Right = np.asarray(self.solution_MPC_planner[f'R_Right'])
                R_Left = np.asarray(self.solution_MPC_planner[f'R_Left'])
                ### -----------------------------------------------------------
                self.ti_MPC = 0
                force_R_goal = []
                force_L_goal = []
                ori_R_goal = []
                ori_L_goal = []
                for i in range(self.T_MPC):
                    if(self.ti_MPC <= self.duration):
                        self.ti_MPC = self._t[self._idx-1]  + self.dt_MPC*i
                    if(self.ti_MPC > self.duration):
                        self.ti_MPC = self.duration
                    try:
                        g_rarm_ee_force = self._RARM_ee_force_trajectory(self.ti_MPC).flatten()
                        force_R_goal.append(g_rarm_ee_force.tolist())
                        g_larm_ee_force = self._LARM_ee_force_trajectory(self.ti_MPC).flatten()
                        force_L_goal.append(g_larm_ee_force.tolist())
                        g_rarm_ee_ori = self._RARM_ee_Quat_trajectory(self.ti_MPC_planner).flatten()
                        g_rarm_ee_ori[0] = np.sqrt(1-g_rarm_ee_ori[1]**2-g_rarm_ee_ori[2]**2-g_rarm_ee_ori[3]**2)
                        ori_R_goal.append(g_rarm_ee_ori.tolist())
                        g_larm_ee_ori = self._LARM_ee_Quat_trajectory(self.ti_MPC_planner).flatten()
                        g_larm_ee_ori[0] = np.sqrt(1-g_larm_ee_ori[1]**2-g_larm_ee_ori[2]**2-g_larm_ee_ori[3]**2)
                        ori_L_goal.append(g_larm_ee_ori.tolist())
                    except ValueError:
                        force_R_goal.append(g_rarm_ee_force.tolist()) # i.e. previous goal
                        force_L_goal.append(g_larm_ee_force.tolist()) # i.e. previous goal
                        ori_L_goal.append(g_larm_ee_ori.tolist()) # i.e. previous goal
                        ori_R_goal.append(g_rarm_ee_ori.tolist()) # i.e. previous goal

                force_R_goal = optas.np.array(force_R_goal).T
                force_L_goal = optas.np.array(force_L_goal).T
                ori_R_goal = optas.np.array(ori_R_goal).T
                ori_L_goal = optas.np.array(ori_L_goal).T

                # read current robot joint positions
                self.q_curr = np.concatenate((self.q_curr_base, self.q_curr_joint), axis=None)
                self.dq_curr = np.concatenate((self.dq_curr_base, self.dq_curr_joint), axis=None)

                Derivation_RARM_pos_start = np.zeros(3)
                Derivation_LARM_pos_start = np.zeros(3)
                Derivation_RARM_pos_start[1] = np.asarray(self.wholebodyMPC.get_global_link_position(link=self._link_ee_right, q=self.q_curr)).T[0][1] - R_Right[1, 0]
                Derivation_LARM_pos_start[1] = np.asarray(self.wholebodyMPC.get_global_link_position(link=self._link_ee_left, q=self.q_curr)).T[0][1] - R_Left[1, 0]



                ### optas. Solve the whole-body MPC
                # set initial seed
                if self.solution_MPC is None:
                    self.solver_wholebodyMPC.reset_initial_seed({f'Q': self.curr_MPC,
                                                                 f'P_Right': np.zeros((3, self.T_MPC)),
                                                                 f'P_Left': np.zeros((3, self.T_MPC))  })
                # set initial seed
                if self.solution_MPC is not None:
                    self.solver_wholebodyMPC.reset_initial_seed({ f'Q': self.solution_MPC[f'Q'],
                                                                  f'P_Right': self.solution_MPC[f'P_Right'],
                                                                  f'P_Left': self.solution_MPC[f'P_Left'] })


                self.solver_wholebodyMPC.reset_parameters({'ori_R_reasonal': ori_R_goal, 'ori_L_reasonal': ori_L_goal,
                                                           't': self.timebyT,
                                                           'init_position_MPC': self.q_curr, 'init_velocity_MPC': self.dq_curr,
                                                           'F_ext_Right_target': force_R_goal, 'F_ext_Left_target': force_L_goal,
                                                           'inertia_Right': self.Lambda_ee_Right, 'inertia_Left': self.Lambda_ee_Left,
                                                           'init_Delta_position_Right': Derivation_RARM_pos_start,
                                                           'init_Delta_position_Left': Derivation_LARM_pos_start,
                                                           'R_Right_reasonal': R_Right,
                                                           'R_Left_reasonal': R_Left } )
#                self.solver_wholebodyMPC.reset_parameters({'pos_R': pos_R_goal, 'ori_R': ori_R_goal, 'pos_L': pos_L_goal, 'ori_L': ori_L_goal, 't': self.timebyT} )

#                print(self.Lambda_ee_Right)
                # solve problem
                self.solution_MPC = self.solver_wholebodyMPC.opt.decision_variables.vec2dict(self.solver_wholebodyMPC._solve())
                Q = np.asarray(self.solution_MPC[f'Q'])
#                P_Right = np.asarray(self.solution_MPC[f'P_Right'])
#                P_Left = np.asarray(self.solution_MPC[f'P_Left'])
#                R_Right = np.asarray(self.solution_MPC[f'R_Right'])
#                R_Left = np.asarray(self.solution_MPC[f'R_Left'])


#                print('---------------------------------------------------------------------------------------------------------------------------')
#                print(P_Right[1,:])
                ### ---------------------------------------------------------
                # compute next configuration with lambda function
                t = (1./self._freq)/self.duration_MPC
                n = self.T_MPC -1
                q_next = np.zeros(self.ndof)
                p_right = np.zeros(3)
                dp_right = np.zeros(3)
                ddp_right = np.zeros(3)
                for j in range(self.T_MPC):
                    q_next += self.BC(n, j) * t**j * (1-t)**(n-j) * Q[:, j]
#                for j in range(self.T_MPC):
#                    p_right += self.BC(n, j) * t**j * (1-t)**(n-j) * P_Right[:, j]
#                for j in range(self.T_MPC-1):
#                    dp_right += self.BC(self.n-1, j) * t**j * (1-t)**(self.n-1-j) * self.n * (P_Right[:, j+1] -  P_Right[:, j])
#                for j in range(self.T_MPC-2):
#                    ddp_right += self.BC(self.n-2, j) * t**j * (1-t)**(self.n-2-j) * self.n * (self.n-1)* (P_Right[:, j+2] -  2*P_Right[:, j+1] + P_Right[:, j])

#                print(self.Lambda_ee_Right @ ddp_right + self.K_Right @ p_right + self.D_Right @ dp_right)

                dq_next = np.zeros(self.ndof)
                for j in range(self.T_MPC-1):
                    dq_next += self.BC(n-1, j) * t**j * (1-t)**(n-1-j) * n * (Q[:, j+1] -  Q[:, j])
                # compute the donkey velocity in its local frame
                Global_w_b = np.asarray([0., 0., dq_next[2]])
                Global_v_b = np.asarray([dq_next[0], dq_next[1], 0.])
                Local_w_b = self.donkey_R.T @ Global_w_b
                Local_v_b = self.donkey_R.T @ Global_v_b
                # update message
                self._msg.data = q_next[-self.ndof_position_control:]
                self._msg_velocity.linear.x = Local_v_b[0]
                self._msg_velocity.linear.y = Local_v_b[1]
                self._msg_velocity.angular.z = Local_w_b[2]
                # publish message
                self._joint_pub.publish(self._msg)
                self._joint_pub_velocity.publish(self._msg_velocity)
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

            self.start_RARM_force = self._RARM_ee_force_trajectory(self.duration)
            self.start_LARM_force = self._LARM_ee_force_trajectory(self.duration)

            return

    def read_joint_states_cb(self, msg):
        self.q_curr_joint = np.asarray(list(msg.position)[:self.ndof_position_control])
        self.joint_names_position = msg.name[:self.ndof_position_control]
        self.dq_curr_joint = np.asarray(list(msg.velocity)[:self.ndof_position_control])

    def read_ft_sensor_right_data_cb(self, msg):
        """ paranet to child: the force/torque from robot to ee"""
        self.ft_right = -np.asarray([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z, msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])

    def read_ft_sensor_left_data_cb(self, msg):
        """ paranet to child: the force/torque from robot to ee"""
        self.ft_left = -np.asarray([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z, msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])

    def read_base_states_cb(self, msg):
        base_euler_angle = tf.transformations.euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        self.q_curr_base = [msg.pose.pose.position.x, msg.pose.pose.position.y, base_euler_angle[2]]
        self.donkey_R = optas.spatialmath.rotz(base_euler_angle[2])
        self.donkey_position = np.asarray([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
        self.donkey_velocity = np.asarray([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
        self.donkey_angular_velocity = np.asarray([msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z])
        self.dq_curr_base = [float(msg.twist.twist.linear.x), float(msg.twist.twist.linear.y), float(msg.twist.twist.angular.z)]

    def read_two_ee_grasp_inertia_data_cb(self, msg):
        self.Lambda_ee_Right = np.asarray([ [msg.data[0], msg.data[1], msg.data[2]],
                                            [msg.data[1], msg.data[3], msg.data[4]],
                                            [msg.data[2], msg.data[4], msg.data[5]] ])
        self.Lambda_ee_Left = np.asarray([  [msg.data[6], msg.data[7], msg.data[8]],
                                            [msg.data[7], msg.data[9], msg.data[10]],
                                            [msg.data[8], msg.data[10], msg.data[11]] ])


    def read_two_ee_grasp_inertia_1DOF_data_cb(self, msg):
        self.Lambda_ee_1DOF_Right = np.asarray([ msg.data[0] ])
        self.Lambda_ee_1DOF_Left = np.asarray([  msg.data[1] ])
        self.Lambda_ee_Right[1, 1] = msg.data[0]
        self.Lambda_ee_Left[1, 1] = msg.data[1]

    def read_mux_selection(self, msg):
        self._correct_mux_selection = (msg.data == self._pub_cmd_topic_name)

    def preempt_cb(self):
        rospy.loginfo("%s: Preempted.", self._name)
        # set the action state to preempted
        self._action_server.set_preempted()

    def BC(self, n, i):
        return np.math.factorial(n)/(np.math.factorial(i) * (np.math.factorial(n-i)))

if __name__=="__main__":
    # Initialize node
    rospy.init_node("cmd_pose_server_MPC_BC_operational_AD", anonymous=True)
    # Initialize node class
    cmd_pose_server = CmdPoseActionServer(rospy.get_name())
    # executing node
    rospy.spin()
