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
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


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
        self.q_curr = np.zeros(self.ndof); self.q_curr_joint = np.zeros(self.ndof_position_control); self.q_curr_base = np.zeros(self.ndof_base);
        self.dq_curr = np.zeros(self.ndof); self.dq_curr_joint = np.zeros(self.ndof_position_control); self.dq_curr_base = np.zeros(self.ndof_base);
        self.joint_names_position = []; self.joint_names_base = ['base_joint_1', 'base_joint_2', 'base_joint_3'];
        self.donkey_R =np.zeros((3,3)); self.donkey_position =np.zeros(3); self.donkey_velocity =np.zeros(3); self.donkey_angular_velocity =np.zeros(3);
        ### optas
        ### ---------------------------------------------------------
        # set up whole-body MPC
        wholebodyMPC_LIMITS = optas.RobotModel(urdf_string=self._robot_description, time_derivs=[0, 1], param_joints=[], name='chonk_wholebodyMPC_LIMITS')
        self.wholebodyMPC = optas.RobotModel(
            urdf_string=self._robot_description, time_derivs=[0],
            param_joints=['base_joint_1', 'base_joint_2', 'base_joint_3', 'CHEST_JOINT0', 'HEAD_JOINT0', 'HEAD_JOINT1',
                          'LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2', 'LARM_JOINT3', 'LARM_JOINT4', 'LARM_JOINT5',
                          'RARM_JOINT0', 'RARM_JOINT1', 'RARM_JOINT2', 'RARM_JOINT3', 'RARM_JOINT4', 'RARM_JOINT5'], name='chonk_wholebodyMPC' )
        lower, upper = wholebodyMPC_LIMITS.get_limits(time_deriv=0)
        dlower, dupper = wholebodyMPC_LIMITS.get_limits(time_deriv=1)
        self.wholebodyMPC_name = self.wholebodyMPC.get_name()
        self.dt_MPC = 0.1 # time step
        self.T_MPC = 7 # T is number of time steps
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
        self.Derivation_RARM_pos_start = np.zeros(3)
        self.Derivation_LARM_pos_start = np.zeros(3)

        self.m_ee_r = 0.3113;
        self.m_ee_l = 0.3113;
        stiffness = 3000;

        inertia_Right = builder_wholebodyMPC.add_parameter('inertia_Right', 3, 3)  # inertia Right parameter
        inertia_Left = builder_wholebodyMPC.add_parameter('inertia_Left', 3, 3)  # inertia Left parameter
        self.K_Right = np.diag([stiffness, stiffness, stiffness]) # Stiffness Right
        self.K_Left = np.diag([stiffness, stiffness, stiffness]) # Stiffness Left
        self.D_Right = np.diag([2 * np.sqrt(self.m_ee_r*self.K_Right[0,0]), 2 * np.sqrt(self.m_ee_r*self.K_Right[1,1]), 2 * np.sqrt(self.m_ee_r*self.K_Right[2,2])]) # Damping Right
        self.D_Left = np.diag([2 * np.sqrt(self.m_ee_l*self.K_Left[0,0]), 2 * np.sqrt(self.m_ee_l*self.K_Left[1,1]), 2 * np.sqrt(self.m_ee_l*self.K_Left[2,2])]) # Damping Left
        ###################################################################################
        i_xx=0.00064665; i_xy=0; i_xz=0.000297068; i_yy=0.00082646; i_yz=0; i_zz=0.000354023;
        self.sensor_p_ee_r = np.array([-0.01773,  0,  0.04772])
        self.sensor_I_angular_ee_r = np.asarray([[i_xx, i_xy, i_xz], [i_xy, i_yy, i_yz], [i_xz, i_yz, i_zz]])
        self.sensor_I_ee_r_conventional = np.zeros((6,6))
        self.sensor_I_ee_r_conventional[0:3, 0:3] = self.sensor_I_angular_ee_r + self.m_ee_r * self.skew(self.sensor_p_ee_r) @ self.skew(self.sensor_p_ee_r).T
        self.sensor_I_ee_r_conventional[0:3, 3:6] = self.m_ee_r * self.skew(self.sensor_p_ee_r)
        self.sensor_I_ee_r_conventional[3:6, 0:3] = self.m_ee_r * self.skew(self.sensor_p_ee_r).T
        self.sensor_I_ee_r_conventional[3:6, 3:6] = self.m_ee_r * np.identity(3)
        self.sensor_pos_ee_Right = np.array([-0.08, 0, 0.039+0.04])
        self.sensor_rot_ee_Right = optas.spatialmath.roty(-np.pi/2)
        self.sensor_X_ee_r_conventional = np.zeros((6,6))
        self.sensor_X_ee_r_conventional[0:3, 0:3] = self.sensor_rot_ee_Right
        self.sensor_X_ee_r_conventional[3:6, 0:3] = self.skew(self.sensor_pos_ee_Right) @ self.sensor_rot_ee_Right
        self.sensor_X_ee_r_conventional[3:6, 3:6] = self.sensor_rot_ee_Right
        self.I_ee_r_conventional = self.sensor_X_ee_r_conventional.T @ self.sensor_I_ee_r_conventional @ self.sensor_X_ee_r_conventional
        self.G_Rotation_ee_right = self.wholebodyMPC.get_global_link_rotation(link=self._link_ee_right, q=np.zeros(18))
        self.G_X_ee_right = np.identity(6)
        self.G_X_ee_right[0:3, 0:3] = self.G_Rotation_ee_right
        self.G_X_ee_right[3:6, 3:6] = self.G_Rotation_ee_right
        self.G_I_ee_r_conventional = self.G_X_ee_right @ self.I_ee_r_conventional @ self.G_X_ee_right.T

        self.sensor_p_ee_l = self.sensor_p_ee_r
        self.sensor_I_angular_ee_l = self.sensor_I_angular_ee_r
        self.sensor_I_ee_l_conventional = self.sensor_I_ee_r_conventional
        self.sensor_X_ee_l_conventional = self.sensor_X_ee_r_conventional
        self.I_ee_l_conventional = self.sensor_X_ee_l_conventional.T @ self.sensor_I_ee_l_conventional @ self.sensor_X_ee_l_conventional
        self.G_Rotation_ee_left = self.wholebodyMPC.get_global_link_rotation(link=self._link_ee_left, q=np.zeros(18))
        self.G_X_ee_left = np.identity(6)
        self.G_X_ee_left[0:3, 0:3] = self.G_Rotation_ee_left
        self.G_X_ee_left[3:6, 3:6] = self.G_Rotation_ee_left
        self.G_I_ee_l_conventional = self.G_X_ee_left @ self.I_ee_l_conventional @ self.G_X_ee_left.T
        #####################################################################################3
        # get end-effector pose as parameters
        pos_R = builder_wholebodyMPC.add_parameter('pos_R', 3, self.T_MPC)
        ori_R = builder_wholebodyMPC.add_parameter('ori_R', 4, self.T_MPC)
        pos_L = builder_wholebodyMPC.add_parameter('pos_L', 3, self.T_MPC)
        ori_L = builder_wholebodyMPC.add_parameter('ori_L', 4, self.T_MPC)

        ddpos_box_goal = builder_wholebodyMPC.add_parameter('ddpos_box_goal', 3, self.T_MPC)
        m_box = builder_wholebodyMPC.add_parameter('m_box', 1)
        #####################################################################################
        self.F_ext_global_Right = np.zeros(6); self.F_ext_global_Left = np.zeros(6);
        self.F_ext_local_Right = np.zeros(6);  self.F_ext_local_Left = np.zeros(6);
        self.acc_box = np.zeros((3, self.T_MPC));
#        self.acc_box = np.zeros(3);
        F_ext_Right_goal = builder_wholebodyMPC.add_parameter('F_ext_Right_goal', 3, self.T_MPC)
        F_ext_Left_goal = builder_wholebodyMPC.add_parameter('F_ext_Left_goal', 3, self.T_MPC)
        F_ext_Right_actual = builder_wholebodyMPC.add_parameter('F_ext_Right_actual', 3)
        F_ext_Left_actual = builder_wholebodyMPC.add_parameter('F_ext_Left_actual', 3)
        F_ext_Right_var_MPC = optas.casadi.SX(np.zeros((3, self.T_MPC)))
        F_ext_Left_var_MPC = optas.casadi.SX(np.zeros((3, self.T_MPC)))
        #####################################################################################
        # functions of right and left arm positions
        self.pos_fnc_Right = self.wholebodyMPC.get_global_link_position_function(link=self._link_ee_right)
        self.pos_fnc_Left = self.wholebodyMPC.get_global_link_position_function(link=self._link_ee_left)
        self.pos_Jac_fnc_Right = self.wholebodyMPC.get_global_link_linear_jacobian_function(link=self._link_ee_right)
        self.pos_Jac_fnc_Left = self.wholebodyMPC.get_global_link_linear_jacobian_function(link=self._link_ee_left)
        # quaternion functions of two arm end effectors
        self.ori_fnc_Right = self.wholebodyMPC.get_global_link_quaternion_function(link=self._link_ee_right)
        self.ori_fnc_Left = self.wholebodyMPC.get_global_link_quaternion_function(link=self._link_ee_left)
        self.rotation_fnc_Right = self.wholebodyMPC.get_global_link_rotation_function(link=self._link_ee_right)
        self.rotation_fnc_Left = self.wholebodyMPC.get_global_link_rotation_function(link=self._link_ee_left)
        #####################################################################################
        # define q function depending on P
        q_var_MPC = optas.casadi.SX(np.zeros((self.ndof, self.T_MPC)))
        #####################################################################################
        Delta_p_Right_var_MPC = optas.casadi.SX(np.zeros((3, self.T_MPC)))
        Delta_p_Left_var_MPC = optas.casadi.SX(np.zeros((3, self.T_MPC)))
        dDelta_p_Right_var_MPC = optas.casadi.SX(np.zeros((3, self.T_MPC)))
        dDelta_p_Left_var_MPC = optas.casadi.SX(np.zeros((3, self.T_MPC)))
        ddDelta_p_Right_var_MPC = optas.casadi.SX(np.zeros((3, self.T_MPC)))
        ddDelta_p_Left_var_MPC = optas.casadi.SX(np.zeros((3, self.T_MPC)))
        #####################################################################################
        for i in range(self.T_MPC):
            for j in range(self.T_MPC):
                q_var_MPC[:, i] += self.BC(self.n, j) * t[i]**j * (1-t[i])**(self.n-j) * Q[:, j]
                Delta_p_Right_var_MPC[:, i] += self.BC(self.n, j) * t[i]**j * (1-t[i])**(self.n-j) * P_Right[:, j]
                Delta_p_Left_var_MPC[:, i] += self.BC(self.n, j) * t[i]**j * (1-t[i])**(self.n-j) * P_Left[:, j]
            for j in range(self.T_MPC-1):
                dDelta_p_Right_var_MPC[:, i] += (1./self.duration_MPC) * self.BC(self.n-1, j) * t[i]**j * (1-t[i])**(self.n-1-j) * self.n * (P_Right[:, j+1] -  P_Right[:, j])
                dDelta_p_Left_var_MPC[:, i] += (1./self.duration_MPC) * self.BC(self.n-1, j) * t[i]**j * (1-t[i])**(self.n-1-j) * self.n * (P_Left[:, j+1] -  P_Left[:, j])
            for j in range(self.T_MPC-2):
                ddDelta_p_Right_var_MPC[:, i] += (1./self.duration_MPC)**2 * self.BC(self.n-2, j) * t[i]**j * (1-t[i])**(self.n-2-j) * self.n * (self.n-1)* (P_Right[:, j+2] -  2*P_Right[:, j+1] + P_Right[:, j])
                ddDelta_p_Left_var_MPC[:, i] += (1./self.duration_MPC)**2 * self.BC(self.n-2, j) * t[i]**j * (1-t[i])**(self.n-2-j) * self.n * (self.n-1)* (P_Left[:, j+2] -  2*P_Left[:, j+1] + P_Left[:, j])
            #####################################################################################
            F_ext_Right_var_MPC[:, i] = F_ext_Right_actual + inertia_Right @ ddDelta_p_Right_var_MPC[:, i] + self.K_Right @ Delta_p_Right_var_MPC[:, i] + self.D_Right @ dDelta_p_Right_var_MPC[:, i]
            F_ext_Left_var_MPC[:, i] = F_ext_Left_actual + inertia_Left @ ddDelta_p_Left_var_MPC[:, i] + self.K_Left @ Delta_p_Left_var_MPC[:, i] + self.D_Left @ dDelta_p_Left_var_MPC[:, i]
            builder_wholebodyMPC.add_bound_inequality_constraint('control_point_' + str(i) + '_bound', lhs=lower, mid=Q[:, i], rhs=upper)
            # optimization cost: close to target
            builder_wholebodyMPC.add_cost_term('Right_arm orientation' + str(i), 10*optas.sumsqr(self.ori_fnc_Right(q_var_MPC[:, i])-ori_R[:, i]))
            builder_wholebodyMPC.add_cost_term('Left_arm orientation' + str(i),  10*optas.sumsqr(self.ori_fnc_Left(q_var_MPC[:, i])-ori_L[:, i]))
            builder_wholebodyMPC.add_cost_term('Right_arm position AD' + str(i), 2*optas.sumsqr(self.pos_fnc_Right(q_var_MPC[:, i])-pos_R[:, i] - init_Delta_position_Right - Delta_p_Right_var_MPC[:, i]))
            builder_wholebodyMPC.add_cost_term('Left_arm position AD' + str(i),  2*optas.sumsqr(self.pos_fnc_Left(q_var_MPC[:, i])-pos_L[:, i]  - init_Delta_position_Left  - Delta_p_Left_var_MPC[:, i]))
            #####################################################################################
            builder_wholebodyMPC.add_cost_term('Right_arm Force world y' + str(i), 0.1*optas.sumsqr(F_ext_Right_var_MPC[1, i] - F_ext_Right_goal[1, i] - 0.5*m_box * ddpos_box_goal[1, i]))
            builder_wholebodyMPC.add_cost_term('Left_arm Force world y' + str(i),  0.1*optas.sumsqr(F_ext_Left_var_MPC[1, i] - F_ext_Left_goal[1, i] - 0.5*m_box * ddpos_box_goal[1, i]))
#            builder_wholebodyMPC.add_cost_term('two Force sum world y' + str(i),   optas.sumsqr(F_ext_Right_var_MPC[1, i] + F_ext_Left_var_MPC[1, i] - F_ext_Right_goal[1, i] - F_ext_Left_goal[1, i] ))
#            builder_wholebodyMPC.add_cost_term('Two arm ee addition motion equal' + str(i), optas.sumsqr(Delta_p_Right_var_MPC[1, i] + Delta_p_Left_var_MPC[1, i]))
            builder_wholebodyMPC.add_cost_term('Two force Delta for box inertial force' + str(i),  optas.sumsqr(F_ext_Right_var_MPC[1, i] + F_ext_Left_var_MPC[1, i] - m_box * ddpos_box_goal[1, i]))
#            builder_wholebodyMPC.add_bound_inequality_constraint('right_force_limit' + str(i) + '_bound', lhs=-50, mid=F_ext_Right_var_MPC[1, i], rhs=50)
#            builder_wholebodyMPC.add_bound_inequality_constraint('left_force_limit' + str(i) + '_bound', lhs=-50, mid=F_ext_Left_var_MPC[1, i], rhs=50)
            #####################################################################################
            builder_wholebodyMPC.add_cost_term('twoarm_miniscope' + str(i), 0.1 * optas.sumsqr(q_var_MPC[6, i] + q_var_MPC[12, i]))
            builder_wholebodyMPC.add_cost_term('chest_miniscope' + str(i),  0.1 * optas.sumsqr(q_var_MPC[3, i]))
            builder_wholebodyMPC.add_cost_term('arm_joint_miniscope' + str(i), 0.001 * optas.sumsqr(q_var_MPC[6:self.ndof, i]))
            builder_wholebodyMPC.add_cost_term('donkey_yaw_miniscope' + str(i), 0.1 * optas.sumsqr(q_var_MPC[2, i]))
            if(i<(self.T_MPC -1)):
                builder_wholebodyMPC.add_cost_term('joint_distance' + str(i), 0.05 * optas.sumsqr(Q[:, i+1] - Q[:, i]))
                builder_wholebodyMPC.add_cost_term('Right_force_distance' + str(i), 0.05 * optas.sumsqr(P_Right[:, i+1] - P_Right[:, i]))
                builder_wholebodyMPC.add_cost_term('Left_force_distance' + str(i), 0.05 * optas.sumsqr(P_Left[:, i+1] - P_Left[:, i]))
        #########################################################################################
        # add position constraint at the beginning state
        builder_wholebodyMPC.add_equality_constraint('init_position', Q[0:4, 0], rhs=init_position_MPC[0:4])
        builder_wholebodyMPC.add_equality_constraint('init_position2', Q[6:self.ndof, 0], rhs=init_position_MPC[6:self.ndof])
        builder_wholebodyMPC.add_equality_constraint('head_miniscope', Q[4:6, :], rhs=np.zeros((2, self.T_MPC)))
        builder_wholebodyMPC.add_equality_constraint('Delta_p_Right_var_MPC_non_motion_direction_x', P_Right[0, :], rhs=np.zeros((1, self.T_MPC)))
        builder_wholebodyMPC.add_equality_constraint('Delta_p_Right_var_MPC_non_motion_direction_z', P_Right[2, :], rhs=np.zeros((1, self.T_MPC)))
        builder_wholebodyMPC.add_equality_constraint('Delta_p_Left_var_MPC_non_motion_direction_x', P_Left[0, :], rhs=np.zeros((1, self.T_MPC)))
        builder_wholebodyMPC.add_equality_constraint('Delta_p_Left_var_MPC_non_motion_direction_z', P_Left[2, :], rhs=np.zeros((1, self.T_MPC)))
        builder_wholebodyMPC.add_equality_constraint('init_Delta_position_Right_constraint_y', P_Right[1, 0], rhs = 0 )
        builder_wholebodyMPC.add_equality_constraint('init_Delta_position_Left_constraint_y',  P_Left[1, 0],  rhs = 0 )
        #########################################################################################
        dq_var_MPC = optas.casadi.SX(np.zeros((self.ndof, self.T_MPC)))
        w_dq = 0.0001/float(self.T_MPC)
        for i in range(self.T_MPC):
            for j in range(self.T_MPC-1):
                dq_var_MPC[:, i] += (1./self.duration_MPC) * self.BC(self.n-1, j) * t[i]**j * (1-t[i])**(self.n-1-j) * self.n * (Q[:, j+1] -  Q[:, j])
            if(i<(self.T_MPC -1)):
                name = 'control_point_deriv_' + str(i) + '_bound'  # add velocity constraint for each Q[:, i]
                builder_wholebodyMPC.add_bound_inequality_constraint(name, lhs=dlower, mid=(1./self.duration_MPC) * self.n * (Q[:, i+1] -  Q[:, i]), rhs=dupper)
            builder_wholebodyMPC.add_cost_term('minimize_velocity' + str(i), w_dq * optas.sumsqr(dq_var_MPC[:, i]))
            builder_wholebodyMPC.add_cost_term('minimize_dDelta_p_Right' + str(i), w_dq * optas.sumsqr(dDelta_p_Right_var_MPC[:, i]))
            builder_wholebodyMPC.add_cost_term('minimize_dDelta_p_Left' + str(i), w_dq * optas.sumsqr(dDelta_p_Left_var_MPC[:, i]))
        #########################################################################################
        ddq_var_MPC = optas.casadi.SX(np.zeros((self.ndof, self.T_MPC)))
        w_ddq = 0.0005/float(self.T_MPC)
        ddlower = np.array([-1.5, -1.5, -1.5]); ddupper = np.array([1.5, 1.5, 1.5]);
        for i in range(self.T_MPC):
            for j in range(self.T_MPC-2):
                ddq_var_MPC[:, i] += (1./self.duration_MPC)**2 * self.BC(self.n-2, j) * t[i]**j * (1-t[i])**(self.n-2-j) * self.n * (self.n-1)* (Q[:, j+2] -  2*Q[:, j+1] + Q[:, j])
            if(i<(self.T_MPC -2)):
                name = 'control_point_dderiv_' + str(i) + '_bound'  # add acceleration constraint for Q[0:3, i]
                builder_wholebodyMPC.add_bound_inequality_constraint(name, lhs=ddlower, mid=(1./self.duration_MPC)**2 * self.n * (self.n - 1) * (Q[0:3, i+2]- 2*Q[0:3, i+1] +  Q[0:3, i]), rhs=ddupper)
            builder_wholebodyMPC.add_cost_term('minimize_acceleration' + str(i), w_ddq * optas.sumsqr(ddq_var_MPC[:, i]))
            builder_wholebodyMPC.add_cost_term('minimize_ddDelta_p_Right' + str(i), w_ddq * optas.sumsqr(ddDelta_p_Right_var_MPC[:, i]))
            builder_wholebodyMPC.add_cost_term('minimize_ddDelta_p_Left' + str(i), w_ddq * optas.sumsqr(ddDelta_p_Left_var_MPC[:, i]))
        #########################################################################################
        acc_box_var = builder_wholebodyMPC.add_decision_variables('acc_box_var', 3, self.T_MPC)
        q_var_MPC_for_box = optas.casadi.SX(np.zeros((self.ndof, self.T_MPC)))
        ddq_var_MPC_for_box = optas.casadi.SX(np.zeros((self.ndof, self.T_MPC)))
        t_loop = (1./self._freq)/self.duration_MPC
        for i in range(self.T_MPC):
            for j in range(self.T_MPC):
                q_var_MPC_for_box[:, i] += self.BC(self.n, j) * (t[i]+t_loop)**j * (1-t[i]-t_loop)**(self.n-j) * Q[:, j]
            for j in range(self.T_MPC-2):
                ddq_var_MPC_for_box[:, i] += (1./self.duration_MPC)**2 * self.BC(self.n-2, j) * (t[i]+t_loop)**j * (1-t[i]-t_loop)**(self.n-2-j) * self.n * (self.n-1)* (Q[:, j+2] -  2*Q[:, j+1] + Q[:, j])
            acc_box_var[:, i] = 0.5*(self.pos_Jac_fnc_Right(q_var_MPC_for_box[:, i]) + self.pos_Jac_fnc_Left(q_var_MPC_for_box[:, i])) @ ddq_var_MPC_for_box[:, i]

        #########################################################################################
        # setup solver
        self.solver_wholebodyMPC = optas.CasADiSolver(optimization=builder_wholebodyMPC.build()).setup('knitro')
        # self.solver_wholebodyMPC = optas.CasADiSolver(optimization=builder_wholebodyMPC.build()).setup('knitro', solver_options={'knitro.OutLev': 10} )
        # self.solver_wholebodyMPC = optas.CasADiSolver(optimization=builder_wholebodyMPC.build()).setup('knitro', solver_options={'knitro.OutLev': 0, 'print_time': 0} )
        self.solver_wholebodyMPC = optas.CasADiSolver(optimization=builder_wholebodyMPC.build()).setup('knitro', solver_options={
                                                                                                       'knitro.OutLev': 0,
                                                                                                       'print_time': 0,
                                                                                                       'knitro.FeasTol': 1e-5, 'knitro.OptTol': 1e-5,
                                                                                                       'knitro.ftol':1e-5, 'knitro.algorithm':1,
                                                                                                       'knitro.linsolver':2,
                                                                                                       'knitro.maxtime_real': 1.8e-2,
                                                                                                       'knitro.bar_initpt':3, 'knitro.bar_murule':4,
                                                                                                       'knitro.bar_penaltycons': 1, 'knitro.bar_penaltyrule':2,
                                                                                                       'knitro.bar_switchrule':2, 'knitro.linesearch': 1})
        #########################################################################################
        self.solution_MPC = None
        self.time_linspace = np.linspace(0., self.duration_MPC, self.T_MPC)
        self.timebyT = np.asarray(self.time_linspace)/self.duration_MPC

        self.start_RARM_force = np.zeros(3); self.start_RARM_torque = np.zeros(3);
        self.start_LARM_force = np.zeros(3); self.start_LARM_torque = np.zeros(3);
        #########################################################################################
        # initialize the message
        self._msg = Float64MultiArray()
        self._msg.layout = MultiArrayLayout()
        self._msg.layout.data_offset = 0
        self._msg.layout.dim.append(MultiArrayDimension())
        self._msg.layout.dim[0].label = "columns"
        self._msg.layout.dim[0].size = self.ndof_position_control

        # initialize the message
        self._msg_acceleration = Float64MultiArray()
        self._msg_acceleration.layout = MultiArrayLayout()
        self._msg_acceleration.layout.data_offset = 0
        self._msg_acceleration.layout.dim.append(MultiArrayDimension())
        self._msg_acceleration.layout.dim[0].label = "columns"
        self._msg_acceleration.layout.dim[0].size = self.ndof

        self._msg_velocity = Twist()
        self._msg_velocity.linear.x  = 0; self._msg_velocity.linear.y  = 0; self._msg_velocity.linear.z  = 0;
        self._msg_velocity.angular.x = 0; self._msg_velocity.angular.y = 0; self._msg_velocity.angular.z = 0;
        #########################################################################################
        self.eva_trajectory = JointTrajectory()
        self.eva_trajectory.header.frame_id = ''
        self.eva_trajectory.joint_names = ['CHEST_JOINT0', 'HEAD_JOINT0', 'HEAD_JOINT1',
                                           'LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2', 'LARM_JOINT3', 'LARM_JOINT4', 'LARM_JOINT5',
                                           'RARM_JOINT0', 'RARM_JOINT1', 'RARM_JOINT2', 'RARM_JOINT3', 'RARM_JOINT4', 'RARM_JOINT5']
        self.eva_point = JointTrajectoryPoint()
        self.eva_point.time_from_start = rospy.Duration(0.1)
        self.eva_trajectory.points.append(self.eva_point)
        ### ---------------------------------------------------------
        # declare ft_sensor subscriber
#        self._ft_right_sub = rospy.Subscriber("/ft_right/raw/data", WrenchStamped, self.read_ft_sensor_right_data_cb)
#        self._ft_left_sub = rospy.Subscriber("/ft_left/raw/data", WrenchStamped, self.read_ft_sensor_left_data_cb)
        # declare joint subscriber
        self._joint_sub = rospy.Subscriber("/chonk/joint_states", JointState, self.read_joint_states_cb)
#        self._joint_sub_base = rospy.Subscriber("/chonk/donkey_velocity_controller/odom", Odometry, self.read_base_states_cb)
        self._joint_sub_base = rospy.Subscriber("/chonk/base_pose_ground_truth", Odometry, self.read_base_states_cb)
        # declare joint publisher
#        self._joint_pub = rospy.Publisher(self._pub_cmd_topic_name, Float64MultiArray, queue_size=10)
        self._joint_pub = rospy.Publisher("/chonk/trajectory_controller/command", JointTrajectory, queue_size=10)
        # declare acceleration publisher for two arms
        self._joint_acc_pub = rospy.Publisher("/chonk/joint_acc_pub", Float64MultiArray, queue_size=10)
        # This is for donkey_velocity_controller
        self._joint_pub_velocity = rospy.Publisher("/chonk/donkey_velocity_controller/cmd_vel", Twist, queue_size=10)
        # declare two arm ee grasp actual force publisher
        self._sensor_ft_sub_right = rospy.Subscriber("/chonk/sensor_ft_right", Float64MultiArray, self.read_right_ee_grasp_ft_data_cb)
        self._sensor_ft_sub_left = rospy.Subscriber("/chonk/sensor_ft_left", Float64MultiArray, self.read_left_ee_grasp_ft_data_cb)
        # declare two arm ee grasp actual force publisher
#        self._acc_sub = rospy.Subscriber("/chonk/acc_box", Float64MultiArray, self.read_box_acc_data_cb)
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
        self.m_box = acceped_goal.m_box
        pos_Right = np.asarray([acceped_goal.poseR.position.x, acceped_goal.poseR.position.y, acceped_goal.poseR.position.z])
        pos_Left = np.asarray([acceped_goal.poseL.position.x, acceped_goal.poseL.position.y, acceped_goal.poseL.position.z])
        ori_Right = np.asarray([acceped_goal.poseR.orientation.x, acceped_goal.poseR.orientation.y, acceped_goal.poseR.orientation.z, acceped_goal.poseR.orientation.w])
        ori_Left = np.asarray([acceped_goal.poseL.orientation.x, acceped_goal.poseL.orientation.y, acceped_goal.poseL.orientation.z, acceped_goal.poseL.orientation.w])
        self.force_Right = np.asarray([acceped_goal.ForceTorqueR.force.x, acceped_goal.ForceTorqueR.force.y, acceped_goal.ForceTorqueR.force.z])
        self.torque_Right = np.asarray([acceped_goal.ForceTorqueR.torque.x, acceped_goal.ForceTorqueR.torque.y, acceped_goal.ForceTorqueR.torque.z])
        self.force_Left = np.asarray([acceped_goal.ForceTorqueL.force.x, acceped_goal.ForceTorqueL.force.y, acceped_goal.ForceTorqueL.force.z])
        self.torque_Left = np.asarray([acceped_goal.ForceTorqueL.torque.x, acceped_goal.ForceTorqueL.torque.y, acceped_goal.ForceTorqueL.torque.z])
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
        Derivation_RARM_Pos = pos_Right - self.start_RARM_pos; Derivation_RARM_Quat = ori_Right - self.start_RARM_quat;
        Derivation_LARM_Pos = pos_Left - self.start_LARM_pos;  Derivation_LARM_Quat = ori_Left - self.start_LARM_quat;
        Derivation_RARM_force = self.force_Right - self.start_RARM_force; Derivation_RARM_torque = self.torque_Right - self.start_RARM_torque;
        Derivation_LARM_force = self.force_Left - self.start_LARM_force; Derivation_LARM_torque = self.torque_Left - self.start_LARM_torque;
        # interpolate between current and target position polynomial obtained for zero speed (3rd order) and acceleratin (5th order) at the initial and final time
        self._RARM_ee_Pos_trajectory = lambda t: self.start_RARM_pos + (10.*((t/self.duration)**3) - 15.*((t/self.duration)**4) + 6.*((t/self.duration)**5))*Derivation_RARM_Pos # 5th order
        self._LARM_ee_Pos_trajectory = lambda t: self.start_LARM_pos + (10.*((t/self.duration)**3) - 15.*((t/self.duration)**4) + 6.*((t/self.duration)**5))*Derivation_LARM_Pos # 5th order
        self._RARM_ee_ddPos_trajectory = lambda t: (60.*((t/self.duration)/(self.duration**2)) - 180.*((t/self.duration)**2/(self.duration**2)) + 120.*((t/self.duration)**3/(self.duration**2)))*Derivation_RARM_Pos # 5th order
        self._LARM_ee_ddPos_trajectory = lambda t: (60.*((t/self.duration)/(self.duration**2)) - 180.*((t/self.duration)**2/(self.duration**2)) + 120.*((t/self.duration)**3/(self.duration**2)))*Derivation_LARM_Pos # 5th order
        # interpolate between current and target quaternion polynomial obtained for zero speed (3rd order) and acceleratin (5th order) at the initial and final time
        self._RARM_ee_Quat_trajectory = lambda t: self.start_RARM_quat + (10.*((t/self.duration)**3) - 15.*((t/self.duration)**4) + 6.*((t/self.duration)**5))*Derivation_RARM_Quat # 5th order
        self._LARM_ee_Quat_trajectory = lambda t: self.start_LARM_quat + (10.*((t/self.duration)**3) - 15.*((t/self.duration)**4) + 6.*((t/self.duration)**5))*Derivation_LARM_Quat # 5th order
        # interpolate between zero and target force polynomail obtained for zero speed (3rd order) and acceleration (5th order) at the initial and final time
        self._RARM_ee_force_trajectory = lambda t: self.start_RARM_force + (10.*((t/self.duration)**3) - 15.*((t/self.duration)**4) + 6.*((t/self.duration)**5))*Derivation_RARM_force # 5th order
        self._RARM_ee_torque_trajectory = lambda t: self.start_RARM_torque + (10.*((t/self.duration)**3) - 15.*((t/self.duration)**4) + 6.*((t/self.duration)**5))*Derivation_RARM_torque # 5th order
        self._LARM_ee_force_trajectory = lambda t: self.start_LARM_force + (10.*((t/self.duration)**3) - 15.*((t/self.duration)**4) + 6.*((t/self.duration)**5))*Derivation_LARM_force # 5th order
        self._LARM_ee_torque_trajectory = lambda t: self.start_LARM_torque + (10.*((t/self.duration)**3) - 15.*((t/self.duration)**4) + 6.*((t/self.duration)**5))*Derivation_LARM_torque # 5th order

        self._t = np.linspace(0., self.duration, self._steps + 1)
        ### ---------------------------------------------------------
        # create timer
        dur = rospy.Duration(1.0/self._freq)
        self._timer = rospy.Timer(dur, self.timer_cb)

        self.curr_MPC = np.zeros((self.ndof, self.T_MPC))
        for i in range(self.T_MPC):
            self.curr_MPC[:,i] = self.q_curr

    def timer_cb(self, event):
        # make sure that the action is active
        if(not self._action_server.is_active()):
            self._timer.shutdown()
            rospy.logwarn("%s: The action server is NOT active!")
            self._result.reached_goal = False
            self._action_server.set_aborted(self._result)
            return

        # main execution
        if(self._idx < self._steps):
            time = rospy.get_time()
            if(self._correct_mux_selection):
                # increment idx (in here counts starts with 1)
                self.ti_MPC = 0 # time index of the MPC
                self._idx += 1
                pos_R_goal = []; ori_R_goal = []; pos_L_goal = []; ori_L_goal = []; force_R_goal = []; force_L_goal = [];
#                ddpos_box_goal = []
                for i in range(self.T_MPC):
                    if(self.ti_MPC <= self.duration):
                        self.ti_MPC = self._t[self._idx-1]  + self.dt_MPC*i
                    if(self.ti_MPC > self.duration):
                        self.ti_MPC = self.duration
                    try:
                        g_rarm_ee_pos = self._RARM_ee_Pos_trajectory(self.ti_MPC).flatten()
                        pos_R_goal.append(g_rarm_ee_pos.tolist())
                        g_rarm_ee_ori = self._RARM_ee_Quat_trajectory(self.ti_MPC).flatten()
#                        g_rarm_ee_ori[0] = np.sqrt(1-g_rarm_ee_ori[1]**2-g_rarm_ee_ori[2]**2-g_rarm_ee_ori[3]**2)
                        ori_R_goal.append(g_rarm_ee_ori.tolist())
                        g_larm_ee_pos = self._LARM_ee_Pos_trajectory(self.ti_MPC).flatten()
                        pos_L_goal.append(g_larm_ee_pos.tolist())
                        g_larm_ee_ori = self._LARM_ee_Quat_trajectory(self.ti_MPC).flatten()
#                        g_larm_ee_ori[0] = np.sqrt(1-g_larm_ee_ori[1]**2-g_larm_ee_ori[2]**2-g_larm_ee_ori[3]**2)
                        ori_L_goal.append(g_larm_ee_ori.tolist())
                        g_rarm_ee_force = self._RARM_ee_force_trajectory(self.ti_MPC).flatten()
                        force_R_goal.append(g_rarm_ee_force.tolist())
                        g_larm_ee_force = self._LARM_ee_force_trajectory(self.ti_MPC).flatten()
                        force_L_goal.append(g_larm_ee_force.tolist())
#                        g_box_ddpos = self._RARM_ee_ddPos_trajectory(self.ti_MPC).flatten()
#                        ddpos_box_goal.append(g_box_ddpos.tolist())
                    except ValueError:
                        pos_R_goal.append(g_rarm_ee_pos.tolist()) # i.e. previous goal
                        ori_R_goal.append(g_rarm_ee_ori.tolist()) # i.e. previous goal
                        pos_L_goal.append(g_larm_ee_pos.tolist()) # i.e. previous goal
                        ori_L_goal.append(g_larm_ee_ori.tolist()) # i.e. previous goal
                        force_R_goal.append(g_rarm_ee_force.tolist()) # i.e. previous goal
                        force_L_goal.append(g_larm_ee_force.tolist()) # i.e. previous goal
#                        ddpos_box_goal.append(g_box_ddpos.tolist())
                pos_R_goal = optas.np.array(pos_R_goal).T
                ori_R_goal = optas.np.array(ori_R_goal).T
                pos_L_goal = optas.np.array(pos_L_goal).T
                ori_L_goal = optas.np.array(ori_L_goal).T
                force_R_goal = optas.np.array(force_R_goal).T
                force_L_goal = optas.np.array(force_L_goal).T
#                ddpos_box_goal = optas.np.array(ddpos_box_goal).T

#                print(ddpos_box_goal)

                # read current robot joint positions
                self.q_curr = np.concatenate((self.q_curr_base, self.q_curr_joint), axis=None)
                self.dq_curr = np.concatenate((self.dq_curr_base, self.dq_curr_joint), axis=None)

#                self.Derivation_RARM_pos_start[1] = np.asarray(self.wholebodyMPC.get_global_link_position(link=self._link_ee_right, q=self.q_curr)).T[0][1] - pos_R_goal[1, 0]
#                self.Derivation_LARM_pos_start[1] = np.asarray(self.wholebodyMPC.get_global_link_position(link=self._link_ee_left, q=self.q_curr)).T[0][1] - pos_L_goal[1, 0]

                self.G_Rotation_ee_right = self.rotation_fnc_Right(self.q_curr)
                self.G_Rotation_ee_left = self.rotation_fnc_Left(self.q_curr)

                self.G_X_ee_right[0:3, 0:3] = self.G_Rotation_ee_right; self.G_X_ee_right[3:6, 3:6] = self.G_Rotation_ee_right
                self.G_X_ee_left[0:3, 0:3] = self.G_Rotation_ee_left;   self.G_X_ee_left[3:6, 3:6] = self.G_Rotation_ee_left

                self.G_I_ee_r_conventional = self.G_X_ee_right @ self.I_ee_r_conventional @ self.G_X_ee_right.T;
                self.G_I_ee_l_conventional = self.G_X_ee_left @ self.I_ee_l_conventional @ self.G_X_ee_left.T;

#                print(self.G_I_ee_r_conventional[3:6, 3:6])
#                print(self.G_I_ee_l_conventional[3:6, 3:6])
                ### ---------------------------------------------------------
                ### optas. Solve the whole-body MPC
                # set initial seed
                if self.solution_MPC is None:
                    self.solver_wholebodyMPC.reset_initial_seed({f'Q': self.curr_MPC, f'P_Right': np.zeros((3, self.T_MPC)),
                                                                 f'P_Left': np.zeros((3, self.T_MPC)), f'acc_box_var': np.zeros((3, self.T_MPC))})
                # set initial seed
                if self.solution_MPC is not None:
                    self.solver_wholebodyMPC.reset_initial_seed({f'Q': self.solution_MPC[f'Q'], f'P_Right': self.solution_MPC[f'P_Right'],
                                                                 f'P_Left': self.solution_MPC[f'P_Left'], f'acc_box_var': self.solution_MPC[f'acc_box_var'] })

                self.solver_wholebodyMPC.reset_parameters({'pos_R': pos_R_goal, 'ori_R': ori_R_goal, 'pos_L': pos_L_goal, 'ori_L': ori_L_goal,
                                                           't': self.timebyT, 'init_position_MPC': self.q_curr, 'init_velocity_MPC': self.dq_curr,
                                                           'F_ext_Right_goal': force_R_goal, 'F_ext_Left_goal': force_L_goal,
                                                           'inertia_Right': self.G_I_ee_r_conventional[3:6, 3:6], 'inertia_Left': self.G_I_ee_l_conventional[3:6, 3:6],
                                                           'F_ext_Right_actual': self.F_ext_global_Right[3:6], 'F_ext_Left_actual': self.F_ext_global_Left[3:6],
                                                           'init_Delta_position_Right': self.Derivation_RARM_pos_start, 'init_Delta_position_Left': self.Derivation_LARM_pos_start,
                                                           'ddpos_box_goal': self.acc_box, 'm_box': self.m_box} )

                # solve problem
                self.solution_MPC = self.solver_wholebodyMPC.opt.decision_variables.vec2dict(self.solver_wholebodyMPC._solve())
                Q = np.asarray(self.solution_MPC[f'Q'])
                P_Right = np.asarray(self.solution_MPC[f'P_Right'])
                P_Left = np.asarray(self.solution_MPC[f'P_Left'])
                self.acc_box = np.asarray(self.solution_MPC[f'acc_box_var'])
                ### ---------------------------------------------------------
                # compute next configuration with lambda function
                t = (1./self._freq)/self.duration_MPC; n = self.T_MPC -1;
                q_next = np.zeros(self.ndof); p_right = np.zeros(3); dp_right = np.zeros(3); ddp_right = np.zeros(3);
                for j in range(self.T_MPC):
                    q_next += self.BC(n, j) * t**j * (1-t)**(n-j) * Q[:, j]
                    p_right += self.BC(n, j) * t**j * (1-t)**(n-j) * P_Right[:, j]
                for j in range(self.T_MPC-1):
                    dp_right += (1./self.duration_MPC) * self.BC(self.n-1, j) * t**j * (1-t)**(self.n-1-j) * self.n * (P_Right[:, j+1] -  P_Right[:, j])
                for j in range(self.T_MPC-2):
                    ddp_right += (1./self.duration_MPC)**2 * self.BC(self.n-2, j) * t**j * (1-t)**(self.n-2-j) * self.n * (self.n-1)* (P_Right[:, j+2] -  2*P_Right[:, j+1] + P_Right[:, j])

                dq_next = np.zeros(self.ndof)
                for j in range(self.T_MPC-1):
                    dq_next += (1./self.duration_MPC) * self.BC(n-1, j) * t**j * (1-t)**(n-1-j) * n * (Q[:, j+1] -  Q[:, j])

                ddq_next = np.zeros(self.ndof)
                for j in range(self.T_MPC-2):
                    ddq_next += (1./self.duration_MPC)**2 * self.BC(n-2, j) * t**j * (1-t)**(n-2-j) * n * (n-1)* (Q[:, j+2] -  2*Q[:, j+1] + Q[:, j])

#                self.acc_box = 0.5*(self.pos_Jac_fnc_Right(q_next) + self.pos_Jac_fnc_Left(q_next)) @ ddq_next
#                F_r = np.zeros(3)
#                F_l = np.zeros(3)
#                Delta_p_r = np.zeros(3)
#                dDelta_p_r = np.zeros(3)
#                ddDelta_p_r = np.zeros(3)
#                Delta_p_l = np.zeros(3)
#                dDelta_p_l = np.zeros(3)
#                ddDelta_p_l = np.zeros(3)
#                for j in range(self.T_MPC):
#                    Delta_p_r += self.BC(n, j) * t**j * (1-t)**(n-j) * P_Right[:, j]
#                    Delta_p_l += self.BC(n, j) * t**j * (1-t)**(n-j) * P_Left[:, j]
#                for j in range(self.T_MPC-1):
#                    dDelta_p_r += self.BC(n-1, j) * t**j * (1-t)**(n-1-j) * n * (P_Right[:, j+1] -  P_Right[:, j])
#                    dDelta_p_l += self.BC(n-1, j) * t**j * (1-t)**(n-1-j) * n * (P_Left[:, j+1] -  P_Left[:, j])
#                for j in range(self.T_MPC-2):
#                    ddDelta_p_r += self.BC(n-2, j) * t**j * (1-t)**(n-2-j) * n * (n-1)* (P_Right[:, j+2] -  2*P_Right[:, j+1] + P_Right[:, j])
#                    ddDelta_p_l += self.BC(n-2, j) * t**j * (1-t)**(n-2-j) * n * (n-1)* (P_Left[:, j+2] -  2*P_Left[:, j+1] + P_Left[:, j])
#                #####################################################################################
#                F_r = self.F_ext_global_Right[3:6] + self.K_Right @ Delta_p_r
#                F_l = self.F_ext_global_Left[3:6]  + self.K_Left @ Delta_p_l
#                F_r = self.F_ext_global_Right[3:6] + self.G_I_ee_r_conventional[3:6, 3:6] @ ddDelta_p_r + self.K_Right @ Delta_p_r + self.D_Right @ dDelta_p_r
#                F_l = self.F_ext_global_Left[3:6] + self.G_I_ee_l_conventional[3:6, 3:6] @ ddDelta_p_l + self.K_Left @ Delta_p_l + self.D_Left @ dDelta_p_l

#                print(F_r)
#                print(F_l)

                # read current robot joint positions
                self.q_curr = np.concatenate((self.q_curr_base, self.q_curr_joint), axis=None)
                self.Derivation_RARM_pos_start[1] = np.asarray(self.pos_fnc_Right(q_next)).T[0][1] - np.asarray(self.pos_fnc_Right(self.q_curr)).T[0][1]
                self.Derivation_LARM_pos_start[1] = np.asarray(self.pos_fnc_Left(q_next)).T[0][1] - np.asarray(self.pos_fnc_Left(self.q_curr)).T[0][1]

#                self.Derivation_RARM_pos_start[1] = pos_R_goal[1, 0] - np.asarray(self.wholebodyMPC.get_global_link_position(link=self._link_ee_right, q=self.q_curr)).T[0][1]
#                self.Derivation_LARM_pos_start[1] = pos_L_goal[1, 0] - np.asarray(self.wholebodyMPC.get_global_link_position(link=self._link_ee_left, q=self.q_curr)).T[0][1]

#                print(self.Derivation_RARM_pos_start)
#                print(self.Derivation_LARM_pos_start)

                # compute the donkey velocity in its local frame
                Global_w_b = np.asarray([0., 0., dq_next[2]])
                Global_v_b = np.asarray([dq_next[0], dq_next[1], 0.])
                Local_w_b = self.donkey_R.T @ Global_w_b
                Local_v_b = self.donkey_R.T @ Global_v_b

                self.eva_trajectory.header.stamp = rospy.Time.now()
                self.eva_trajectory.points[0].positions = q_next[-self.ndof_position_control:].tolist()
                # update message
                self._msg.data = q_next[-self.ndof_position_control:]
                self._msg_velocity.linear.x = Local_v_b[0]; self._msg_velocity.linear.y = Local_v_b[1]; self._msg_velocity.angular.z = Local_w_b[2];
                self._msg_acceleration.data = ddq_next[-self.ndof:]
                # publish message
                self._joint_pub.publish(self.eva_trajectory)
#                self._joint_pub.publish(self._msg)
                self._joint_pub_velocity.publish(self._msg_velocity)
                self._joint_acc_pub.publish(self._msg_acceleration)

                # compute progress
                self._feedback.progress = (self._idx*100)/self._steps
                # publish feedback
                self._action_server.publish_feedback(self._feedback)

                print(rospy.get_time()-time)

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
            self.start_RARM_torque = self._RARM_ee_torque_trajectory(self.duration)
            self.start_LARM_force = self._LARM_ee_force_trajectory(self.duration)
            self.start_LARM_torque = self._LARM_ee_torque_trajectory(self.duration)

            return

    def read_joint_states_cb(self, msg):
        self.q_curr_joint = np.asarray(list(msg.position)[:self.ndof_position_control])
        self.joint_names_position = msg.name[:self.ndof_position_control]
        self.dq_curr_joint = np.asarray(list(msg.velocity)[:self.ndof_position_control])

#    def read_ft_sensor_right_data_cb(self, msg):
#        self.ft_right = -np.asarray([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z, msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])

#    def read_ft_sensor_left_data_cb(self, msg):
#        self.ft_left = -np.asarray([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z, msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])

    def read_base_states_cb(self, msg):
        base_euler_angle = tf.transformations.euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        self.q_curr_base = [msg.pose.pose.position.x, msg.pose.pose.position.y, base_euler_angle[2]]
        self.donkey_R = optas.spatialmath.rotz(base_euler_angle[2])
        self.donkey_position = np.asarray([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
        self.donkey_velocity = np.asarray([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
        self.donkey_angular_velocity = np.asarray([msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z])
        self.dq_curr_base = [float(msg.twist.twist.linear.x), float(msg.twist.twist.linear.y), float(msg.twist.twist.angular.z)]

    def read_right_ee_grasp_ft_data_cb(self, msg):
#        self.F_ext_global_Right = np.asarray([ msg.data[0], msg.data[1], msg.data[2], msg.data[3], msg.data[4], msg.data[5] ])
        self.F_ext_global_Right = np.asarray([ msg.data[0], msg.data[1], msg.data[2], 0, msg.data[4], 0])

    def read_left_ee_grasp_ft_data_cb(self, msg):
#        self.F_ext_global_Left = np.asarray([ msg.data[0], msg.data[1], msg.data[2], msg.data[3], msg.data[4], msg.data[5] ])
        self.F_ext_global_Left = np.asarray([ msg.data[0], msg.data[1], msg.data[2], 0, msg.data[4], 0 ])

#    def read_box_acc_data_cb(self, msg):
#        self.acc_box = np.asarray([ msg.data[0], msg.data[1], msg.data[2], 0, msg.data[4], 0])


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

if __name__=="__main__":
    # Initialize node
    rospy.init_node("cmd_pose_server_MPC_BC_operational_AD", anonymous=True)
    # Initialize node class
    cmd_pose_server = CmdPoseActionServer(rospy.get_name())
    # executing node
    rospy.spin()
