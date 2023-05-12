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
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension
from pushing_msgs.msg import CmdChonkPoseAction, CmdChonkPoseFeedback, CmdChonkPoseResult
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from urdf_parser_py.urdf import URDF
import tf
from X_fromRandP import X_fromRandP, X_fromRandP_different
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

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
#        self._link_sensor_right = rospy.get_param('~link_sensor_right', 'link_sensor_right')
#        self._link_sensor_left = rospy.get_param('~link_sensor_left', 'link_sensor_left')
        self._link_head = rospy.get_param('~link_head', 'link_head')
        self._link_gaze = rospy.get_param('~link_gaze', 'link_gaze')
        # control frequency
        self._freq = rospy.get_param('~freq', 20)
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
        self.tf_listener = tf.TransformListener()
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
        self.dt_MPC = 0.1 # time step
        self.T_MPC = 6 # T is number of time steps
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
        P = builder_wholebodyMPC.add_decision_variables('P', self.ndof, self.T_MPC)
#        p_epsilon = builder_wholebodyMPC.add_decision_variables('p_epsilon', self.ndof)


        t = builder_wholebodyMPC.add_parameter('t', self.T_MPC)  # time
        self.n = self.T_MPC -1 # N in Bezier curve
        # Add parameters
        init_position_MPC = builder_wholebodyMPC.add_parameter('init_position_MPC', self.ndof)  # initial robot position
        init_velocity_MPC = builder_wholebodyMPC.add_parameter('init_velocity_MPC', self.ndof)  # initial robot velocity
        # get end-effector pose as parameters
        pos_R = builder_wholebodyMPC.add_parameter('pos_R', 3, self.T_MPC)
        ori_R = builder_wholebodyMPC.add_parameter('ori_R', 4, self.T_MPC)
        pos_L = builder_wholebodyMPC.add_parameter('pos_L', 3, self.T_MPC)
        ori_L = builder_wholebodyMPC.add_parameter('ori_L', 4, self.T_MPC)
        # functions of right and left arm positions
        pos_fnc_Right = self.wholebodyMPC.get_global_link_position_function(link=self._link_ee_right)
        pos_fnc_Left = self.wholebodyMPC.get_global_link_position_function(link=self._link_ee_left)
        # quaternion functions of two arm end effectors
        ori_fnc_Right = self.wholebodyMPC.get_global_link_quaternion_function(link=self._link_ee_right)
        ori_fnc_Left = self.wholebodyMPC.get_global_link_quaternion_function(link=self._link_ee_left)
        # define q function depending on P
        q_var_MPC = optas.casadi.SX(np.zeros((self.ndof, self.T_MPC)))

        pos_ee_INsensorFrame_Right = np.asarray([-0.08, 0, 0.039+0.04])
        pos_ee_INsensorFrame_Left = np.asarray([-0.08, 0, 0.039+0.04])
        rot_ee_INsensorFrame_Right = optas.spatialmath.roty(-np.pi/2)
        rot_ee_INsensorFrame_Left = optas.spatialmath.roty(-np.pi/2)

        self.X_ee_INsensorFrame_Right = X_fromRandP(rot_ee_INsensorFrame_Right, pos_ee_INsensorFrame_Right)
        self.X_ee_INsensorFrame_Left = X_fromRandP(rot_ee_INsensorFrame_Left, pos_ee_INsensorFrame_Left)

#        self.rot_ee_right_fnc_global = self.wholebodyMPC.get_global_link_rotation_function(link=self._link_ee_right)
#        self.rot_ee_left_fnc_global = self.wholebodyMPC.get_global_link_rotation_function(link=self._link_ee_left)
#        self.pos_mass_INee_Right = np.asarray([-0.039-0.04+0.047729, 0, -0.08+0.017732])
#        self.pos_mass_INee_Left = np.asarray([-0.039-0.04+0.047729, 0, -0.08+0.017732])

#        self.mass_ee_Force = np.asarray([0,0,0,0,0,-0.3113*9.8])





        # quaternion functions by given q_var_MPC
#        ori_function_R = optas.casadi.SX(np.zeros((4, self.T_MPC)))
#        ori_function_L = optas.casadi.SX(np.zeros((4, self.T_MPC)))
#        # coefficient matrix between quaternion rate and angular velocity in the global frame
#        RARM_Matrix_angular2quat = optas.casadi.SX(np.zeros((4, self.T_MPC * 3)))
#        LARM_Matrix_angular2quat = optas.casadi.SX(np.zeros((4, self.T_MPC * 3)))

        for i in range(self.T_MPC):
            for j in range(self.T_MPC):
                q_var_MPC[:, i] += self.BC(self.n, j) * t[i]**j * (1-t[i])**(self.n-j) * P[:, j]
            name = 'control_point_' + str(i) + '_bound' # add position constraint for each P[:, i]
            builder_wholebodyMPC.add_bound_inequality_constraint(name, lhs=lower, mid=P[:, i], rhs=upper)
#            builder_wholebodyMPC.add_equality_constraint('twoarm_miniscope' + str(i), q_var_MPC[6, i], rhs=-q_var_MPC[12, i])

#            ori_function_R[:, i] = ori_fnc_Right(q_var_MPC[:, i])
#            ori_function_L[:, i] = ori_fnc_Left(q_var_MPC[:, i])
#            RARM_Matrix_angular2quat[:, i * 3 : (i+1)*3][0,0] =  ori_function_R[:, i][3]
#            RARM_Matrix_angular2quat[:, i * 3 : (i+1)*3][0,1] =  ori_function_R[:, i][2]
#            RARM_Matrix_angular2quat[:, i * 3 : (i+1)*3][0,2] =  ori_function_R[:, i][1]
#            RARM_Matrix_angular2quat[:, i * 3 : (i+1)*3][1,0] = -ori_function_R[:, i][2]
#            RARM_Matrix_angular2quat[:, i * 3 : (i+1)*3][1,1] =  ori_function_R[:, i][3]
#            RARM_Matrix_angular2quat[:, i * 3 : (i+1)*3][1,2] =  ori_function_R[:, i][0]
#            RARM_Matrix_angular2quat[:, i * 3 : (i+1)*3][2,0] =  ori_function_R[:, i][1]
#            RARM_Matrix_angular2quat[:, i * 3 : (i+1)*3][2,1] = -ori_function_R[:, i][0]
#            RARM_Matrix_angular2quat[:, i * 3 : (i+1)*3][2,2] =  ori_function_R[:, i][3]
#            RARM_Matrix_angular2quat[:, i * 3 : (i+1)*3][3,0] = -ori_function_R[:, i][0]
#            RARM_Matrix_angular2quat[:, i * 3 : (i+1)*3][3,1] = -ori_function_R[:, i][1]
#            RARM_Matrix_angular2quat[:, i * 3 : (i+1)*3][3,2] = -ori_function_R[:, i][2]

#            LARM_Matrix_angular2quat[:, i * 3 : (i+1)*3][0,0] =  ori_function_L[:, i][3]
#            LARM_Matrix_angular2quat[:, i * 3 : (i+1)*3][0,1] =  ori_function_L[:, i][2]
#            LARM_Matrix_angular2quat[:, i * 3 : (i+1)*3][0,2] =  ori_function_L[:, i][1]
#            LARM_Matrix_angular2quat[:, i * 3 : (i+1)*3][1,0] = -ori_function_L[:, i][2]
#            LARM_Matrix_angular2quat[:, i * 3 : (i+1)*3][1,1] =  ori_function_L[:, i][3]
#            LARM_Matrix_angular2quat[:, i * 3 : (i+1)*3][1,2] =  ori_function_L[:, i][0]
#            LARM_Matrix_angular2quat[:, i * 3 : (i+1)*3][2,0] =  ori_function_L[:, i][1]
#            LARM_Matrix_angular2quat[:, i * 3 : (i+1)*3][2,1] = -ori_function_L[:, i][0]
#            LARM_Matrix_angular2quat[:, i * 3 : (i+1)*3][2,2] =  ori_function_L[:, i][3]
#            LARM_Matrix_angular2quat[:, i * 3 : (i+1)*3][3,0] = -ori_function_L[:, i][0]
#            LARM_Matrix_angular2quat[:, i * 3 : (i+1)*3][3,1] = -ori_function_L[:, i][1]
#            LARM_Matrix_angular2quat[:, i * 3 : (i+1)*3][3,2] = -ori_function_L[:, i][2]

            # optimization cost: close to target
            builder_wholebodyMPC.add_cost_term('Right_arm position' + str(i), optas.sumsqr(pos_fnc_Right(q_var_MPC[:, i])-pos_R[:, i]))
            builder_wholebodyMPC.add_cost_term('Left_arm position' + str(i), optas.sumsqr(pos_fnc_Left(q_var_MPC[:, i])-pos_L[:, i]))
            builder_wholebodyMPC.add_cost_term('Right_arm orientation' + str(i), optas.sumsqr(ori_fnc_Right(q_var_MPC[:, i])-ori_R[:, i]))
            builder_wholebodyMPC.add_cost_term('Left_arm orientation' + str(i), optas.sumsqr(ori_fnc_Left(q_var_MPC[:, i])-ori_L[:, i]))
            builder_wholebodyMPC.add_cost_term('twoarm_miniscope' + str(i), 0.1 * optas.sumsqr(q_var_MPC[6, i]+q_var_MPC[12, i]))
            builder_wholebodyMPC.add_cost_term('chest_miniscope' + str(i), optas.sumsqr(q_var_MPC[3, i]))
            builder_wholebodyMPC.add_cost_term('arm_joint_miniscope' + str(i), 0.001 * optas.sumsqr(q_var_MPC[6:self.ndof, i]))
            if(i<(self.T_MPC -1)):
                builder_wholebodyMPC.add_cost_term('distance' + str(i), 0.001 * optas.sumsqr(P[:, i+1] - P[:, i]))
#            if(i<(self.T_MPC -1)):
#                builder_wholebodyMPC.add_cost_term('minimize_velocity2' + str(i), 0.0005 * optas.sumsqr(q_var_MPC[6:self.ndof, i+1] - q_var_MPC[6:self.ndof, i]))

#            obstacle_pos = np.asarray([[3.8], [0]])
#            obstacle_radius = 0.75
#            offset = np.asarray([[0.5], [0]])
#            builder_wholebodyMPC.add_geq_inequality_constraint('donkey_obstacle' + str(i), lhs=optas.norm_2(P[0:2, i] + offset - obstacle_pos), rhs=obstacle_radius**2)

#            if(i<(self.T_MPC -1)):
#                builder_wholebodyMPC.add_cost_term('Length' + str(i), optas.sumsqr(P[:, i+1]-P[:, i]))
        # add position constraint at the beginning state
        builder_wholebodyMPC.add_equality_constraint('init_position', P[0:4, 0], rhs=init_position_MPC[0:4])
        builder_wholebodyMPC.add_equality_constraint('init_position2', P[6:self.ndof, 0], rhs=init_position_MPC[6:self.ndof])

        builder_wholebodyMPC.add_equality_constraint('head_miniscope', P[4, :],  rhs=np.full((1, self.T_MPC), 0.0 ) )
        builder_wholebodyMPC.add_equality_constraint('head_miniscope1', P[5, :], rhs=np.full((1, self.T_MPC), -0.25) )


#        builder_wholebodyMPC.add_bound_inequality_constraint('init_p_equal', lhs=init_position_MPC[3:self.ndof] - np.full((15, 1), 0.005), mid=P[3:self.ndof, 0] + p_epsilon[3:self.ndof], rhs=init_position_MPC[3:self.ndof] + np.full((15, 1), 0.005))
#        builder_wholebodyMPC.add_bound_inequality_constraint('init_p_equal2', lhs=init_position_MPC[0:3] - np.full((3, 1), 0.001), mid=P[0:3, 0] + p_epsilon[0:3], rhs=init_position_MPC[0:3] + np.full((3, 1), 0.001))
#        builder_wholebodyMPC.add_cost_term('minimize_p_epsilon', optas.sumsqr(p_epsilon))

#        builder_wholebodyMPC.add_inequality_constraint('init_velocity', lhs=self.n * (P[0:3, 1] - P[0:3, 0]), rhs=init_velocity_MPC[0:3])


#        # get end-effector pose as parameters
#        dpos_R = builder_wholebodyMPC.add_parameter('dpos_R', 3, self.T_MPC)
#        dori_R = builder_wholebodyMPC.add_parameter('dori_R', 4, self.T_MPC)
#        dpos_L = builder_wholebodyMPC.add_parameter('dpos_L', 3, self.T_MPC)
#        dori_L = builder_wholebodyMPC.add_parameter('dori_L', 4, self.T_MPC)
#        # functions of right and left arm positions
#        dpos_fnc_Right = self.wholebodyMPC.get_global_link_linear_jacobian_function(link=self._link_ee_right)
#        dpos_fnc_Left = self.wholebodyMPC.get_global_link_linear_jacobian_function(link=self._link_ee_left)
#        # quaternion functions of two arm end effectors
#        dori_fnc_Right = self.wholebodyMPC.get_global_link_angular_geometric_jacobian_function(link=self._link_ee_right)
#        dori_fnc_Left = self.wholebodyMPC.get_global_link_angular_geometric_jacobian_function(link=self._link_ee_left)
        # define dq function depending on P

        dq_var_MPC = optas.casadi.SX(np.zeros((self.ndof, self.T_MPC)))
        w_dq = 0.0001/float(self.T_MPC)
        for i in range(self.T_MPC):
            for j in range(self.T_MPC-1):
                dq_var_MPC[:, i] += self.BC(self.n-1, j) * t[i]**j * (1-t[i])**(self.n-1-j) * self.n * (P[:, j+1] -  P[:, j])
            if(i<(self.T_MPC -1)):
                name = 'control_point_deriv_' + str(i) + '_bound'  # add velocity constraint for each P[:, i]
                builder_wholebodyMPC.add_bound_inequality_constraint(name, lhs=dlower, mid=self.n * (P[:, i+1] -  P[:, i]), rhs=dupper)
            builder_wholebodyMPC.add_cost_term('minimize_velocity' + str(i), w_dq * optas.sumsqr(dq_var_MPC[:, i]))
#            builder_wholebodyMPC.add_cost_term('Right_arm linear velocity' + str(i), optas.sumsqr(dpos_fnc_Right(q_var_MPC[:, i]) @ dq_var_MPC[:, i]-dpos_R[:, i]))
#            builder_wholebodyMPC.add_cost_term('Left_arm linear velocity' + str(i), optas.sumsqr(dpos_fnc_Left(q_var_MPC[:, i]) @ dq_var_MPC[:, i]-dpos_L[:, i]))
        ddq_var_MPC = optas.casadi.SX(np.zeros((self.ndof, self.T_MPC)))
        w_ddq = 0.0005/float(self.T_MPC)
        for i in range(self.T_MPC):
            for j in range(self.T_MPC-2):
                ddq_var_MPC[:, i] += self.BC(self.n-2, j) * t[i]**j * (1-t[i])**(self.n-2-j) * self.n * (self.n-1)* (P[:, j+2] -  2*P[:, j+1] + P[:, j])
            builder_wholebodyMPC.add_cost_term('minimize_acceleration' + str(i), w_ddq * optas.sumsqr(ddq_var_MPC[:, i]))
#            # optimization cost: close to target
#            builder_wholebodyMPC.add_cost_term('Right_arm linear velocity' + str(i), optas.sumsqr(dpos_fnc_Right(q_var_MPC[:, i]) @ dq_var_MPC[:, i]-dpos_R[:, i]))
#            builder_wholebodyMPC.add_cost_term('Left_arm linear velocity' + str(i), optas.sumsqr(dpos_fnc_Left(q_var_MPC[:, i]) @ dq_var_MPC[:, i]-dpos_L[:, i]))
#            builder_wholebodyMPC.add_cost_term('Right_arm angular velocity' + str(i), optas.sumsqr(0.5 * RARM_Matrix_angular2quat[:, i * 3 : (i+1)*3] @ dori_fnc_Right(q_var_MPC[:, i]) @ dq_var_MPC[:, i]-dori_R[:, i]))
#            builder_wholebodyMPC.add_cost_term('Left_arm angular velocity' + str(i), optas.sumsqr(0.5 * LARM_Matrix_angular2quat[:, i * 3 : (i+1)*3] @ dori_fnc_Left(q_var_MPC[:, i]) @ dq_var_MPC[:, i] -dori_L[:, i]))


        # setup solver
#        self.solver_wholebodyMPC = optas.CasADiSolver(optimization=builder_wholebodyMPC.build()).setup('ipopt', solver_options={'ipopt.print_level': 0, 'print_time': 0})
        self.solver_wholebodyMPC = optas.CasADiSolver(optimization=builder_wholebodyMPC.build()).setup('knitro')
#        self.solver_wholebodyMPC = optas.CasADiSolver(optimization=builder_wholebodyMPC.build()).setup('knitro', solver_options={'knitro.OutLev': 10} )
#        self.solver_wholebodyMPC = optas.CasADiSolver(optimization=builder_wholebodyMPC.build()).setup('knitro', solver_options={'knitro.OutLev': 0, 'print_time': 0} )
        self.solver_wholebodyMPC = optas.CasADiSolver(optimization=builder_wholebodyMPC.build()).setup('knitro', solver_options={
                                                                                                       'knitro.OutLev': 0, 'print_time': 0,
                                                                                                       'knitro.FeasTol': 1e-6, 'knitro.OptTol': 1e-6, 'knitro.ftol':1e-6,
                                                                                                       'knitro.algorithm':1,
                                                                                                       'knitro.linsolver':2,
#                                                                                                       'knitro.maxtime_real': 1.8e-2,
                                                                                                       'knitro.bar_initpt':3, 'knitro.bar_murule':4, 'knitro.bar_penaltycons': 1,
                                                                                                       'knitro.bar_penaltyrule':2, 'knitro.bar_switchrule':2, 'knitro.linesearch': 1} )
        self.ti_MPC = 0 # time index of the MPC
        self.solution_MPC = None
        self.time_linspace = np.linspace(0., self.duration_MPC, self.T_MPC)
        self.timebyT = np.asarray(self.time_linspace)/self.duration_MPC

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
#        self._joint_sub_base = rospy.Subscriber(
#            "/chonk/base_pose_ground_truth",
#            Odometry,
#            self.read_base_states_cb
#        )
        # declare joint publisher
#        self._joint_pub = rospy.Publisher(
#            self._pub_cmd_topic_name,
#            Float64MultiArray,
#            queue_size=10
#        )
#        self._joint_pub = rospy.Publisher(
#            "/chonk/streaming_controller/command",
#            Float64MultiArray,
#            queue_size=10
#        )
        self._joint_pub = rospy.Publisher(
            "/chonk/trajectory_controller/command",
            JointTrajectory,
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
        # declare mux service
        self._srv_mux_sel = rospy.ServiceProxy(rospy.get_namespace() + '/mux_joint_position/select', MuxSelect)
        # declare subscriber for selected controller
        self._sub_selected_controller = rospy.Subscriber(
            "/mux_selected",
            String,
            self.read_mux_selection
        )
        # initialize action messages
        self._feedback = CmdChonkPoseFeedback()
        self._result = CmdChonkPoseResult()
        # declare action server
        self._action_server = actionlib.SimpleActionServer(
            'cmd_pose',
            CmdChonkPoseAction,
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
        pos_Right = np.asarray([
                acceped_goal.poseR.position.x,
                acceped_goal.poseR.position.y,
                acceped_goal.poseR.position.z
        ])
        pos_Left = np.asarray([
                acceped_goal.poseL.position.x,
                acceped_goal.poseL.position.y,
                acceped_goal.poseL.position.z
        ])
        ori_Right = np.asarray([
                acceped_goal.poseR.orientation.x,
                acceped_goal.poseR.orientation.y,
                acceped_goal.poseR.orientation.z,
                acceped_goal.poseR.orientation.w
        ])
        ori_Left = np.asarray([
                acceped_goal.poseL.orientation.x,
                acceped_goal.poseL.orientation.y,
                acceped_goal.poseL.orientation.z,
                acceped_goal.poseL.orientation.w
        ])
        # check boundaries of the position
        if (pos_Right > self._pos_max).any() or (pos_Right < self._pos_min).any():
            rospy.logwarn("%s: Request aborted. Goal position (%.2f, %.2f, %.2f) is outside of the workspace boundaries. Check parameters for this node." % (self._name, pos_Right[0], pos_Right[1], pos_Right[2]))
            self._result.reached_goal = False
            self._action_server.set_aborted(self._result)
            return
        if (pos_Left > self._pos_max).any() or (pos_Left < self._pos_min).any():
            rospy.logwarn("%s: Lequest aborted. Goal position (%.2f, %.2f, %.2f) is outside of the workspace boundaries. Check parameters for this node." % (self._name, pos_Left[0], pos_Left[1], pos_Left[2]))
            self._result.reached_goal = False
            self._action_server.set_aborted(self._result)
            return
        # print goal request
        rospy.loginfo("%s: Request to send right arm to position (%.2f, %.2f, %.2f) with orientation (%.2f, %.2f, %.2f, %.2f), and left arm to position (%.2f, %.2f, %.2f) with orientation (%.2f, %.2f, %.2f, %.2f) in %.1f seconds." % (
                self._name,
                pos_Right[0], pos_Right[1], pos_Right[2],
                ori_Right[0], ori_Right[1], ori_Right[2], ori_Right[3],
                pos_Left[0], pos_Left[1], pos_Left[2],
                ori_Left[0], ori_Left[1], ori_Left[2], ori_Left[3],
                acceped_goal.duration
            )
        )
        # read current robot joint positions
        try:
            (trans,rot) = self.tf_listener.lookupTransform('/vicon/world', 'vicon/chonk/CHONK',  rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print("error: cannot find vicon data!!!!")
        self.base_euler_angle = tf.transformations.euler_from_quaternion([rot[0], rot[1], rot[2], rot[3]])
        self.q_curr_base = [trans[0], trans[1], self.base_euler_angle[2]]

        self.q_curr = np.concatenate((self.q_curr_base, self.q_curr_joint), axis=None)
        self.dq_curr = np.concatenate((self.dq_curr_base, self.dq_curr_joint), axis=None)
        qT = np.zeros(self.ndof)
        self.joint_names = self.joint_names_base + self.joint_names_position
        ### optas
        ### ---------------------------------------------------------
        # get two-arm end effector trajectory in the operational space
        q0 = self.q_curr.T
        self.duration = acceped_goal.duration
        self._steps = int(self.duration * self._freq)
        self._idx = 0
        # current right and left arm end effector position and quaternion
        start_RARM_quat = np.asarray(self.wholebodyMPC.get_global_link_quaternion(link=self._link_ee_right, q=q0)).T[0]
        start_RARM_pos = np.asarray(self.wholebodyMPC.get_global_link_position(link=self._link_ee_right, q=q0)).T[0]
        start_LARM_quat = np.asarray(self.wholebodyMPC.get_global_link_quaternion(link=self._link_ee_left, q=q0)).T[0]
        start_LARM_pos = np.asarray(self.wholebodyMPC.get_global_link_position(link=self._link_ee_left, q=q0)).T[0]
        # derivation of right and left arm end effector position and quaternion compared with the beginning ee position and quaternion
        Derivation_RARM_Pos = pos_Right - start_RARM_pos
        Derivation_RARM_Quat = ori_Right - start_RARM_quat
        Derivation_LARM_Pos = pos_Left - start_LARM_pos
        Derivation_LARM_Quat = ori_Left - start_LARM_quat
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

        self.eva_trajectory = JointTrajectory()
        self.eva_trajectory.header.frame_id = ''
        self.eva_trajectory.joint_names = ['CHEST_JOINT0', 'HEAD_JOINT0', 'HEAD_JOINT1',
                                  'LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2', 'LARM_JOINT3', 'LARM_JOINT4', 'LARM_JOINT5',
                                  'RARM_JOINT0', 'RARM_JOINT1', 'RARM_JOINT2', 'RARM_JOINT3', 'RARM_JOINT4', 'RARM_JOINT5']
        self.eva_point = JointTrajectoryPoint()
        self.eva_point.time_from_start = rospy.Duration(0.1)
        self.eva_trajectory.points.append(self.eva_point)


        # create timer
        dur = rospy.Duration(1.0/self._freq)
        self._timer = rospy.Timer(dur, self.timer_cb)

        self.curr_MPC = np.zeros((self.ndof, self.T_MPC))
        for i in range(self.T_MPC):
            self.curr_MPC[:,i] = self.q_curr

    def timer_cb(self, event):
        """ Publish the robot configuration """
        try:
            (trans,rot) = self.tf_listener.lookupTransform('/vicon/world', 'vicon/chonk/CHONK', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print("error: cannot find vicon data!!!!")
        self.base_euler_angle = tf.transformations.euler_from_quaternion([rot[0], rot[1], rot[2], rot[3]])
        self.q_curr_base = [trans[0], trans[1], self.base_euler_angle[2]]
        self.donkey_R = optas.spatialmath.rotz(self.base_euler_angle[2])

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
                self._idx += 1

                pos_R_goal = []
                ori_R_goal = []
                pos_L_goal = []
                ori_L_goal = []
                dpos_R_goal = []
                dori_R_goal = []
                dpos_L_goal = []
                dori_L_goal = []

                for i in range(self.T_MPC):
#                    self.ti_MPC = self._t[self._idx-1]  + self.dt_MPC*i
                    if(self.ti_MPC <= self.duration):
                        self.ti_MPC = self._t[self._idx-1]  + self.dt_MPC*i
                    if(self.ti_MPC > self.duration):
                        self.ti_MPC = self.duration
                    try:
                        g_rarm_ee_pos = self._RARM_ee_Pos_trajectory(self.ti_MPC).flatten()
                        pos_R_goal.append(g_rarm_ee_pos.tolist())
                        g_rarm_ee_ori = self._RARM_ee_Quat_trajectory(self.ti_MPC).flatten()
#                        g_rarm_ee_ori[0] = np.sqrt(1-g_rarm_ee_ori[1]**2-g_rarm_ee_ori[2]**2-g_rarm_ee_ori[3]**2)
#                        print(g_rarm_ee_ori[0]**2+g_rarm_ee_ori[1]**2+g_rarm_ee_ori[2]**2 +g_rarm_ee_ori[3]**2)
                        ori_R_goal.append(g_rarm_ee_ori.tolist())
                        g_larm_ee_pos = self._LARM_ee_Pos_trajectory(self.ti_MPC).flatten()
                        pos_L_goal.append(g_larm_ee_pos.tolist())
                        g_larm_ee_ori = self._LARM_ee_Quat_trajectory(self.ti_MPC).flatten()
#                        g_larm_ee_ori[0] = np.sqrt(1-g_larm_ee_ori[1]**2-g_larm_ee_ori[2]**2-g_larm_ee_ori[3]**2)
                        ori_L_goal.append(g_larm_ee_ori.tolist())

#                        dg_rarm_ee_pos = self._DRARM_ee_Pos_trajectory(self.ti_MPC).flatten()
#                        dpos_R_goal.append(dg_rarm_ee_pos.tolist())
#                        dg_rarm_ee_ori = self._DRARM_ee_Quat_trajectory(self.ti_MPC).flatten()
#                        dori_R_goal.append(dg_rarm_ee_ori.tolist())
#                        dg_larm_ee_pos = self._DLARM_ee_Pos_trajectory(self.ti_MPC).flatten()
#                        dpos_L_goal.append(dg_larm_ee_pos.tolist())
#                        dg_larm_ee_ori = self._DLARM_ee_Quat_trajectory(self.ti_MPC).flatten()
#                        dori_L_goal.append(dg_larm_ee_ori.tolist())
                    except ValueError:
                        pos_R_goal.append(g_rarm_ee_pos.tolist()) # i.e. previous goal
                        ori_R_goal.append(g_rarm_ee_ori.tolist()) # i.e. previous goal
                        pos_L_goal.append(g_larm_ee_pos.tolist()) # i.e. previous goal
                        ori_L_goal.append(g_larm_ee_ori.tolist()) # i.e. previous goal
#                        dpos_R_goal.append(dg_rarm_ee_pos.tolist()) # i.e. previous goal
#                        dori_R_goal.append(dg_rarm_ee_ori.tolist()) # i.e. previous goal
#                        dpos_L_goal.append(dg_rarm_ee_pos.tolist()) # i.e. previous goal
#                        dori_L_goal.append(dg_rarm_ee_ori.tolist()) # i.e. previous goal

                pos_R_goal = optas.np.array(pos_R_goal).T
                ori_R_goal = optas.np.array(ori_R_goal).T
                pos_L_goal = optas.np.array(pos_L_goal).T
                ori_L_goal = optas.np.array(ori_L_goal).T
#                dpos_R_goal = optas.np.array(dpos_R_goal).T
#                dori_R_goal = optas.np.array(dori_R_goal).T
#                dpos_L_goal = optas.np.array(dpos_L_goal).T
#                dori_L_goal = optas.np.array(dori_L_goal).T



                ### optas
                ### ---------------------------------------------------------
                ### solve the whole-body MPC
                # set initial seed
                if self.solution_MPC is None:
                    self.solver_wholebodyMPC.reset_initial_seed({f'P': self.curr_MPC})
                # set initial seed
                if self.solution_MPC is not None:
                    self.solver_wholebodyMPC.reset_initial_seed({f'P': self.solution_MPC[f'P']})

                self.solver_wholebodyMPC.reset_parameters({'pos_R': pos_R_goal, 'ori_R': ori_R_goal, 'pos_L': pos_L_goal, 'ori_L': ori_L_goal, 't': self.timebyT, 'init_position_MPC': self.q_curr, 'init_velocity_MPC': self.dq_curr } )
#                self.solver_wholebodyMPC.reset_parameters({'pos_R': pos_R_goal, 'ori_R': ori_R_goal, 'pos_L': pos_L_goal, 'ori_L': ori_L_goal, 't': self.timebyT} )

                # solve problem
                self.solution_MPC = self.solver_wholebodyMPC.opt.decision_variables.vec2dict(self.solver_wholebodyMPC._solve())
                P = np.asarray(self.solution_MPC[f'P'])

                ### ---------------------------------------------------------
                # compute next configuration with lambda function
                t = (1./self._freq)/self.duration_MPC
                n = self.T_MPC -1
                q_next = np.zeros(self.ndof)
                for j in range(self.T_MPC):
                    q_next += self.BC(n, j) * t**j * (1-t)**(n-j) * P[:, j]
                dq_next = np.zeros(self.ndof)
                for j in range(self.T_MPC-1):
                    dq_next += self.BC(n-1, j) * t**j * (1-t)**(n-1-j) * n * (P[:, j+1] -  P[:, j])
#                q_next = P[:, 0]
#                dq_next = self.n * (P[:, 1] -  P[:, 0])
                # compute the donkey velocity in its local frame
                Global_w_b = np.asarray([0., 0., dq_next[2]])
                Global_v_b = np.asarray([dq_next[0], dq_next[1], 0.])
                Local_w_b = self.donkey_R.T @ Global_w_b
                Local_v_b = self.donkey_R.T @ Global_v_b
                # update message
#                self._msg.data[0:12] = q_next[-12:]
#                self._msg.data[12:15] = q_next[3:6]
#                self.eva_point.positions = q_next[-self.ndof_position_control:].tolist()
                self.eva_trajectory.header.stamp = rospy.Time.now()
                self.eva_trajectory.points[0].positions = q_next[-self.ndof_position_control:].tolist()
#                print(q_next[-self.ndof_position_control:])

#                self._msg.data = [float('%.3f' % x) for x in self._msg.data]
#                self._msg.data = q_next[-self.ndof_position_control:]
                self._msg_velocity.linear.x = Local_v_b[0]
                self._msg_velocity.linear.y = Local_v_b[1]
                self._msg_velocity.angular.z = Local_w_b[2] * 6.05
                # publish message
                self._joint_pub.publish(self.eva_trajectory)
#                self._joint_pub.publish(self._msg)
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

#    def read_base_states_cb(self, msg):
#        base_euler_angle = tf.transformations.euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
#        self.q_curr_base = [msg.pose.pose.position.x, msg.pose.pose.position.y, base_euler_angle[2]]
#        self.donkey_R = optas.spatialmath.rotz(base_euler_angle[2])
#        self.donkey_position = np.asarray([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
#        self.donkey_velocity = np.asarray([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
#        self.donkey_angular_velocity = np.asarray([msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z])
#        self.dq_curr_base = [float(msg.twist.twist.linear.x), float(msg.twist.twist.linear.y), float(msg.twist.twist.angular.z)]


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
    rospy.init_node("cmd_pose_server_MPC_BC_operational", anonymous=True)
    # Initialize node class
    cmd_pose_server = CmdPoseActionServer(rospy.get_name())
    # executing node
    rospy.spin()
