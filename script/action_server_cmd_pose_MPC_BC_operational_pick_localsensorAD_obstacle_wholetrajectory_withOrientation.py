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
        # set up whole-body MPC planner in real time
        self.wholebodyMPC_planner= optas.RobotModel(
            urdf_string=self._robot_description,
            time_derivs=[0],
            param_joints=['base_joint_1', 'base_joint_2', 'base_joint_3', 'CHEST_JOINT0', 'HEAD_JOINT0', 'HEAD_JOINT1', 'LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2', 'LARM_JOINT3', 'LARM_JOINT4', 'LARM_JOINT5', 'RARM_JOINT0', 'RARM_JOINT1', 'RARM_JOINT2', 'RARM_JOINT3', 'RARM_JOINT4', 'RARM_JOINT5'],
            name='chonk_wholebodyMPC'
        )
        self.wholebodyMPC_planner_name = self.wholebodyMPC_planner.get_name()
#        self.dt_MPC_planner = 0.1 # time step
        self.T_MPC_planner = 12 # T is number of time steps
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
#            builder_wholebodyMPC_planner.add_cost_term('Two_arm orientation parallel' + str(i), optas.sumsqr(R_ori_Right[:, i].T @ R_ori_Left[:, i]))
            builder_wholebodyMPC_planner.add_cost_term('Two_arm end height same' + str(i), optas.sumsqr(r_pos_RARM_var_MPC[2, i] - r_pos_LARM_var_MPC[2, i]))
#            builder_wholebodyMPC_planner.add_cost_term('Right_arm_align' + str(i), optas.sumsqr( self.skew_optas(self.quatToRotationZ(R_ori_Right[:, i])) @ (R_pos_Right[:, i] - R_pos_Left[:, i])   ))
#            builder_wholebodyMPC_planner.add_cost_term('Left_arm_align' + str(i), optas.sumsqr( self.skew_optas(self.quatToRotationZ(R_ori_Left[:, i])) @ (R_pos_Right[:, i] - R_pos_Left[:, i])   ))
            builder_wholebodyMPC_planner.add_cost_term('Right_arm_align_x' + str(i), 20*optas.sumsqr( self.quatToRotationX(r_ori_RARM_var_MPC[:, i]).T @ (r_pos_RARM_var_MPC[:, i] - r_pos_LARM_var_MPC[:, i])   ))
            builder_wholebodyMPC_planner.add_cost_term('Right_arm_align_y' + str(i), 5*optas.sumsqr( self.quatToRotationY(r_ori_RARM_var_MPC[:, i]).T @ (r_pos_RARM_var_MPC[:, i] - r_pos_LARM_var_MPC[:, i])   ))

            builder_wholebodyMPC_planner.add_cost_term('Left_arm_align_x' + str(i), 20*optas.sumsqr( self.quatToRotationX(r_ori_LARM_var_MPC[:, i]).T @ (r_pos_RARM_var_MPC[:, i] - r_pos_LARM_var_MPC[:, i])   ))
            builder_wholebodyMPC_planner.add_cost_term('Left_arm_align_y' + str(i), 5*optas.sumsqr( self.quatToRotationY(r_ori_LARM_var_MPC[:, i]).T @ (r_pos_RARM_var_MPC[:, i] - r_pos_LARM_var_MPC[:, i])   ))


            builder_wholebodyMPC_planner.add_cost_term('Two_arm orientation parallel' + str(i), 20*optas.sumsqr(R_ori_Right[:, i] -  self.qaQb( R_ori_Left[:, i], quaternion_fixed180 )))
#            builder_wholebodyMPC_planner.add_cost_term('Right_arm_align' + str(i), 5*optas.sumsqr( self.skew_optas(self.quatToRotationZ(r_ori_RARM_var_MPC[:, i])) @ (r_pos_RARM_var_MPC[:, i] - r_pos_LARM_var_MPC[:, i])   ))
#            builder_wholebodyMPC_planner.add_cost_term('Left_arm_align' + str(i), 5*optas.sumsqr( self.skew_optas(self.quatToRotationZ(r_ori_LARM_var_MPC[:, i])) @ (r_pos_RARM_var_MPC[:, i] - r_pos_LARM_var_MPC[:, i])   ))
            builder_wholebodyMPC_planner.add_cost_term('Right_arm_align_x' + str(i), 50*optas.sumsqr( self.quatToRotationX(r_ori_RARM_var_MPC[:, i]).T @ (r_pos_RARM_var_MPC[:, i] - r_pos_LARM_var_MPC[:, i])   ))
            builder_wholebodyMPC_planner.add_cost_term('Right_arm_align_y' + str(i), 5*optas.sumsqr( self.quatToRotationY(r_ori_RARM_var_MPC[:, i]).T @ (r_pos_RARM_var_MPC[:, i] - r_pos_LARM_var_MPC[:, i])   ))


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
            obstacle_pos = np.asarray([[4], [0]])
            obstacle_radius = 0.3 + 1
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
                builder_wholebodyMPC_planner.add_cost_term('Right_distance_z' + str(i), 100 * optas.sumsqr(R_pos_Right[2, i+1] - R_pos_Right[2, i]))

                builder_wholebodyMPC_planner.add_cost_term('Left_distance_x' + str(i), 50 * optas.sumsqr(R_pos_Left[0, i+1] - R_pos_Left[0, i]))
                builder_wholebodyMPC_planner.add_cost_term('Left_distance_y' + str(i), 50 * optas.sumsqr(R_pos_Left[1, i+1] - R_pos_Left[1, i]))
                builder_wholebodyMPC_planner.add_cost_term('Left_distance_z' + str(i), 100 * optas.sumsqr(R_pos_Left[2, i+1] - R_pos_Left[2, i]))

                builder_wholebodyMPC_planner.add_cost_term('Right_ori_distance' + str(i),  20 * optas.sumsqr(R_ori_Right[:, i+1] - R_ori_Right[:, i]))
                builder_wholebodyMPC_planner.add_cost_term('Left_ori_distance' + str(i),   20 * optas.sumsqr(R_ori_Left[:, i+1]  - R_ori_Left[:, i]))
            if(i<(self.T_MPC_planner -2)):
#                builder_wholebodyMPC_planner.add_cost_term('dRight_distance' + str(i),     50 * optas.sumsqr(R_pos_Right[:, i+2]-2*R_pos_Right[:, i+1] + R_pos_Right[:, i]))
#                builder_wholebodyMPC_planner.add_cost_term('dLeft_distance' + str(i),      50 * optas.sumsqr(R_pos_Left[:, i+2] -2*R_pos_Left[:, i+1]  + R_pos_Left[:, i]))
                builder_wholebodyMPC_planner.add_cost_term('dRight_distance_x' + str(i), 50 * optas.sumsqr(R_pos_Right[0, i+2]-2*R_pos_Right[0, i+1] + R_pos_Right[0, i]))
                builder_wholebodyMPC_planner.add_cost_term('dRight_distance_y' + str(i), 50 * optas.sumsqr(R_pos_Right[1, i+2]-2*R_pos_Right[1, i+1] + R_pos_Right[1, i]))
                builder_wholebodyMPC_planner.add_cost_term('dRight_distance_z' + str(i), 100 * optas.sumsqr(R_pos_Right[2, i+2]-2*R_pos_Right[2, i+1] + R_pos_Right[2, i]))

                builder_wholebodyMPC_planner.add_cost_term('dLeft_distance_x' + str(i),  50 * optas.sumsqr(R_pos_Left[0, i+2]-2*R_pos_Left[0, i+1] + R_pos_Left[0, i]))
                builder_wholebodyMPC_planner.add_cost_term('dLeft_distance_y' + str(i),  50 * optas.sumsqr(R_pos_Left[1, i+2]-2*R_pos_Left[1, i+1] + R_pos_Left[1, i]))
                builder_wholebodyMPC_planner.add_cost_term('dLeft_distance_z' + str(i),  100 * optas.sumsqr(R_pos_Left[2, i+2]-2*R_pos_Left[2, i+1] + R_pos_Left[2, i]))

                builder_wholebodyMPC_planner.add_cost_term('dRight_ori_distance' + str(i), 20 * optas.sumsqr(R_ori_Right[:, i+2]-2*R_ori_Right[:, i+1] + R_ori_Right[:, i]))
                builder_wholebodyMPC_planner.add_cost_term('dLeft_ori_distance' + str(i),  20 * optas.sumsqr(R_ori_Left[:, i+2] -2*R_ori_Left[:, i+1]  + R_ori_Left[:, i]))

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
            builder_wholebodyMPC_planner.add_cost_term('minimize_dr_right' + str(i), w_dr * optas.sumsqr(dr_pos_RARM_var_MPC[:, i]))
            builder_wholebodyMPC_planner.add_cost_term('minimize_dr_left' + str(i), w_dr * optas.sumsqr(dr_pos_LARM_var_MPC[:, i]))
            builder_wholebodyMPC_planner.add_cost_term('minimize_dr_ori_right' + str(i), w_dr * optas.sumsqr(dr_ori_RARM_var_MPC[:, i]))
            builder_wholebodyMPC_planner.add_cost_term('minimize_dr_ori_left' + str(i), w_dr * optas.sumsqr(dr_ori_LARM_var_MPC[:, i]))


#        builder_wholebodyMPC_planner.add_equality_constraint('init_dr_middle', dr_middle_var_MPC[:, 0], rhs=0.5*(init_dr_position_Right + init_dr_position_Left))
        builder_wholebodyMPC_planner.add_equality_constraint('init_dr_Right', dr_pos_RARM_var_MPC[0:2, 0], rhs=init_dr_position_Right[0:2])
        builder_wholebodyMPC_planner.add_equality_constraint('init_dr_Left', dr_pos_LARM_var_MPC[0:2, 0], rhs=init_dr_position_Left[0:2])
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
            builder_wholebodyMPC_planner.add_cost_term('minimize_ddr_right' + str(i), w_ddr * optas.sumsqr(ddr_pos_RARM_var_MPC[:, i]))
            builder_wholebodyMPC_planner.add_cost_term('minimize_ddr_left' + str(i), w_ddr * optas.sumsqr(ddr_pos_LARM_var_MPC[:, i]))
            builder_wholebodyMPC_planner.add_cost_term('minimize_ddr_ori_right' + str(i), w_ddr * optas.sumsqr(ddr_ori_RARM_var_MPC[:, i]))
            builder_wholebodyMPC_planner.add_cost_term('minimize_ddr_ori_left' + str(i), w_ddr * optas.sumsqr(ddr_ori_LARM_var_MPC[:, i]))

#            builder_wholebodyMPC_planner.add_cost_term('minimize_ddr_two arm equal' + str(i), w_ddr * 10 * optas.sumsqr(ddr_pos_RARM_var_MPC[:, i] - ddr_pos_LARM_var_MPC[:, i]))


#        builder_wholebodyMPC_planner.add_equality_constraint('final_ddr_middle', ddr_middle_var_MPC[:,-1], rhs=np.zeros(3))
        builder_wholebodyMPC_planner.add_equality_constraint('final_ddr_right', ddr_pos_RARM_var_MPC[:,self.T_MPC_planner-1], rhs=np.zeros(3))
        builder_wholebodyMPC_planner.add_equality_constraint('final_ddr_left', ddr_pos_LARM_var_MPC[:,self.T_MPC_planner-1], rhs=np.zeros(3))
        builder_wholebodyMPC_planner.add_equality_constraint('final_ddr_ori_right', ddr_ori_RARM_var_MPC[:,self.T_MPC_planner-1], rhs=np.zeros(4))
        builder_wholebodyMPC_planner.add_equality_constraint('final_ddr_ori_left', ddr_ori_LARM_var_MPC[:,self.T_MPC_planner-1], rhs=np.zeros(4))

        lower_acceleration = np.array([-1., -1., -0.5]); upper_acceleration = np.array([1., 1., 0.5]);
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
                                                                                                       'knitro.FeasTol': 5e-4, 'knitro.OptTol': 5e-4, 'knitro.ftol':5e-4,
                                                                                                       'knitro.algorithm':3, 'knitro.linsolver':2,
#                                                                                                       'knitro.maxtime_real': 4.0e-3,
                                                                                                       'knitro.bar_initpt':3, 'knitro.bar_murule':4, 'knitro.bar_penaltycons': 1,
                                                                                                       'knitro.bar_penaltyrule':2, 'knitro.bar_switchrule':2, 'knitro.linesearch': 1
                                                                                                       } )
        self.solution_MPC_planner = None

        ### ---------------------------------------------------------
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
        stiffness = 1500;

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
#        pos_R = builder_wholebodyMPC.add_parameter('pos_R', 3, self.T_MPC)
#        ori_R = builder_wholebodyMPC.add_parameter('ori_R', 4, self.T_MPC)
#        pos_L = builder_wholebodyMPC.add_parameter('pos_L', 3, self.T_MPC)
#        ori_L = builder_wholebodyMPC.add_parameter('ori_L', 4, self.T_MPC)

#        pos_R_reasonal = optas.casadi.SX(np.zeros((3, self.T_MPC)))
#        pos_L_reasonal = optas.casadi.SX(np.zeros((3, self.T_MPC)))
        pos_R_reasonal = builder_wholebodyMPC.add_parameter('pos_R_reasonal', 3, self.T_MPC)
        pos_L_reasonal = builder_wholebodyMPC.add_parameter('pos_L_reasonal', 3, self.T_MPC)
        ori_R_reasonal = builder_wholebodyMPC.add_parameter('ori_R_reasonal', 4, self.T_MPC)
        ori_L_reasonal = builder_wholebodyMPC.add_parameter('ori_L_reasonal', 4, self.T_MPC)

        ddpos_box_goal = builder_wholebodyMPC.add_parameter('ddpos_box_goal', 3, self.T_MPC)
        m_box = builder_wholebodyMPC.add_parameter('m_box', 1)
        #####################################################################################
        self.F_ext_global_Right = np.zeros(6); self.F_ext_global_Left = np.zeros(6);
        self.F_ext_local_Right = np.zeros(6);  self.F_ext_local_Left = np.zeros(6);
        self.acc_box = np.zeros((3, self.T_MPC));
#        self.acc_box = np.zeros(3);
        F_ext_Right_goal = builder_wholebodyMPC.add_parameter('F_ext_Right_goal', 3, self.T_MPC)
        F_ext_Left_goal = builder_wholebodyMPC.add_parameter('F_ext_Left_goal', 3, self.T_MPC)
        F_ext_Right_actual_local = builder_wholebodyMPC.add_parameter('F_ext_Right_actual_local', 3)
        F_ext_Left_actual_local = builder_wholebodyMPC.add_parameter('F_ext_Left_actual_local', 3)
#        F_ext_Right_actual = builder_wholebodyMPC.add_parameter('F_ext_Right_actual', 3)
#        F_ext_Left_actual = builder_wholebodyMPC.add_parameter('F_ext_Left_actual', 3)
        F_ext_Right_var_MPC = optas.casadi.SX(np.zeros((3, self.T_MPC)))
        F_ext_Left_var_MPC = optas.casadi.SX(np.zeros((3, self.T_MPC)))
        #####################################################################################
        # functions of right and left arm positions
        self.pos_fnc_Right = self.wholebodyMPC.get_global_link_position_function(link=self._link_ee_right)
        self.pos_fnc_Left = self.wholebodyMPC.get_global_link_position_function(link=self._link_ee_left)
        self.pos_fnc_Right_Base = self.wholebodyMPC.get_link_position_function(link=self._link_ee_right, base_link = self._link_donkey)
        self.pos_fnc_Left_Base = self.wholebodyMPC.get_link_position_function(link=self._link_ee_left, base_link = self._link_donkey)
        self.pos_Jac_fnc_Right = self.wholebodyMPC.get_global_link_linear_jacobian_function(link=self._link_ee_right)
        self.pos_Jac_fnc_Left = self.wholebodyMPC.get_global_link_linear_jacobian_function(link=self._link_ee_left)
        # quaternion functions of two arm end effectors
        self.ori_fnc_Right = self.wholebodyMPC.get_global_link_quaternion_function(link=self._link_ee_right)
        self.ori_fnc_Left = self.wholebodyMPC.get_global_link_quaternion_function(link=self._link_ee_left)
        self.ori_fnc_donkey = self.wholebodyMPC.get_global_link_quaternion_function(link=self._link_donkey)
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
            F_ext_Right_var_MPC[:, i] = F_ext_Right_actual_local + inertia_Right @ ddDelta_p_Right_var_MPC[:, i] + self.K_Right @ Delta_p_Right_var_MPC[:, i] + self.D_Right @ dDelta_p_Right_var_MPC[:, i]
            F_ext_Left_var_MPC[:, i] = F_ext_Left_actual_local + inertia_Left @ ddDelta_p_Left_var_MPC[:, i] + self.K_Left @ Delta_p_Left_var_MPC[:, i] + self.D_Left @ dDelta_p_Left_var_MPC[:, i]
#            F_ext_Right_var_MPC[:, i] = F_ext_Right_actual + inertia_Right @ ddDelta_p_Right_var_MPC[:, i] + self.K_Right @ Delta_p_Right_var_MPC[:, i] + self.D_Right @ dDelta_p_Right_var_MPC[:, i]
#            F_ext_Left_var_MPC[:, i] = F_ext_Left_actual + inertia_Left @ ddDelta_p_Left_var_MPC[:, i] + self.K_Left @ Delta_p_Left_var_MPC[:, i] + self.D_Left @ dDelta_p_Left_var_MPC[:, i]
            builder_wholebodyMPC.add_bound_inequality_constraint('control_point_' + str(i) + '_bound', lhs=lower, mid=Q[:, i], rhs=upper)
            # optimization cost: close to target
            builder_wholebodyMPC.add_cost_term('Right_arm orientation' + str(i), optas.sumsqr(self.ori_fnc_Right(q_var_MPC[:, i])-ori_R_reasonal[:, i]))
            builder_wholebodyMPC.add_cost_term('Left_arm orientation' + str(i),  optas.sumsqr(self.ori_fnc_Left(q_var_MPC[:, i])-ori_L_reasonal[:, i]))
#            builder_wholebodyMPC.add_cost_term('Two_arm orientation parallel' + str(i), 0.1*optas.sumsqr(self.ori_fnc_Right(q_var_MPC[:, i]).T @ self.ori_fnc_Left(q_var_MPC[:, i])))
#            quaternion_donkey_start = optas.spatialmath.Quaternion.fromrpy([np.pi/2,    0,    0]).getquat()
#            builder_wholebodyMPC.add_cost_term('Two_arm orientation parallel with donkey plane' + str(i), 10*optas.sumsqr(self.ori_fnc_donkey(q_var_MPC[:, i]).__mul__(quaternion_donkey_start).T @ self.ori_fnc_Right(q_var_MPC[:, i]) ))

            builder_wholebodyMPC.add_cost_term('Right_arm position AD' + str(i), optas.sumsqr(self.pos_fnc_Right(q_var_MPC[:, i])-pos_R_reasonal[:, i] - self.rotation_fnc_Right(q_var_MPC[:, i]) @ Delta_p_Right_var_MPC[:, i]))
            builder_wholebodyMPC.add_cost_term('Left_arm position AD' + str(i),  optas.sumsqr(self.pos_fnc_Left(q_var_MPC[:, i])-pos_L_reasonal[:, i]  - self.rotation_fnc_Left(q_var_MPC[:, i])  @ Delta_p_Left_var_MPC[:, i]))

#            builder_wholebodyMPC.add_cost_term('Two_arm position forward relative to floatingbase x' + str(i), 5*optas.sumsqr(self.pos_fnc_Right_Base(q_var_MPC[:, i])[0]- self.pos_fnc_Left_Base(q_var_MPC[:, i])[0]))
#            builder_wholebodyMPC.add_cost_term('Two_arm position forward relative to floatingbase z' + str(i), 5*optas.sumsqr(self.pos_fnc_Right_Base(q_var_MPC[:, i])[2]- self.pos_fnc_Left_Base(q_var_MPC[:, i])[2]))

            #####################################################################################
            builder_wholebodyMPC.add_cost_term('Right_arm Force world y' + str(i), 0.1*optas.sumsqr((self.rotation_fnc_Right(init_position_MPC) @ (F_ext_Right_var_MPC[:, i] - F_ext_Right_goal[:, i]))[0:2] - 0.5*m_box * ddpos_box_goal[0:2, i]))
            builder_wholebodyMPC.add_cost_term('Left_arm Force world y' + str(i),  0.1*optas.sumsqr((self.rotation_fnc_Left(init_position_MPC) @ (F_ext_Left_var_MPC[:, i] - F_ext_Left_goal[:, i]))[0:2] - 0.5*m_box * ddpos_box_goal[0:2, i]))
#            builder_wholebodyMPC.add_cost_term('two Force sum world y' + str(i),   optas.sumsqr(F_ext_Right_var_MPC[1, i] + F_ext_Left_var_MPC[1, i] - F_ext_Right_goal[1, i] - F_ext_Left_goal[1, i] ))
#            builder_wholebodyMPC.add_cost_term('Two arm ee addition motion equal' + str(i), optas.sumsqr(Delta_p_Right_var_MPC[1, i] + Delta_p_Left_var_MPC[1, i]))
#            builder_wholebodyMPC.add_cost_term('Two force Delta for box inertial force' + str(i),  optas.sumsqr( ((self.rotation_fnc_Right(init_position_MPC) @ F_ext_Right_var_MPC[:, i] + self.rotation_fnc_Left(init_position_MPC) @ F_ext_Left_var_MPC[:, i]))[0:2] - m_box * ddpos_box_goal[0:2, i]))
#            builder_wholebodyMPC.add_bound_inequality_constraint('right_force_limit' + str(i) + '_bound', lhs=-50, mid=F_ext_Right_var_MPC[1, i], rhs=50)
#            builder_wholebodyMPC.add_bound_inequality_constraint('left_force_limit' + str(i) + '_bound', lhs=-50, mid=F_ext_Left_var_MPC[1, i], rhs=50)
            #####################################################################################
            builder_wholebodyMPC.add_cost_term('twoarm_miniscope' + str(i), 0.1 * optas.sumsqr(q_var_MPC[6, i] + q_var_MPC[12, i]))
            builder_wholebodyMPC.add_cost_term('chest_miniscope' + str(i), 10* optas.sumsqr(q_var_MPC[3, i]))
            builder_wholebodyMPC.add_cost_term('arm_joint_miniscope' + str(i), 0.001 * optas.sumsqr(q_var_MPC[6:self.ndof, i]))
#            builder_wholebodyMPC.add_cost_term('donkey_yaw_miniscope' + str(i), 0.001 * optas.sumsqr(q_var_MPC[2, i]))
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
        builder_wholebodyMPC.add_equality_constraint('Delta_p_Right_var_MPC_non_motion_direction_z', P_Right[1, :], rhs=np.zeros((1, self.T_MPC)))
        builder_wholebodyMPC.add_equality_constraint('Delta_p_Left_var_MPC_non_motion_direction_x', P_Left[0, :], rhs=np.zeros((1, self.T_MPC)))
        builder_wholebodyMPC.add_equality_constraint('Delta_p_Left_var_MPC_non_motion_direction_z', P_Left[1, :], rhs=np.zeros((1, self.T_MPC)))
        builder_wholebodyMPC.add_equality_constraint('init_Delta_position_Right_constraint_y', P_Right[2, 0], rhs = 0 )
        builder_wholebodyMPC.add_equality_constraint('init_Delta_position_Left_constraint_y',  P_Left[2, 0],  rhs = 0 )
        #########################################################################################
        dq_var_MPC = optas.casadi.SX(np.zeros((self.ndof, self.T_MPC)))
        w_dq = self.duration_MPC**2 * 0.05/float(self.T_MPC)
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
        w_ddq = self.duration_MPC**4 * 0.05/float(self.T_MPC)
        for i in range(self.T_MPC):
            for j in range(self.T_MPC-2):
                ddq_var_MPC[:, i] += (1./self.duration_MPC)**2 * self.BC(self.n-2, j) * t[i]**j * (1-t[i])**(self.n-2-j) * self.n * (self.n-1)* (Q[:, j+2] -  2*Q[:, j+1] + Q[:, j])
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
        acc_box_var[:, self.T_MPC-1] = acc_box_var[:, self.T_MPC-2]
        #########################################################################################
        # setup solver
#        self.solver_wholebodyMPC = optas.CasADiSolver(optimization=builder_wholebodyMPC.build()).setup('knitro')
        # self.solver_wholebodyMPC = optas.CasADiSolver(optimization=builder_wholebodyMPC.build()).setup('knitro', solver_options={'knitro.OutLev': 10} )
        # self.solver_wholebodyMPC = optas.CasADiSolver(optimization=builder_wholebodyMPC.build()).setup('knitro', solver_options={'knitro.OutLev': 0, 'print_time': 0} )
        self.solver_wholebodyMPC = optas.CasADiSolver(optimization=builder_wholebodyMPC.build()).setup('knitro', solver_options={
                                                                                                       'knitro.OutLev': 0,
                                                                                                       'print_time': 0,
                                                                                                       'knitro.FeasTol': 1e-5, 'knitro.OptTol': 1e-5, 'knitro.ftol':1e-5,
                                                                                                       'knitro.algorithm':1,
                                                                                                       'knitro.linsolver':2,
#                                                                                                       'knitro.maxtime_real': 1.8e-2,
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
        # declare joint subscriber
        self._joint_sub = rospy.Subscriber("/chonk/joint_states", JointState, self.read_joint_states_cb)
#        self._joint_sub_base = rospy.Subscriber("/chonk/donkey_velocity_controller/odom", Odometry, self.read_base_states_cb)
        self._joint_sub_base = rospy.Subscriber("/chonk/base_pose_ground_truth", Odometry, self.read_base_states_cb)
        # declare joint publisher
        self._joint_pub = rospy.Publisher("/chonk/trajectory_controller/command", JointTrajectory, queue_size=10)
        # declare acceleration publisher for two arms
        self._joint_acc_pub = rospy.Publisher("/chonk/joint_acc_pub", Float64MultiArray, queue_size=10)
        # This is for donkey_velocity_controller
        self._joint_pub_velocity = rospy.Publisher("/chonk/donkey_velocity_controller/cmd_vel", Twist, queue_size=10)
        # declare two arm ee grasp actual force publisher
#        self._sensor_ft_sub_right = rospy.Subscriber("/chonk/sensor_ft_right", Float64MultiArray, self.read_right_ee_grasp_ft_data_cb)
#        self._sensor_ft_sub_left = rospy.Subscriber("/chonk/sensor_ft_left", Float64MultiArray, self.read_left_ee_grasp_ft_data_cb)
        self._sensor_ft_sub_local_right = rospy.Subscriber("/chonk/sensor_ft_local_right", Float64MultiArray, self.read_right_ee_grasp_ft_local_data_cb)
        self._sensor_ft_sub_local_left = rospy.Subscriber("/chonk/sensor_ft_local_left", Float64MultiArray, self.read_left_ee_grasp_ft_local_data_cb)
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
        self.pos_Right = np.asarray([acceped_goal.poseR.position.x, acceped_goal.poseR.position.y, acceped_goal.poseR.position.z])
        self.pos_Left = np.asarray([acceped_goal.poseL.position.x, acceped_goal.poseL.position.y, acceped_goal.poseL.position.z])
        self.ori_Right = np.asarray([acceped_goal.poseR.orientation.x, acceped_goal.poseR.orientation.y, acceped_goal.poseR.orientation.z, acceped_goal.poseR.orientation.w])
        self.ori_Left = np.asarray([acceped_goal.poseL.orientation.x, acceped_goal.poseL.orientation.y, acceped_goal.poseL.orientation.z, acceped_goal.poseL.orientation.w])
        self.force_Right = np.asarray([acceped_goal.ForceTorqueR.force.x, acceped_goal.ForceTorqueR.force.y, acceped_goal.ForceTorqueR.force.z])
        self.torque_Right = np.asarray([acceped_goal.ForceTorqueR.torque.x, acceped_goal.ForceTorqueR.torque.y, acceped_goal.ForceTorqueR.torque.z])
        self.force_Left = np.asarray([acceped_goal.ForceTorqueL.force.x, acceped_goal.ForceTorqueL.force.y, acceped_goal.ForceTorqueL.force.z])
        self.torque_Left = np.asarray([acceped_goal.ForceTorqueL.torque.x, acceped_goal.ForceTorqueL.torque.y, acceped_goal.ForceTorqueL.torque.z])
        # print goal request
        rospy.loginfo("%s: Request to send right arm to position (%.2f, %.2f, %.2f) with orientation (%.2f, %.2f, %.2f, %.2f), and left arm to position (%.2f, %.2f, %.2f) with orientation (%.2f, %.2f, %.2f, %.2f) in %.1f seconds." % (
                self._name, self.pos_Right[0], self.pos_Right[1], self.pos_Right[2], self.ori_Right[0], self.ori_Right[1], self.ori_Right[2], self.ori_Right[3],
                self.pos_Left[0], self.pos_Left[1], self.pos_Left[2], self.ori_Left[0], self.ori_Left[1], self.ori_Left[2], self.ori_Left[3], acceped_goal.duration))
        # read current robot joint positions
        self.q_curr = np.concatenate((self.q_curr_base, self.q_curr_joint), axis=None)
        self.dq_curr = np.concatenate((self.dq_curr_base, self.dq_curr_joint), axis=None)
        qT = np.zeros(self.ndof)
        self.joint_names = self.joint_names_base + self.joint_names_position
        ### ---------------------------------------------------------
        # get two-arm end effector trajectory in the operational space
        q0 = self.q_curr.T
        self.duration = acceped_goal.duration
        self.duration_MPC_planner = acceped_goal.duration
        self._steps = int(self.duration * self._freq)
        self._idx = 0
        # current right and left arm end effector position and quaternion
        self.start_RARM_quat = np.asarray(self.wholebodyMPC.get_global_link_quaternion(link=self._link_ee_right, q=q0)).T[0]
        self.start_RARM_pos = np.asarray(self.wholebodyMPC.get_global_link_position(link=self._link_ee_right, q=q0)).T[0]
        self.start_LARM_quat = np.asarray(self.wholebodyMPC.get_global_link_quaternion(link=self._link_ee_left, q=q0)).T[0]
        self.start_LARM_pos = np.asarray(self.wholebodyMPC.get_global_link_position(link=self._link_ee_left, q=q0)).T[0]
        # derivation of right and left arm end effector position and quaternion compared with the beginning ee position and quaternion
        Derivation_RARM_Pos = self.pos_Right - self.start_RARM_pos; Derivation_RARM_Quat = self.ori_Right - self.start_RARM_quat;
        Derivation_LARM_Pos = self.pos_Left - self.start_LARM_pos;  Derivation_LARM_Quat = self.ori_Left - self.start_LARM_quat;
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
        self.curr_MPC = np.zeros((self.ndof, self.T_MPC))
        for i in range(self.T_MPC):
            self.curr_MPC[:,i] = self.q_curr

        # create timer
        dur = rospy.Duration(1.0/self._freq)
        self._timer = rospy.Timer(dur, self.timer_cb)



    def timer_cb(self, event):
        # make sure that the action is active
        if(not self._action_server.is_active()):
            self._timer.shutdown()
            rospy.logwarn("%s: The action server is NOT active!")
            self._result.reached_goal = False
            self._action_server.set_aborted(self._result)
            return

        # main execution
        # main execution
        if(self._idx < self._steps):
            if(self._correct_mux_selection):
                # increment idx (in here counts starts with 1)
                self._idx += 1

                # read current robot joint positions
                self.q_curr = np.concatenate((self.q_curr_base, self.q_curr_joint), axis=None)
                self.dq_curr = np.concatenate((self.dq_curr_base, self.dq_curr_joint), axis=None)

                r_pos_actual_Right = np.array(self.pos_fnc_Right_planner(self.q_curr))[0:3]
                r_pos_actual_Left = np.array(self.pos_fnc_Left_planner(self.q_curr))[0:3]
                dr_pos_actual_Right = np.asarray(self.pos_Jac_fnc_Right_planner(self.q_curr)) @ self.dq_curr
                dr_pos_actual_Left = np.asarray(self.pos_Jac_fnc_Left_planner(self.q_curr)) @ self.dq_curr

                r_ori_actual_Right = np.array(self.ori_fnc_Right_planner(self.q_curr))[0:4]
                r_ori_actual_Left = np.array(self.ori_fnc_Left_planner(self.q_curr))[0:4]
                dr_ori_actual_Right = self.angular_velocity_to_quaternionRate(r_ori_actual_Right[:, 0]) @ np.asarray(self.ori_Jac_fnc_Right_planner(self.q_curr)) @ self.dq_curr
                dr_ori_actual_Left = self.angular_velocity_to_quaternionRate(r_ori_actual_Left[:, 0]) @ np.asarray(self.ori_Jac_fnc_Left_planner(self.q_curr)) @ self.dq_curr

#                print(r_ori_actual_Right)
#                print(r_ori_actual_Left)
#                print(r_ori_actual_Right.T @ r_ori_actual_Left)
                ### -----------------------------------------------------------
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
#                print(self._t[self._idx-1])
#                print(self.duration_MPC_planner)
#                print(R_ori_Right)
#                print(pos_R_reasonal[2, :])
#                print('left')
#                print(R_pos_Left[2, :])
#                print(pos_L_reasonal[2, :])
                ### -----------------------------------------------------------
                self.ti_MPC = 0
                force_R_goal = []
                force_L_goal = []
#                ori_R_goal = []
#                ori_L_goal = []
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
#                        g_rarm_ee_ori = self._RARM_ee_Quat_trajectory(self.ti_MPC).flatten()
#                        ori_R_goal.append(g_rarm_ee_ori.tolist())
#                        g_larm_ee_ori = self._LARM_ee_Quat_trajectory(self.ti_MPC).flatten()
#                        ori_L_goal.append(g_larm_ee_ori.tolist())
                    except ValueError:
                        force_R_goal.append(g_rarm_ee_force.tolist()) # i.e. previous goal
                        force_L_goal.append(g_larm_ee_force.tolist()) # i.e. previous goal
#                        ori_L_goal.append(g_larm_ee_ori.tolist())     # i.e. previous goal
#                        ori_R_goal.append(g_rarm_ee_ori.tolist())     # i.e. previous goal

                force_R_goal = optas.np.array(force_R_goal).T
                force_L_goal = optas.np.array(force_L_goal).T
#                ori_R_goal = optas.np.array(ori_R_goal).T
#                ori_L_goal = optas.np.array(ori_L_goal).T

                # read current robot joint positions
#                self.q_curr = np.concatenate((self.q_curr_base, self.q_curr_joint), axis=None)
#                self.dq_curr = np.concatenate((self.dq_curr_base, self.dq_curr_joint), axis=None)

                self.G_Rotation_ee_right = self.rotation_fnc_Right(self.q_curr)
                self.G_Rotation_ee_left = self.rotation_fnc_Left(self.q_curr)

                self.G_X_ee_right[0:3, 0:3] = self.G_Rotation_ee_right; self.G_X_ee_right[3:6, 3:6] = self.G_Rotation_ee_right
                self.G_X_ee_left[0:3, 0:3] = self.G_Rotation_ee_left;   self.G_X_ee_left[3:6, 3:6] = self.G_Rotation_ee_left

                self.G_I_ee_r_conventional = self.G_X_ee_right @ self.I_ee_r_conventional @ self.G_X_ee_right.T;
                self.G_I_ee_l_conventional = self.G_X_ee_left @ self.I_ee_l_conventional @ self.G_X_ee_left.T;
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

                self.solver_wholebodyMPC.reset_parameters({'pos_R_reasonal': pos_R_reasonal, 'pos_L_reasonal': pos_L_reasonal,
                                                           'ori_R_reasonal': ori_R_reasonal, 'ori_L_reasonal': ori_L_reasonal,
                                                           't': self.timebyT, 'init_position_MPC': self.q_curr, 'init_velocity_MPC': self.dq_curr,
                                                           'F_ext_Right_goal': force_R_goal, 'F_ext_Left_goal': force_L_goal,
                                                           'inertia_Right': self.G_I_ee_r_conventional[3:6, 3:6], 'inertia_Left': self.G_I_ee_l_conventional[3:6, 3:6],
#                                                           'F_ext_Right_actual': self.F_ext_global_Right[3:6], 'F_ext_Left_actual': self.F_ext_global_Left[3:6],
                                                           'F_ext_Right_actual_local': self.F_ext_local_Right[3:6], 'F_ext_Left_actual_local': self.F_ext_local_Left[3:6],
                                                           'init_Delta_position_Right': self.Derivation_RARM_pos_start, 'init_Delta_position_Left': self.Derivation_LARM_pos_start,
                                                           'ddpos_box_goal': self.acc_box, 'm_box': self.m_box } )

                # solve problem
                self.solution_MPC = self.solver_wholebodyMPC.opt.decision_variables.vec2dict(self.solver_wholebodyMPC._solve())
                Q = np.asarray(self.solution_MPC[f'Q'])
                P_Right = np.asarray(self.solution_MPC[f'P_Right'])
                P_Left = np.asarray(self.solution_MPC[f'P_Left'])
                self.acc_box = np.asarray(self.solution_MPC[f'acc_box_var'])
                ### ---------------------------------------------------------
                # compute next configuration with lambda function
                t = (1./self._freq)/self.duration_MPC; n = self.T_MPC -1;
                self.q_next = np.zeros(self.ndof); p_right = np.zeros(3); dp_right = np.zeros(3); ddp_right = np.zeros(3);
                for j in range(self.T_MPC):
                    self.q_next += self.BC(n, j) * t**j * (1-t)**(n-j) * Q[:, j]
                    p_right += self.BC(n, j) * t**j * (1-t)**(n-j) * P_Right[:, j]
                for j in range(self.T_MPC-1):
                    dp_right += (1./self.duration_MPC) * self.BC(self.n-1, j) * t**j * (1-t)**(self.n-1-j) * self.n * (P_Right[:, j+1] -  P_Right[:, j])
                for j in range(self.T_MPC-2):
                    ddp_right += (1./self.duration_MPC)**2 * self.BC(self.n-2, j) * t**j * (1-t)**(self.n-2-j) * self.n * (self.n-1)* (P_Right[:, j+2] -  2*P_Right[:, j+1] + P_Right[:, j])

                self.dq_next = np.zeros(self.ndof)
                for j in range(self.T_MPC-1):
                    self.dq_next += (1./self.duration_MPC) * self.BC(n-1, j) * t**j * (1-t)**(n-1-j) * n * (Q[:, j+1] -  Q[:, j])

                self.ddq_next = np.zeros(self.ndof)
                for j in range(self.T_MPC-2):
                    self.ddq_next += (1./self.duration_MPC)**2 * self.BC(n-2, j) * t**j * (1-t)**(n-2-j) * n * (n-1)* (Q[:, j+2] -  2*Q[:, j+1] + Q[:, j])

                # read current robot joint positions
                self.q_curr = np.concatenate((self.q_curr_base, self.q_curr_joint), axis=None)
                self.Derivation_RARM_pos_start[1] = np.asarray(self.pos_fnc_Right(self.q_next)).T[0][1] - np.asarray(self.pos_fnc_Right(self.q_curr)).T[0][1]
                self.Derivation_LARM_pos_start[1] = np.asarray(self.pos_fnc_Left(self.q_next)).T[0][1] - np.asarray(self.pos_fnc_Left(self.q_curr)).T[0][1]

                # compute the donkey velocity in its local frame
                Global_w_b = np.asarray([0., 0., self.dq_next[2]])
                Global_v_b = np.asarray([self.dq_next[0], self.dq_next[1], 0.])
                Local_w_b = self.donkey_R.T @ Global_w_b
                Local_v_b = self.donkey_R.T @ Global_v_b

                self.duration_MPC_planner = self.duration - self._idx/self._freq

#                self.eva_trajectory.header.stamp = rospy.Time.now()
                self.eva_trajectory.header.stamp = rospy.Time(0)

                self.eva_trajectory.points[0].positions = self.q_next[-self.ndof_position_control:].tolist()
                # update message
                self._msg.data = self.q_next[-self.ndof_position_control:]
                self._msg_velocity.linear.x = Local_v_b[0]; self._msg_velocity.linear.y = Local_v_b[1]; self._msg_velocity.angular.z = Local_w_b[2];
                self._msg_acceleration.data = self.ddq_next[-self.ndof:]
                # publish message
                self._joint_pub.publish(self.eva_trajectory)
#                self._joint_pub.publish(self._msg)
                self._joint_pub_velocity.publish(self._msg_velocity)
                self._joint_acc_pub.publish(self._msg_acceleration)

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
            self.start_RARM_torque = self._RARM_ee_torque_trajectory(self.duration)
            self.start_LARM_force = self._LARM_ee_force_trajectory(self.duration)
            self.start_LARM_torque = self._LARM_ee_torque_trajectory(self.duration)

            return

    def read_joint_states_cb(self, msg):
        self.q_curr_joint = np.asarray(list(msg.position)[:self.ndof_position_control])
        self.joint_names_position = msg.name[:self.ndof_position_control]
        self.dq_curr_joint = np.asarray(list(msg.velocity)[:self.ndof_position_control])

    def read_base_states_cb(self, msg):
        base_euler_angle = tf.transformations.euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        self.q_curr_base = [msg.pose.pose.position.x, msg.pose.pose.position.y, base_euler_angle[2]]
        self.donkey_R = optas.spatialmath.rotz(base_euler_angle[2])
        self.donkey_position = np.asarray([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
        self.donkey_velocity = np.asarray([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
        self.donkey_angular_velocity = np.asarray([msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z])
        self.dq_curr_base = [float(msg.twist.twist.linear.x), float(msg.twist.twist.linear.y), float(msg.twist.twist.angular.z)]

#    def read_right_ee_grasp_ft_data_cb(self, msg):
#        self.F_ext_global_Right = np.asarray([ msg.data[0], msg.data[1], msg.data[2], 0, msg.data[4], 0])

#    def read_left_ee_grasp_ft_data_cb(self, msg):
#        self.F_ext_global_Left = np.asarray([ msg.data[0], msg.data[1], msg.data[2], 0, msg.data[4], 0 ])

    def read_right_ee_grasp_ft_local_data_cb(self, msg):
        self.F_ext_local_Right = np.asarray([ msg.data[0], msg.data[1], msg.data[2], 0, 0, msg.data[5]])

    def read_left_ee_grasp_ft_local_data_cb(self, msg):
        self.F_ext_local_Left = np.asarray([ msg.data[0], msg.data[1], msg.data[2], 0, 0, msg.data[5] ])

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

    def skew_optas(self, vec):
        A = optas.casadi.SX(np.zeros((3, 3)))
        A[0, 0] =  0;      A[0, 1] = -vec[2]; A[0, 2] =  vec[1];
        A[1, 0] =  vec[2]; A[1, 1] = 0;       A[1, 2] = -vec[0];
        A[2, 0] = -vec[1]; A[2, 1] = vec[0];  A[2, 2] =  0;
        return A

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
        
        

if __name__=="__main__":
    # Initialize node
    rospy.init_node("cmd_pose_server_MPC_BC_operational_AD", anonymous=True)
    # Initialize node class
    cmd_pose_server = CmdPoseActionServer(rospy.get_name())
    # executing node
    rospy.spin()
