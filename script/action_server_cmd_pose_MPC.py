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
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension
from pushing_msgs.msg import CmdChonkPoseAction, CmdChonkPoseFeedback, CmdChonkPoseResult
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from urdf_parser_py.urdf import URDF
import tf

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
        # set up whole-body motion optimization
        wholebodyPlanner = optas.RobotModel(
            urdf_string=self._robot_description,
            time_derivs=[0],
            param_joints=[],
            name='chonk_wholebodyPlanner'
        )
        self.wholebodyPlanner_name = wholebodyPlanner.get_name()
        self.ndof = wholebodyPlanner.ndof
        # nominal robot configuration
        self.q_nom = optas.DM.zeros(self.ndof)
        self.wholebodyPlanner_opt_idx = wholebodyPlanner.optimized_joint_indexes
        self.wholebodyPlanner_param_idx = wholebodyPlanner.parameter_joint_indexes
        # set up optimization builder
        builder_wholebodyPlanner = optas.OptimizationBuilder(T=1, robots=[wholebodyPlanner])
        # get robot state and parameters variables
        q_var = builder_wholebodyPlanner.get_robot_states_and_parameters(self.wholebodyPlanner_name)
        # get end-effector pose as parameters
        pos_R = builder_wholebodyPlanner.add_parameter('pos_R', 3)
        ori_R = builder_wholebodyPlanner.add_parameter('ori_R', 4)
        pos_L = builder_wholebodyPlanner.add_parameter('pos_L', 3)
        ori_L = builder_wholebodyPlanner.add_parameter('ori_L', 4)
        # set variable boudaries
        builder_wholebodyPlanner.enforce_model_limits(self.wholebodyPlanner_name)
        # equality constraint on right and left arm positions
        pos_fnc_Right = wholebodyPlanner.get_global_link_position_function(link=self._link_ee_right)
        pos_fnc_Left = wholebodyPlanner.get_global_link_position_function(link=self._link_ee_left)
        builder_wholebodyPlanner.add_equality_constraint('final_pos_Right', pos_fnc_Right(q_var), rhs=pos_R)
        builder_wholebodyPlanner.add_equality_constraint('final_pos_Left', pos_fnc_Left(q_var), rhs=pos_L)
        # rotation of the right and left arms
        self.Rotation_fnc_Right = wholebodyPlanner.get_global_link_rotation_function(link=self._link_ee_right)
        self.Rotation_fnc_Left = wholebodyPlanner.get_global_link_rotation_function(link=self._link_ee_left)
        # equality constraint on orientations
        ori_fnc_Right = wholebodyPlanner.get_global_link_quaternion_function(link=self._link_ee_right)
        ori_fnc_Left = wholebodyPlanner.get_global_link_quaternion_function(link=self._link_ee_left)
        builder_wholebodyPlanner.add_equality_constraint('final_ori_Right', ori_fnc_Right(q_var), rhs=ori_R)
        builder_wholebodyPlanner.add_equality_constraint('final_ori_Left', ori_fnc_Left(q_var), rhs=ori_L)
        # optimization cost: close to nominal config
        builder_wholebodyPlanner.add_cost_term('nom_config', optas.sumsqr(q_var-self.q_nom))
        # setup solver
        self.solver_wholebodyPlanner = optas.CasADiSolver(optimization=builder_wholebodyPlanner.build()).setup('ipopt')
        ### ---------------------------------------------------------
        # set up whole-body MPC
        wholebodyMPC = optas.RobotModel(
            urdf_string=self._robot_description,
            time_derivs=[0, 1],
            param_joints=[],
            name='chonk_wholebodyMPC'
        )
        self.wholebodyMPC_name = wholebodyMPC.get_name()
        self.dt_MPC = 0.02 # time step
        self.T_MPC = 8 # T is number of time steps
        self.duration_MPC = float(self.T_MPC-1)*self.dt_MPC
        # nominal robot configuration
        self.wholebodyMPC_opt_idx = wholebodyMPC.optimized_joint_indexes
        self.wholebodyMPC_param_idx = wholebodyMPC.parameter_joint_indexes
        # set up optimization builder.
        builder_wholebodyMPC = optas.OptimizationBuilder(T=self.T_MPC, robots=[wholebodyMPC], derivs_align=True)
        # get robot state variables, get velocity state variables
        q_var_MPC = builder_wholebodyMPC.get_robot_states_and_parameters(self.wholebodyMPC_name)
        dq_var_MPC = builder_wholebodyMPC.get_model_states(self.wholebodyMPC_name, time_deriv=1)
        p_epsilon = builder_wholebodyMPC.add_decision_variables('p_epsilon', self.ndof)
#        v_epsilon = builder_wholebodyMPC.add_decision_variables('v_epsilon', self.ndof)
        # set variable boudaries
        builder_wholebodyMPC.enforce_model_limits(self.wholebodyMPC_name, time_deriv=0)
        builder_wholebodyMPC.enforce_model_limits(self.wholebodyMPC_name, time_deriv=1)
        # Add parameters
        init_position_MPC = builder_wholebodyMPC.add_parameter('init_position_MPC', self.ndof)  # initial robot position
        init_velocity_MPC = builder_wholebodyMPC.add_parameter('init_velocity_MPC', self.ndof)  # initial robot velocity
        goal = builder_wholebodyMPC.add_parameter('goal', self.ndof, self.T_MPC)  # trajectory points
        dgoal = builder_wholebodyMPC.add_parameter('dgoal', self.ndof, self.T_MPC)  # trajectory point velocities
        # Constraint: dynamics
        builder_wholebodyMPC.integrate_model_states(self.wholebodyMPC_name, time_deriv=1, dt=self.dt_MPC)
        builder_wholebodyMPC.add_bound_inequality_constraint('init_p_equal', lhs=init_position_MPC - np.full((self.ndof, 1), 0.01), mid=q_var_MPC[0:self.ndof, 0] + p_epsilon, rhs=init_position_MPC + np.full((self.ndof, 1), 0.01))
#        builder_wholebodyMPC.add_bound_inequality_constraint('init_v_equal', lhs=init_velocity_MPC - np.full((self.ndof, 1), 0.01), mid=dq_var_MPC[0:self.ndof, 0] + v_epsilon, rhs=init_velocity_MPC + np.full((self.ndof, 1), 0.01))

        # Constraint: initial state
#        builder_wholebodyMPC.fix_configuration(self.wholebodyMPC_name, config=init_position_MPC)
#        builder_wholebodyMPC.fix_configuration(self.wholebodyMPC_name, config=init_velocity_MPC, time_deriv=1)
        # optimization cost: close to nominal config        
        builder_wholebodyMPC.add_cost_term('goal_MPC', optas.sumsqr(q_var_MPC - goal))
        builder_wholebodyMPC.add_cost_term('dgoal_MPC', optas.sumsqr(dq_var_MPC - dgoal))
        builder_wholebodyMPC.add_cost_term('minimize_p_epsilon', optas.sumsqr(p_epsilon))
#        builder_wholebodyMPC.add_cost_term('minimize_v_epsilon', optas.sumsqr(v_epsilon))
        # Cost: minimize velocity
#        w_dq = 0.01/float(self.T_MPC)  # weight on cost term
#        builder_wholebodyMPC.add_cost_term('minimize_velocity', w_dq*optas.sumsqr(dq_var_MPC))
        # Cost: minimize acceleration
#        w_ddq = 0.0025/float(self.T_MPC)  # weight on cost term
#        ddq_var_MPC = (dq_var_MPC[:, 1:] - dq_var_MPC[:, :-1])/self.dt_MPC
#        builder_wholebodyMPC.add_cost_term('minimize_acceleration', w_ddq*optas.sumsqr(ddq_var_MPC))
        # setup solver
#        solver_options={'ipopt.print_level': 0, 'print_time': 0}
#        self.solver_wholebodyMPC = optas.CasADiSolver(optimization=builder_wholebodyMPC.build()).setup('knitro', solver_options)
        self.solver_wholebodyMPC = optas.CasADiSolver(optimization=builder_wholebodyMPC.build()).setup('knitro', solver_options={'knitro.OutLev': 0})
#        self.solver_wholebodyMPC = optas.CasADiSolver(optimization=builder_wholebodyMPC.build()).setup('knitro', solver_options={'knitro.OutLev': 0, 'print_time': 0} )

#        self.solver_wholebodyMPC = optas.CasADiSolver(optimization=builder_wholebodyMPC.build()).setup('ipopt')
        self.ti_MPC = 0 # time index of the MPC
        self.solution_MPC = None
        ### ---------------------------------------------------------
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
        self._joint_sub_base = rospy.Subscriber(
            "/chonk/base_pose_ground_truth",
            Odometry,
            self.read_base_states_cb
        )
        # declare joint publisher
        self._joint_pub = rospy.Publisher(
            self._pub_cmd_topic_name,
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
        self.q_curr = np.concatenate((self.q_curr_base, self.q_curr_joint), axis=None)
        self.dq_curr = np.concatenate((self.dq_curr_base, self.dq_curr_joint), axis=None)
        qT = np.zeros(self.ndof)
        self.joint_names = self.joint_names_base + self.joint_names_position
        ### optas
        ### ---------------------------------------------------------
        ### solve whole-body problem
        # set initial seed
        self.solver_wholebodyPlanner.reset_initial_seed({f'{self.wholebodyPlanner_name}/q': self.q_nom[self.wholebodyPlanner_opt_idx]})
        self.solver_wholebodyPlanner.reset_parameters({'pos_R': pos_Right, 'ori_R': ori_Right, 'pos_L': pos_Left, 'ori_L': ori_Left, f'{self.wholebodyPlanner_name}/P': self.q_nom[self.wholebodyPlanner_param_idx]})
#        self.solver_wholebody.reset_parameters({'pos_R': pos_Right, 'ori_R': ori_Right, 'pos_L': pos_Left, f'{self.wholebody_name}/P': self.q_nom[self.wholebody_param_idx]})
        # solve problem
        solution = self.solver_wholebodyPlanner.solve()
        qT_array = solution[f'{self.wholebodyPlanner_name}/q']
        # save solution
        qT[self.wholebodyPlanner_opt_idx] = np.asarray(qT_array).T[0]
        qT[self.wholebodyPlanner_param_idx] = np.asarray(self.q_nom[self.wholebodyPlanner_param_idx]).T[0]
        ### ---------------------------------------------------------
        # helper variables
        q0 = self.q_curr
        self.duration = acceped_goal.duration
        self._steps = int(self.duration * self._freq)
        self._idx = 0
        Dq = qT - q0
        # interpolate between current and target configuration polynomial obtained for zero speed (3rd order) and acceleratin (5th order)
        # at the initial and final time
        # self._q = lambda t: q0 + (3.*((t/self.duration)**2) - 2.*((t/self.duration)**3))*Dq # 3rd order
        self._q = lambda t: q0 + (10.*((t/self.duration)**3) - 15.*((t/self.duration)**4) + 6.*((t/self.duration)**5))*Dq # 5th order
        self._dq = lambda t: (30.*((t/self.duration)**2) - 60.*((t/self.duration)**3) +30.*((t/self.duration)**4))*(Dq/self.duration)
        self._t = np.linspace(0., self.duration, self._steps + 1)
        ### ---------------------------------------------------------
        # print goal request
        rospy.loginfo("%s: Request to send robot joints to %s in %.1f seconds." % (self._name, qT, self.duration))
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

    def timer_cb(self, event):
        """ Publish the robot configuration """

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
                # read current robot joint positions
                goal = []
                dgoal = []

                for i in range(self.T_MPC):
                    self.ti_MPC = self._t[self._idx-1]  + self.dt_MPC*i
                    try:
                        g = self._q(self.ti_MPC).flatten()
                        goal.append(g.tolist())
                    except ValueError:
                        goal.append(g.tolist()) # i.e. previous goal

                for i in range(self.T_MPC):
                    self.ti_MPC = self._t[self._idx-1]  + self.dt_MPC*i
                    try:
                        dg = self._dq(self.ti_MPC).flatten()
                        dgoal.append(dg.tolist())
                    except ValueError:
                        dgoal.append(dg.tolist())

                goal = optas.np.array(goal).T
                dgoal = optas.np.array(dgoal).T
                ### optas
                ### ---------------------------------------------------------
                ### solve the whole-body MPC
                # set initial seed
                if self.solution_MPC is None:
#                    self.solver_wholebodyMPC.reset_initial_seed({f'{self.wholebodyMPC_name}/q': np.zeros((self.ndof, self.T_MPC)), f'{self.wholebodyMPC_name}/dq': np.zeros((self.ndof, self.T_MPC)), f'p_epsilon': np.zeros(self.ndof), f'v_epsilon': np.zeros(self.ndof)} )
                    self.solver_wholebodyMPC.reset_initial_seed({f'{self.wholebodyMPC_name}/q': np.zeros((self.ndof, self.T_MPC)), f'{self.wholebodyMPC_name}/dq': np.zeros((self.ndof, self.T_MPC)), f'p_epsilon': np.zeros(self.ndof)} )
#                    self.solver_wholebodyMPC.reset_initial_seed({f'{self.wholebodyMPC_name}/q': np.zeros((self.ndof, self.T_MPC)), f'{self.wholebodyMPC_name}/dq': np.zeros((self.ndof, self.T_MPC))} )

                # set initial seed
                if self.solution_MPC is not None:
#                    self.solver_wholebodyMPC.reset_initial_seed({f'{self.wholebodyMPC_name}/q': self.solution_MPC[f'{self.wholebodyMPC_name}/q'], f'{self.wholebodyMPC_name}/dq': self.solution_MPC[f'{self.wholebodyMPC_name}/dq'], f'p_epsilon': self.solution_MPC[f'p_epsilon'], f'v_epsilon': self.solution_MPC[f'v_epsilon']})
                    self.solver_wholebodyMPC.reset_initial_seed({f'{self.wholebodyMPC_name}/q': self.solution_MPC[f'{self.wholebodyMPC_name}/q'], f'{self.wholebodyMPC_name}/dq': self.solution_MPC[f'{self.wholebodyMPC_name}/dq'], f'p_epsilon': self.solution_MPC[f'p_epsilon']})
#                    self.solver_wholebodyMPC.reset_initial_seed({f'{self.wholebodyMPC_name}/q': self.solution_MPC[f'{self.wholebodyMPC_name}/q'], f'{self.wholebodyMPC_name}/dq': self.solution_MPC[f'{self.wholebodyMPC_name}/dq']})
#                self.solver_wholebodyMPC.reset_parameters({'goal': goal, 'dgoal': dgoal, 'init_position_MPC': self.q_curr})
                self.q_curr = np.concatenate((self.q_curr_base, self.q_curr_joint), axis=None)
                self.dq_curr = np.concatenate((self.dq_curr_base, self.dq_curr_joint), axis=None)
                self.solver_wholebodyMPC.reset_parameters({'goal': goal, 'dgoal': dgoal, 'init_position_MPC': self.q_curr, 'init_velocity_MPC': self.dq_curr})
                # solve problem
                self.solution_MPC = self.solver_wholebodyMPC.solve()
                ### ---------------------------------------------------------
                # compute next configuration with lambda function
                q_plan = self.solver_wholebodyMPC.interpolate(self.solution_MPC[f'{self.wholebodyMPC_name}/q'], self.duration_MPC)
                q_next = q_plan(1/self._freq)
                dq_plan = self.solver_wholebodyMPC.interpolate(self.solution_MPC[f'{self.wholebodyMPC_name}/dq'], self.duration_MPC)
                dq_next = dq_plan(1/self._freq)
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
        base_linear_velocity_global = self.donkey_R @ self.donkey_velocity
        base_angular_velocity_global = self.donkey_R @ self.donkey_angular_velocity
#        self.dq_curr_base = [base_linear_velocity_global[0], base_linear_velocity_global[1], base_angular_velocity_global[2]]
        self.dq_curr_base = [float(msg.twist.twist.linear.x), float(msg.twist.twist.linear.y), float(msg.twist.twist.angular.z)]


    def read_mux_selection(self, msg):
        self._correct_mux_selection = (msg.data == self._pub_cmd_topic_name)

    def preempt_cb(self):
        rospy.loginfo("%s: Preempted.", self._name)
        # set the action state to preempted
        self._action_server.set_preempted()

if __name__=="__main__":
    # Initialize node
    rospy.init_node("cmd_pose_server_MPC", anonymous=True)
    # Initialize node class
    cmd_pose_server = CmdPoseActionServer(rospy.get_name())
    # executing node
    rospy.spin()
