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
#np.set_printoptions(suppress=True)

import rospy
import actionlib
import optas

from scipy.spatial.transform import Rotation as R

from sensor_msgs.msg import JointState
# ROS messages types for the real robot
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension
# ROS messages for command configuration action
# from pushing_msgs.msg import CmdPoseAction, CmdPoseFeedback, CmdPoseResult
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
        self._freq = rospy.get_param('~freq', 100)
        # publishing command node name
        self._pub_cmd_topic_name = rospy.get_param('~cmd_topic_name', '/command')
        # load robot_description
        param_robot_description = '~/robot_description_chonk'
        if rospy.has_param(param_robot_description):
            self._robot_description = rospy.get_param(param_robot_description)
            self._urdf = URDF.from_parameter_server(param_robot_description)
        else:
            rospy.logerr("%s: Param %s is unavailable!" % (self._name, param_robot_description))
            rospy.signal_shutdown('Incorrect parameter name.')
        self.joint_names = [jnt.name for jnt in self._urdf.joints if (jnt.type != 'fixed')]
        self.ndof_base = 3
        self.ndof_position_control = len(self.joint_names) - self.ndof_base
        ### optas
        ### ---------------------------------------------------------
        # set up donkey base motion optimization
        donkey = optas.RobotModel(
            urdf_string=self._robot_description,
            time_derivs=[0],
            param_joints=['CHEST_JOINT0', 'HEAD_JOINT0', 'HEAD_JOINT1', 'LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2', 'LARM_JOINT3', 'LARM_JOINT4', 'LARM_JOINT5', 'RARM_JOINT0', 'RARM_JOINT1', 'RARM_JOINT2', 'RARM_JOINT3', 'RARM_JOINT4', 'RARM_JOINT5'],
            name='chonk_donkey'
        )
        self.donkey_name = donkey.get_name()
        self.ndof = donkey.ndof
        # nominal robot configuration
        self.q_nom = optas.DM.zeros(self.ndof)
        self.donkey_opt_idx = donkey.optimized_joint_indexes
        self.donkey_param_idx = donkey.parameter_joint_indexes
        # set up optimization builder
        builder_donkey = optas.OptimizationBuilder(T=1, robots=[donkey])
        # get robot state and parameters variables
        q_var_donkey = builder_donkey.get_robot_states_and_parameters(self.donkey_name)
        # get end-effector pose as parameters
        pos_donkey = builder_donkey.add_parameter('pos_donkey', 3)
        ori_donkey = builder_donkey.add_parameter('ori_donkey', 4)
        # set variable boudaries
        builder_donkey.enforce_model_limits(self.donkey_name)
        # equality constraint on donkey position
        donkey_pos_fnc = donkey.get_global_link_position_function(link=self._link_donkey)
        builder_donkey.add_equality_constraint('final_pos_donkey', donkey_pos_fnc(q_var_donkey), rhs=pos_donkey)
        # rotation of the donkey
        self.donkey_fnc = donkey.get_global_link_rotation_function(link=self._link_donkey)
        # equality constraint on orientation
        donkey_ori_fnc = donkey.get_global_link_quaternion_function(link=self._link_donkey)
        builder_donkey.add_equality_constraint('final_ori_donkey', donkey_ori_fnc(q_var_donkey), rhs=ori_donkey)
        # optimization cost: close to nominal config
        builder_donkey.add_cost_term('nom_config', optas.sumsqr(q_var_donkey-self.q_nom))
        # setup solver
        self.solver_donkey = optas.CasADiSolver(optimization=builder_donkey.build()).setup('ipopt')
        ### ---------------------------------------------------------
        # set up right arm optimizataion
        right_arm = optas.RobotModel(
            urdf_string=self._robot_description,
            time_derivs=[0],
            param_joints=['base_joint_1', 'base_joint_2', 'base_joint_3', 'HEAD_JOINT0', 'HEAD_JOINT1', 'LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2', 'LARM_JOINT3', 'LARM_JOINT4', 'LARM_JOINT5'],
            name='chonk_right_arm'
        )
        self.right_arm_name = right_arm.get_name()
        self.ndof = right_arm.ndof
        # nominal robot configuration
        self.q_nom = optas.DM.zeros(self.ndof)
        self.right_arm_opt_idx = right_arm.optimized_joint_indexes
        self.right_arm_param_idx = right_arm.parameter_joint_indexes
        # set up optimization builder
        builder_right_arm = optas.OptimizationBuilder(T=1, robots=[right_arm])
        # get robot state and parameters variables
        q_var = builder_right_arm.get_robot_states_and_parameters(self.right_arm_name)
        # get end-effector pose as parameters
        pos = builder_right_arm.add_parameter('pos', 3)
        ori = builder_right_arm.add_parameter('ori', 4)
        # set variable boudaries
        builder_right_arm.enforce_model_limits(self.right_arm_name)
        # equality constraint on right arm position
        pos_fnc = right_arm.get_global_link_position_function(link=self._link_ee_right)
        builder_right_arm.add_equality_constraint('final_pos', pos_fnc(q_var), rhs=pos)
#        builder_right_arm.add_equality_constraint()
        # rotation of the right arm position
        self.R_fnc = right_arm.get_global_link_rotation_function(link=self._link_ee_right)
        # equality constraint on orientation
        ori_fnc = right_arm.get_global_link_quaternion_function(link=self._link_ee_right)
        builder_right_arm.add_equality_constraint('final_ori', ori_fnc(q_var), rhs=ori)
        # optimization cost: close to nominal config
        builder_right_arm.add_cost_term('nom_config', optas.sumsqr(q_var-self.q_nom))
        # setup solver
        self.solver_right_arm = optas.CasADiSolver(optimization=builder_right_arm.build()).setup('ipopt')
        ### ---------------------------------------------------------
        # set up head optimization
        head = optas.RobotModel(
            urdf_string=self._robot_description,
            time_derivs=[0],
            param_joints=['base_joint_1', 'base_joint_2', 'base_joint_3', 'CHEST_JOINT0', 'LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2', 'LARM_JOINT3', 'LARM_JOINT4', 'LARM_JOINT5', 'RARM_JOINT0', 'RARM_JOINT1', 'RARM_JOINT2', 'RARM_JOINT3', 'RARM_JOINT4', 'RARM_JOINT5'],
            name='chonk_head'
        )
        self.head_name = head.get_name()
        self.head_opt_idx = head.optimized_joint_indexes
        self.head_param_idx = head.parameter_joint_indexes
        # set up optimization builder
        builder_head = optas.OptimizationBuilder(T=1, robots=[head])
        # get robot state and parameters variables
        q_var = builder_head.get_robot_states_and_parameters(self.head_name)
        # get end-effector pose as parameters
        pos = builder_head.add_parameter('pos', 3)
        # get head heading
        pos_head = head.get_global_link_position_function(link=self._link_head)
        # get gaze position
        self.pos_gaze_fnc = head.get_global_link_position_function(link=self._link_gaze)
        # set variable boudaries
        builder_head.enforce_model_limits(self.head_name)
        # optimization cost: close to nominal config
        builder_head.add_cost_term('heading', optas.norm_2(pos_head(q_var)-pos))
        # setup solver
        self.solver_head = optas.CasADiSolver(optimization=builder_head.build()).setup('ipopt')
        ### ---------------------------------------------------------
        # set up left arm optimizataion
        left_arm = optas.RobotModel(
            urdf_string=self._robot_description,
            time_derivs=[0],
            param_joints=['base_joint_1', 'base_joint_2', 'base_joint_3', 'CHEST_JOINT0', 'HEAD_JOINT0', 'HEAD_JOINT1', 'RARM_JOINT0', 'RARM_JOINT1', 'RARM_JOINT2', 'RARM_JOINT3', 'RARM_JOINT4', 'RARM_JOINT5'],
            name='chonk_left_arm'
        )
        self.left_arm_name = left_arm.get_name()
        # nominal robot configuration
        self.left_arm_opt_idx = left_arm.optimized_joint_indexes
        self.left_arm_param_idx = left_arm.parameter_joint_indexes
        # set up optimization builder
        builder_left_arm = optas.OptimizationBuilder(T=1, robots=[left_arm])
        # get robot state and parameters variables
        q_var = builder_left_arm.get_robot_states_and_parameters(self.left_arm_name)
        q_opt = builder_left_arm.get_model_states(self.left_arm_name)
        # get end-effector pose as parameters
        pos = builder_left_arm.add_parameter('pos', 3)
        # set variable boudaries
        builder_left_arm.enforce_model_limits(self.left_arm_name)
        # equality constraint on position
        pos_fnc = left_arm.get_global_link_position_function(link=self._link_ee_left)
        builder_left_arm.add_equality_constraint('final_pos', pos_fnc(q_var), rhs=pos)
        # optimization cost: close to nominal config
        builder_left_arm.add_cost_term('nom_config', optas.sumsqr(q_opt-self.q_nom[self.left_arm_opt_idx]))
        # setup solver
        self.solver_left_arm = optas.CasADiSolver(optimization=builder_left_arm.build()).setup('ipopt')
        ### ---------------------------------------------------------
        # initialize variables
        self.q_curr = np.zeros(self.ndof)
        self.q_curr_joint = np.zeros(self.ndof_position_control)
        self.q_curr_base = np.zeros(self.ndof_base)
        self.joint_names = []
        # declare joint subscriber
        self._joint_sub = rospy.Subscriber(
            "/chonk/joint_states",
            JointState,
            self.read_joint_states_cb
        )
        self._joint_sub_base = rospy.Subscriber(
            "/chonk/donkey_velocity_controller/odom",
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
        Donkey_pos = np.asarray([
                     acceped_goal.poseDonkey.position.x,
                     acceped_goal.poseDonkey.position.y,
                     acceped_goal.poseDonkey.position.z,
        ])
        Donkey_ori = np.asarray([
                     acceped_goal.poseDonkey.orientation.x,
                     acceped_goal.poseDonkey.orientation.y,
                     acceped_goal.poseDonkey.orientation.z,
                     acceped_goal.poseDonkey.orientation.w
        ])
        pos_R = np.asarray([
                acceped_goal.poseR.position.x,
                acceped_goal.poseR.position.y,
                acceped_goal.poseR.position.z
        ])
        pos_L = np.asarray([
                acceped_goal.poseL.position.x,
                acceped_goal.poseL.position.y,
                acceped_goal.poseL.position.z
        ])
        ori_T = np.asarray([
                acceped_goal.poseR.orientation.x,
                acceped_goal.poseR.orientation.y,
                acceped_goal.poseR.orientation.z,
                acceped_goal.poseR.orientation.w
        ])
        # check boundaries of the position
        if (pos_R > self._pos_max).any() or (pos_R < self._pos_min).any():
            rospy.logwarn("%s: Request aborted. Goal position (%.2f, %.2f, %.2f) is outside of the workspace boundaries. Check parameters for this node." % (self._name, pos_R[0], pos_R[1], pos_R[2]))
            self._result.reached_goal = False
            self._action_server.set_aborted(self._result)
            return
        if (pos_L > self._pos_max).any() or (pos_L < self._pos_min).any():
            rospy.logwarn("%s: Lequest aborted. Goal position (%.2f, %.2f, %.2f) is outside of the workspace boundaries. Check parameters for this node." % (self._name, pos_L[0], pos_L[1], pos_L[2]))
            self._result.reached_goal = False
            self._action_server.set_aborted(self._result)
            return
        # print goal request
        rospy.loginfo("%s: Request to send Doneky to position (%.2f, %.2f, %.2f) with orientation (%.2f, %.2f, %.2f, %.2f), send right arm to position (%.2f, %.2f, %.2f) with orientation (%.2f, %.2f, %.2f, %.2f), and left arm to (%.2f, %.2f, %.2f) in %.1f seconds." % (
                self._name,
                Donkey_pos[0], Donkey_pos[1], Donkey_pos[2],
                Donkey_ori[0], Donkey_ori[1], Donkey_ori[2], Donkey_ori[3],
                pos_R[0], pos_R[1], pos_R[2],
                ori_T[0], ori_T[1], ori_T[2], ori_T[3],
                pos_L[0], pos_L[1], pos_L[2],
                acceped_goal.duration
            )
        )
        # read current robot joint positions
        self.q_curr = np.concatenate((self.q_curr_base, self.q_curr_joint), axis=None)
        qT = np.zeros(self.ndof)
        self.donkey_R = np.zeros((3, 3))
        self.donkey_position = np.zeros(3)
        ### optas
        ### ---------------------------------------------------------
        ### Donkey problem
        # set initial seed
        self.solver_donkey.reset_initial_seed({f'{self.donkey_name}/q': self.q_nom[self.donkey_opt_idx]})
        self.solver_donkey.reset_parameters({'pos_donkey': Donkey_pos, 'ori_donkey': Donkey_ori, f'{self.donkey_name}/P': self.q_nom[self.donkey_param_idx]})
        # solve problem
        solution = self.solver_donkey.solve()
        qT_array = solution[f'{self.donkey_name}/q']
        # save solution
        qT[self.donkey_opt_idx] = np.asarray(qT_array).T[0]
        qT[self.donkey_param_idx] = np.asarray(self.q_nom[self.donkey_param_idx]).T[0]
        self.q_nom = qT

        ### ---------------------------------------------------------
        ### right hand problem
        # set initial seed
        self.solver_right_arm.reset_initial_seed({f'{self.right_arm_name}/q': self.q_nom[self.right_arm_opt_idx]})
        self.solver_right_arm.reset_parameters({'pos': pos_R, 'ori': ori_T, f'{self.right_arm_name}/P': self.q_nom[self.right_arm_param_idx]})
        # solve problem
        solution = self.solver_right_arm.solve()
        qT_array = solution[f'{self.right_arm_name}/q']
        # save solution
        qT[self.right_arm_opt_idx] = np.asarray(qT_array).T[0]
#        qT[self.right_arm_param_idx] = np.asarray(self.q_nom[self.right_arm_param_idx]).T[0]
        self.q_nom = qT

        ### ---------------------------------------------------------
        ### head problem
        # set initial seed
        self.solver_head.reset_initial_seed({f'{self.head_name}/q': qT[self.head_opt_idx]})
        self.solver_head.reset_parameters({'pos': self.pos_gaze_fnc(qT), f'{self.head_name}/P': qT[self.head_param_idx]})
        # solve problem
        solution = self.solver_head.solve()
        qT_array = solution[f'{self.head_name}/q']
        # save solution
        qT[self.head_opt_idx] = np.asarray(qT_array).T[0]
        self.q_nom = qT

        ### ---------------------------------------------------------
        ### left arm problem
        # set initial seed
        self.solver_left_arm.reset_initial_seed({f'{self.left_arm_name}/q': qT[self.left_arm_opt_idx]})
        self.solver_left_arm.reset_parameters({'pos': pos_L, f'{self.left_arm_name}/P': qT[self.left_arm_param_idx]})
        # solve problem
        solution = self.solver_left_arm.solve()
        qT_array = solution[f'{self.left_arm_name}/q']
        # save solution
        qT[self.left_arm_opt_idx] = np.asarray(qT_array).T[0]
        ### ---------------------------------------------------------
        # helper variables
        q0 = self.q_curr
        T = acceped_goal.duration
        # print goal request
        rospy.loginfo("%s: Request to send robot joints to %s in %.1f seconds." % (self._name, qT, T))
        self._steps = int(T * self._freq)
        self._idx = 0
        Dq = qT - q0
        # interpolate between current and target configuration
        # polynomial obtained for zero speed (3rd order) and acceleratin (5th order)
        # at the initial and final time
        # self._q = lambda t: q0 + (3.*((t/T)**2) - 2.*((t/T)**3))*Dq # 3rd order
        self._q = lambda t: q0 + (10.*((t/T)**3) - 15.*((t/T)**4) + 6.*((t/T)**5))*Dq # 5th order
        self._dq = lambda t: (30.*((t/T)**2) - 60.*((t/T)**3) +30.*((t/T)**4))*(Dq/T)
        self._t = np.linspace(0., T, self._steps + 1)
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
                # compute next configuration with lambda function
                q_next = self._q(self._t[self._idx])
                dq_next = self._dq(self._t[self._idx])
                # compute the donkey velocity in its local frame
                Global_w_b = np.asarray([0., 0., dq_next[2]])
                Global_v_b = np.asarray([dq_next[0], dq_next[1], 0.])
                Local_w_b = self.donkey_R.T @ Global_w_b
                Local_v_b = self.donkey_R.T @ Global_v_b
#                - self.donkey_R.T @ optas.spatialmath.skew(self.donkey_position) @ Global_w_b
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
        self.joint_names = msg.name

    def read_base_states_cb(self, msg):
        base_euler_angle = tf.transformations.euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        self.q_curr_base = [msg.pose.pose.position.x, msg.pose.pose.position.y, base_euler_angle[2]]
        self.donkey_R = optas.spatialmath.rotz(base_euler_angle[2])
        self.donkey_position = np.asarray([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])

    def read_mux_selection(self, msg):
        self._correct_mux_selection = (msg.data == self._pub_cmd_topic_name)

    def preempt_cb(self):
        rospy.loginfo("%s: Preempted.", self._name)
        # set the action state to preempted
        self._action_server.set_preempted()

if __name__=="__main__":
    # Initialize node
    rospy.init_node("cmd_pose_server_4tasks", anonymous=True)
    # Initialize node class
    cmd_pose_server = CmdPoseActionServer(rospy.get_name())
    # executing node
    rospy.spin()
