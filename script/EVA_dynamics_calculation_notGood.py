#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
To run this example, first install iDynTree for python,
then navigate to this directory and run:
python KinDynComputationsTutorial.py

Otherwise, modify the URDF_FILE parameter to point to an existing
URDF in your system.
"""
"""
Notice: By using idyntree, including base linear motion, angular motion, in this sequence.
For C and G, it includes linear force, and angular force, in this sequence. 
"""
import rospy
import idyntree.bindings as iDynTree
import numpy as np
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
import tf
import optas
import scipy


class InertiaProcessing:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('inertia_processor')
        self.rate = rospy.Rate(100)
        # declare joint subscriber
        self._joint_sub = rospy.Subscriber("/chonk/joint_states", JointState, self.read_joint_states_cb)
        self._joint_sub_base = rospy.Subscriber("/chonk/base_pose_ground_truth", Odometry, self.read_base_states_cb)
        # declare inertia publishers
        self._inertia_pub = rospy.Publisher("/chonk/arm_ee_inertias", Float64MultiArray, queue_size=10)
        # initialize the two arm ee grasp inertia message
        self._msg = Float64MultiArray()
        self._msg.layout = MultiArrayLayout()
        self._msg.layout.data_offset = 0
        self._msg.layout.dim.append(MultiArrayDimension())
        self._msg.layout.dim[0].label = "columns"
        self._msg.layout.dim[0].size = 6 + 6  # meaning: 6 for right arm ee grasp, 6 for left arm ee grasp
        # declare inertia publishers
        self._inertia_1DOF_pub = rospy.Publisher("/chonk/arm_ee_1DOF_inertias_1DOF", Float64MultiArray, queue_size=10)
        # initialize the two arm ee grasp inertia message
        self._msg_1DOF = Float64MultiArray()
        self._msg_1DOF.layout = MultiArrayLayout()
        self._msg_1DOF.layout.data_offset = 0
        self._msg_1DOF.layout.dim.append(MultiArrayDimension())
        self._msg_1DOF.layout.dim[0].label = "columns"
        self._msg_1DOF.layout.dim[0].size = 1 + 1  # meaning: 6 for right arm ee grasp, 6 for left arm ee grasp
        # define the urdf file used by idyntree, this urdf is generated from gazebo
        self.URDF_FILE = '/home/dwq/chonk/src/chonk_pushing/urdf/chonk_gazebo_fixed_wheels_combinebody.urdf'
        # load urdf model
        self.dynComp = iDynTree.KinDynComputations()
        self.mdlLoader = iDynTree.ModelLoader()
        self.mdlLoader.loadModelFromFile(self.URDF_FILE)
        self.dynComp.loadRobotModel(self.mdlLoader.model())
        self.model = self.dynComp.model()
        # set floating-base link
        self.dynComp.setFloatingBase("donkey_link")
        # set the frame reference, all expressed in inertial fixed frame
        self.dynComp.setFrameVelocityRepresentation(iDynTree.INERTIAL_FIXED_REPRESENTATION)
        # print model informations
        print("The loaded model has", self.dynComp.getNrOfDegreesOfFreedom(), "internal DOFs and", self.dynComp.getNrOfLinks(), "links.")
        print("The floating base is", self.dynComp.getFloatingBase())
        # define actual joint configuration and velocity feedbacks
        self.q_feedback_gazebo = np.zeros(23)  # because gazebo model has 4 wheel joints and 4 wheel passive joints
        self.dq_feedback_gazebo = np.zeros(23)
        # get number of DOFs and Links
        self.Dofs = self.dynComp.getNrOfDegreesOfFreedom()
        self.link_num = self.dynComp.getNrOfLinks()
        # The gravity acceleration is a 3d acceleration vector.
        self.gravity = np.asarray([0.0, 0.0, -9.8])
        # declare floating-base and joint configuration and velocity variables
        self.rotation_base = iDynTree.Rotation().Identity()
        self.p_base = np.zeros(3)
        self.world_T_base = iDynTree.Transform(self.rotation_base, self.p_base)
        self.s = iDynTree.VectorDynSize(self.Dofs)
        self.ds = iDynTree.VectorDynSize(self.Dofs)
        self.base_velocity = iDynTree.Twist(iDynTree.GeomVector3(0, 0, 0), iDynTree.GeomVector3(0, 0, 0))
        # declare jacobian variables to achieve the right arm ee grasp Jacobian
        self.Jac_JOINT5_right = iDynTree.MatrixDynSize(6, 6 + self.Dofs)
        self.Jac_ee_right = np.zeros((6, 6+self.Dofs))
        self.G_T_JOINT5_right = np.zeros((4, 4))
        # declare jacobian variables to achieve the left arm ee grasp Jacobian
        self.Jac_JOINT5_left = iDynTree.MatrixDynSize(6, 6 + self.Dofs)
        self.Jac_ee_left = np.zeros((6, 6+self.Dofs))
        self.G_T_JOINT5_left = np.zeros((4, 4))
        # declare ee and ft position relationship variables
        self.JOINT5_p_ee_right = np.asarray([-0.08, 0, -0.039-0.04-0.0333])
        self.JOINT5_p_ee_left = np.asarray([-0.08, 0, -0.039-0.04-0.0333])
        self.X_transform_right = np.identity(6)
        self.X_transform_left = np.identity(6)
        # declare operational-space inertia variables
        self.G_lambda_r = np.zeros((6, 6))
        self.G_lambda_l = np.zeros((6, 6))
        self.G_lambda_r_1DOF = 0
        self.G_lambda_l_1DOF = 0
        # declare variables of mass matrix M, centripedal and coriolis force C and gravity force G
        self.M = iDynTree.MatrixDynSize(self.Dofs + 6, self.Dofs + 6)
        self.M_inv = np.zeros((self.Dofs + 6, self.Dofs + 6))
        self.C_G = iDynTree.FreeFloatingGeneralizedTorques(self.model)
        self.G = iDynTree.FreeFloatingGeneralizedTorques(self.model)
        # set external torque in the joint space
        self.linkWrenches = iDynTree.LinkWrenches(self.link_num)
        self.linkWrenches.zero()
        self.External_torque = iDynTree.FreeFloatingGeneralizedTorques(self.model)
        ok = self.dynComp.generalizedExternalForces(self.linkWrenches, self.External_torque)

    def run(self):
        while not rospy.is_shutdown():
            # set all robot states
            self.dynComp.setRobotState(self.world_T_base, self.s, self.base_velocity, self.ds, self.gravity)
            # calculate two right arm end-effector grasp Jacobian
            ok = self.dynComp.getFrameFreeFloatingJacobian("RARM_JOINT5_Link", self.Jac_JOINT5_right)
            self.G_T_JOINT5_right = self.dynComp.getWorldTransform("RARM_JOINT5_Link")
            self.X_transform_right[0:3, 3:6] = -optas.spatialmath.skew(self.toNumpyArray(self.G_T_JOINT5_right.getRotation(), 3, 3) @ self.JOINT5_p_ee_right)
            self.Jac_ee_right = self.X_transform_right @ self.Jac_JOINT5_right.toNumPy()
            # calculate two right arm end-effector grasp Jacobian
            ok = self.dynComp.getFrameFreeFloatingJacobian("LARM_JOINT5_Link", self.Jac_JOINT5_left)
            self.G_T_JOINT5_left = self.dynComp.getWorldTransform("LARM_JOINT5_Link")
            self.X_transform_left[0:3, 3:6] = -optas.spatialmath.skew(self.toNumpyArray(self.G_T_JOINT5_left.getRotation(), 3, 3) @ self.JOINT5_p_ee_left)
            self.Jac_ee_left = self.X_transform_left @ self.Jac_JOINT5_left.toNumPy()
            # calculate the mass matrix
            self.dynComp.getFreeFloatingMassMatrix(self.M)
            # calculate the centripedal and coriolis force, C, and gravity force G. In this step, it means C+G
            ok = self.dynComp.generalizedBiasForces(self.C_G)
            # calculate the gravity force G. In this step, it means G
            ok = self.dynComp.generalizedGravityForces(self.G)
            # calculate the inverse of the mass matrix
            self.M_inv = np.linalg.inv(self.M.toNumPy())
            # calculate the operational-space inertia of right arm end-effector grasp
            self.G_lambda_r = np.linalg.inv(self.Jac_ee_right[0:3,:] @ self.M_inv @ np.transpose(self.Jac_ee_right[0:3,:]))
            self.G_lambda_r_1DOF = 1/(self.Jac_ee_right[1,:] @ self.M_inv @ np.transpose(self.Jac_ee_right[1,:]))
            # calculate the operational-space inertia of left arm end-effector grasp
            self.G_lambda_l = np.linalg.inv(self.Jac_ee_left[0:3,:] @ self.M_inv @ np.transpose(self.Jac_ee_left[0:3,:]))
            self.G_lambda_l_1DOF = 1/(self.Jac_ee_left[1,:] @ self.M_inv @ np.transpose(self.Jac_ee_left[1,:]))
#            a = np.zeros((6, self.Dofs +6))
#            a[0:3, :] = self.Jac_ee_right[0:3,:]
#            a[3:6, :] = self.Jac_ee_left[0:3,:]
#            print(np.linalg.inv(a @ self.M_inv @ np.transpose(a)))
            # update message
            self._msg.data = np.array([self.G_lambda_r[0, 0], self.G_lambda_r[0, 1], self.G_lambda_r[0, 2],
                                       self.G_lambda_r[1, 1], self.G_lambda_r[1, 2], self.G_lambda_r[2, 2],
                                       self.G_lambda_l[0, 0], self.G_lambda_l[0, 1], self.G_lambda_l[0, 2],
                                       self.G_lambda_l[1, 1], self.G_lambda_l[1, 2], self.G_lambda_l[2, 2]])
            self._msg_1DOF.data = np.array([self.G_lambda_r_1DOF, self.G_lambda_l_1DOF])
            # publish message
            self._inertia_pub.publish(self._msg)
            self._inertia_1DOF_pub.publish(self._msg_1DOF)
            # Jac_donkey = iDynTree.MatrixDynSize(6, 6 + self.Dofs)
            # ok = self.dynComp.getFrameFreeFloatingJacobian("donkey_link", Jac_donkey)
            # v= np.zeros(self.Dofs+6)
            # v[0:6] = self.base_velocity.toNumPy()
            # v[6:(6+self.Dofs)] = self.ds.toNumPy()
            # print(v)
            # print(Jac_donkey)
            # print(Jac_donkey.toNumPy() @ v)
            # Sleep for the remaining time to maintain the specified rate
            self.rate.sleep()

    def read_joint_states_cb(self, msg):
        self.q_feedback_gazebo = np.asarray(list(msg.position))
        self.dq_feedback_gazebo = np.asarray(list(msg.velocity))
        # since idyntree joint number is different with th gazebo.
        self.s.setVal(0,  self.q_feedback_gazebo[0])
        self.s.setVal(1,  self.q_feedback_gazebo[1])
        self.s.setVal(14, self.q_feedback_gazebo[2])
        self.s.setVal(2,  self.q_feedback_gazebo[3])
        self.s.setVal(9,  self.q_feedback_gazebo[4])
        self.s.setVal(10, self.q_feedback_gazebo[5])
        self.s.setVal(11, self.q_feedback_gazebo[6])
        self.s.setVal(12, self.q_feedback_gazebo[7])
        self.s.setVal(13, self.q_feedback_gazebo[8])
        self.s.setVal(3,  self.q_feedback_gazebo[9])
        self.s.setVal(4,  self.q_feedback_gazebo[10])
        self.s.setVal(5,  self.q_feedback_gazebo[11])
        self.s.setVal(6,  self.q_feedback_gazebo[12])
        self.s.setVal(7,  self.q_feedback_gazebo[13])
        self.s.setVal(8,  self.q_feedback_gazebo[14])
        # since idyntree joint number is different with th gazebo.
        self.ds.setVal(0,  self.dq_feedback_gazebo[0])
        self.ds.setVal(1,  self.dq_feedback_gazebo[1])
        self.ds.setVal(14, self.dq_feedback_gazebo[2])
        self.ds.setVal(2,  self.dq_feedback_gazebo[3])
        self.ds.setVal(9,  self.dq_feedback_gazebo[4])
        self.ds.setVal(10, self.dq_feedback_gazebo[5])
        self.ds.setVal(11, self.dq_feedback_gazebo[6])
        self.ds.setVal(12, self.dq_feedback_gazebo[7])
        self.ds.setVal(13, self.dq_feedback_gazebo[8])
        self.ds.setVal(3,  self.dq_feedback_gazebo[9])
        self.ds.setVal(4,  self.dq_feedback_gazebo[10])
        self.ds.setVal(5,  self.dq_feedback_gazebo[11])
        self.ds.setVal(6,  self.dq_feedback_gazebo[12])
        self.ds.setVal(7,  self.dq_feedback_gazebo[13])
        self.ds.setVal(8,  self.dq_feedback_gazebo[14])

    def read_base_states_cb(self, msg):
        self.p_base = np.asarray([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
        quaternion = np.array([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        self.rotation_base = scipy.spatial.transform.Rotation.from_quat(quaternion).as_matrix()
        self.world_T_base = iDynTree.Transform(self.rotation_base, self.p_base)
        # sequence: linear and angular
        self.base_velocity = iDynTree.Twist(iDynTree.GeomVector3(msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z),
                                            iDynTree.GeomVector3(msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z))

    def j_X_i(self, j_R_i, i_p_j):
        """only for idyntree X transformation, because it use linear velocity first and angular velocity second"""
        X = np.zeros((6, 6))
        X[0:3, 0:3] = j_R_i @ optas.spatialmath.skew(i_p_j)
        X[0:3, 3:6] = j_R_i
        X[3:6, 0:3] = j_R_i
        X[3:6, 3:6] = np.zeros((3, 3))
        return X

    def j_X_i(self, j_R_i, j_p_i):
        """only for idyntree X transformation, because it use linear velocity first and angular velocity second"""
        X = np.zeros((6, 6))
        X[0:3, 0:3] = optas.spatialmath.skew(j_p_i) @ j_R_i
        X[0:3, 3:6] = j_R_i
        X[3:6, 0:3] = j_R_i
        X[3:6, 3:6] = np.zeros((3, 3))
        return X

    def toNumpyArray(self, IdyntreeMatrix, row_num, col_num):
        NumpyArray = np.zeros((row_num, col_num))
        for i in range(row_num):
            for j in range(col_num):
                NumpyArray[i, j] = IdyntreeMatrix[i, j]
        return NumpyArray

if __name__ == '__main__':
    inertia_processor = InertiaProcessing()
    inertia_processor.run()
