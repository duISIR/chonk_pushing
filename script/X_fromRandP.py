#! /usr/bin/env python3
import rospy
import numpy as np
import optas

def X_fromRandP(R, p):
    """ calculate the spatial transformation matrix """
    X = np.zeros((6,6))

    X[0:3, 0:3] = R
    X[0:3, 3:6] = np.zeros((3,3))
    X[3:6, 0:3] = optas.spatialmath.skew(p) @ R
    X[3:6, 3:6] = R
    return X

def X_fromRandP_different(R, p):
    """ calculate the spatial transformation matrix """
    X = np.zeros((6,6))

    X[0:3, 0:3] = R
    X[0:3, 3:6] = np.zeros((3,3))
    X[3:6, 0:3] = -R @ optas.spatialmath.skew(p)
    X[3:6, 3:6] = R
    return X
