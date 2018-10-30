# -*- coding: utf-8 -*-

'''
ライブラリで拡張できない計算関数群です．
'''

import numpy as np


def bc2ned(phi, theta, psi, x, y, z):
    '''
    Convert Body Coordinate System to North East Down System.

    Parameters
    ----------
    phi : float64
        Roll angle.
    theta : float64
        Pitch angle.
    psi : float64
        Yaw angle.
    x : float64
        Body Coordinate System x.
    y : float64
        Body Coordinate System y.
    z : float64
        Body Coordinate System z.

    Returns
    -------
    ned : array-like
        North East Down System [x, y, z]

    Raises
    ------
    ValueError
        Raises ValueError if size of 'bc' is not 3.
    '''

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)

    euler = np.array(
        [[cos_theta*cos_psi, cos_theta*sin_psi, -sin_theta],
        [sin_phi*sin_theta*cos_psi-cos_phi*sin_psi, sin_phi*sin_theta*sin_psi+cos_phi*cos_psi, sin_phi*cos_theta],
        [cos_phi*sin_theta*cos_psi+sin_phi*sin_psi, cos_phi*sin_theta*sin_psi-sin_phi*cos_psi, cos_phi*cos_theta]]
    )
    bc = np.array([x, y, z])
    ned = np.dot(euler,bc)

    return ned
