# -*- coding: utf-8 -*-

'''
ライブラリで拡張できない計算関数群です．
'''

import numpy as np


def ned2bc(phi, theta, psi, x, y, z):
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
    ned = np.array([x, y, z])
    bc = np.dot(euler,ned)

    return bc


def both_side_diff(x):
    '''
    Calculate differences.

    Parameters
    ----------
    x : array-like

    Returns
    -------
    diff : array-like
        Result.

    Raises
    ------
    ValueError
        if x contains only 2 or less numbers, cannot calculate and raise error.

    Examples
    --------
    >>> x = np.array([1,2,4,5,8])
    >>> diff = both_side_diff(x)
    >>> print(diff)
    [3,3,4]
    '''

    x = np.array(x) # Convert ndarray style.
    size = np.size(x) # Get array size.
    diff_x = np.insert(x, [0,0], [0,0]) # Insert first two [0,0].
    diff_x = np.delete(diff_x,[size,size+1]) # Delete last two.
    diff = x - diff_x # Calculate differences.
    diff = np.delete(diff, [0,1]) # Delete first two.

    return diff


def central_diff(y, dx):
    '''
    Calculate with central differences.

    Parameters
    ----------
    y: array-like
    dx: array-like

    Returns
    -------
    dydx: array-like
        Result.
    '''

    dydx = both_side_diff(y) / both_side_diff(dx)
    dydx = np.insert(dydx,0,0) # First facotor is 0.
    dydx = np.append(dydx,dydx[-1]) # Copy the last factor.

    return dydx
