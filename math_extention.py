# -*- coding: utf-8 -*-

'''
ライブラリで拡張できない計算関数群です．
'''

import numpy as np

def bc2ned(phi, theta, psi, bc):
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
bc : array-like
    Body Coordinate System [x, y, z]

Returns
-------
ned : array-like
    North East Down System [x, y, z]

Raises
------
ValueError
    Raises ValueError if size of 'bc' is not 3.
'''
    return ned
