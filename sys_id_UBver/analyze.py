# -*- coding: utf-8 -*-
# author: ub

'''
周波数特性の解析に関する関数．
'''

import numpy as np
import numpy.linalg as LA
from numpy import pi
import pandas as pd

import const
import math_extention as matex


def linearlize(format_log_data):
    '''
    微小擾乱を考えて線形化された運動方程式を用いて，
    周波数特性を計算する関数．

    Parameters
    ----------
    format_log_data: pandas.DataFrame
        ピッチ角速度, 迎角, 迎角時間微分，対気速度, エレベータ舵角, 揚力, 抗力, ピッチモーメント
        のそれぞれの実験データを含むデータ群．

    Returns
    -------

    '''

    #---------------------------
    # 定数を持ってくる
    #---------------------------


    #---------------------------
    # 整理されたデータから値を取り出す
    #---------------------------

    data_size = len(format_log_data)
    theta = np.array(format_log_data['theta'])
    d_theta = np.array(format_log_data['d_theta'])
    alpha = np.array(format_log_data['alpha'])
    d_alpha = np.array(format_log_data['d_alpha'])
    u = np.array(format_log_data['u'])
    v = np.array(format_log_data['v'])
    w = np.array(format_log_data['w'])
    delta_e = np.array(format_log_data['delta_e'])
    tilt = np.array(format_log_data['tilt'])
    L = np.array(format_log_data['L'])
    D = np.array(format_log_data['D'])
    Ma = np.array(format_log_data['Ma'])
    CL_0 = np.array(format_log_data['CL_0'])
    CL_alpha = np.array(format_log_data['CL_alpha'])
    CL_d_alpha = np.array(format_log_data['CL_d_alpha'])
    CL_q = np.array(format_log_data['CL_q'])
    CL_delta_e = np.array(format_log_data['CL_delta_e'])
    k_L = np.array(format_log_data['k_L'])
    CD_0 = np.array(format_log_data['CD_0'])
    kappa = np.array(format_log_data['kappa'])
    k_D = np.array(format_log_data['k_D'])
    Cm_0 = np.array(format_log_data['Cm_0'])
    Cm_alpha = np.array(format_log_data['Cm_alpha'])
    Cm_d_alpha = np.array(format_log_data['Cm_d_alpha'])
    Cm_q = np.array(format_log_data['Cm_q'])
    Cm_delta_e = np.array(format_log_data['Cm_delta_e'])
    k_m = np.array(format_log_data['k_m'])
    CL = np.array(format_log_data['CL'])
    CD = np.array(format_log_data['CD'])
    Cm = np.array(format_log_data['Cm'])

    # sin_alpha = alpha
    # cos_alpha = 1

    sin_alpha = np.sin(alpha)
    cos_alpha = np.cos(alpha)

    #---------------------------
    # 状態方程式 dx = Ax + Bu の，AとBを計算する．
    #---------------------------

    # "p_diff_f_to_x" means
    # "Partial differential of f with respect to x"

    # CL
    p_diff_CL_to_u = - (1/2)*CL_q*const.MAC*cos_alpha*d_theta*(1/u**2)
    p_diff_CL_to_alpha = CL_alpha
    p_diff_CL_to_d_alpha = const.MAC*cos_alpha*CL_d_alpha / (2*u)
    p_diff_CL_to_q = const.MAC*cos_alpha*CL_q / (2*u)
    p_diff_CL_to_delta_e = CL_delta_e

    # L
    p_diff_L_to_u = const.RHO*const.S*CL*u/(cos_alpha)**2 \
                    + const.RHO*const.S*u**2*p_diff_CL_to_u/(2*(cos_alpha)**2) \
                    + k_L/cos_alpha
    p_diff_L_to_alpha = const.RHO*const.S*u**2*CL*(sin_alpha/(cos_alpha**3)) \
                        + const.RHO*const.S*u**2*p_diff_CL_to_alpha/(2*cos_alpha**2) \
                        + k_L*u*(sin_alpha/(cos_alpha**2))
    p_diff_L_to_d_alpha = const.RHO*const.S*u**2*p_diff_CL_to_d_alpha/(2*cos_alpha**2)
    p_diff_L_to_q = const.RHO*const.S*u**2*p_diff_CL_to_q/(2*cos_alpha**2)
    p_diff_L_to_delta_e = const.RHO*const.S*u**2*p_diff_CL_to_delta_e/(2*cos_alpha**2)

    # CD
    p_diff_CD_to_u = 2*kappa*CL*p_diff_CL_to_u
    p_diff_CD_to_alpha = 2*kappa*CL*p_diff_CL_to_alpha
    p_diff_CD_to_d_alpha = 2*kappa*CL*p_diff_CL_to_d_alpha
    p_diff_CD_to_q = 2*kappa*CL*p_diff_CL_to_q
    p_diff_CD_to_delta_e = 2*kappa*CL*p_diff_CL_to_delta_e

    # D
    p_diff_D_to_u = const.RHO*const.S*CD*u/(cos_alpha)**2 \
                    + const.RHO*const.S*u**2*p_diff_CD_to_u/(2*(cos_alpha)**2) \
                    + k_D/cos_alpha
    p_diff_D_to_alpha = const.RHO*const.S*u**2*CD*(sin_alpha/(cos_alpha**3)) \
                        + const.RHO*const.S*u**2*p_diff_CD_to_alpha/(2*cos_alpha**2) \
                        + k_D*u*(sin_alpha/(cos_alpha**2))
    p_diff_D_to_d_alpha = const.RHO*const.S*u**2*p_diff_CD_to_d_alpha/(2*cos_alpha**2)
    p_diff_D_to_q = const.RHO*const.S*u**2*p_diff_CD_to_q/(2*cos_alpha**2)
    p_diff_D_to_delta_e = const.RHO*const.S*u**2*p_diff_CD_to_delta_e/(2*cos_alpha**2)

    # Cm
    p_diff_Cm_to_u = - (1/2)*Cm_q*const.MAC*cos_alpha*d_theta*(1/u**2)
    p_diff_Cm_to_alpha = Cm_alpha
    p_diff_Cm_to_d_alpha = const.MAC*cos_alpha*CL_d_alpha / (2*u)
    p_diff_Cm_to_q = const.MAC*cos_alpha*Cm_q / (2*u)
    p_diff_Cm_to_delta_e = Cm_delta_e

    # Ma
    p_diff_Ma_to_u = const.RHO*const.S*Cm*u/(cos_alpha)**2 \
                    + const.RHO*const.S*u**2*p_diff_Cm_to_u/(2*(cos_alpha)**2) \
                    + k_m/cos_alpha
    p_diff_Ma_to_alpha = const.RHO*const.S*u**2*Cm*(sin_alpha/(cos_alpha**3)) \
                        + const.RHO*const.S*u**2*p_diff_Cm_to_alpha/(2*cos_alpha**2) \
                        + k_m*u*(sin_alpha/(cos_alpha**2))
    p_diff_Ma_to_d_alpha = const.RHO*const.S*u**2*p_diff_Cm_to_d_alpha/(2*cos_alpha**2)
    p_diff_Ma_to_q = const.RHO*const.S*u**2*p_diff_Cm_to_q/(2*cos_alpha**2)
    p_diff_Ma_to_delta_e = const.RHO*const.S*u**2*p_diff_Cm_to_delta_e/(2*cos_alpha**2)

    # Xa
    p_diff_Xa_to_u = p_diff_L_to_u*sin_alpha \
                     - p_diff_D_to_u*cos_alpha
    p_diff_Xa_to_alpha = (p_diff_L_to_alpha + D)*sin_alpha \
                         + (L - p_diff_D_to_alpha)*cos_alpha
    p_diff_Xa_to_d_alpha = p_diff_L_to_d_alpha*sin_alpha \
                           - p_diff_D_to_d_alpha*cos_alpha
    p_diff_Xa_to_q = p_diff_L_to_q*sin_alpha \
                     - p_diff_D_to_q*cos_alpha
    p_diff_Xa_to_delta_e = p_diff_L_to_delta_e*sin_alpha \
                           - p_diff_D_to_delta_e*cos_alpha

    # Za
    p_diff_Za_to_u = - p_diff_L_to_u*cos_alpha \
                     - p_diff_D_to_u*sin_alpha
    p_diff_Za_to_alpha = - (p_diff_L_to_alpha + D)*cos_alpha \
                         - (p_diff_D_to_alpha - L)*sin_alpha
    p_diff_Za_to_d_alpha = - p_diff_L_to_d_alpha*cos_alpha \
                           - p_diff_D_to_d_alpha*sin_alpha
    p_diff_Za_to_q = - p_diff_L_to_q*cos_alpha \
                     - p_diff_D_to_q*sin_alpha
    p_diff_Za_to_delta_e = - p_diff_L_to_delta_e*cos_alpha \
                           - p_diff_D_to_delta_e*sin_alpha

    # 安定微係数それぞれ
    X_u = (1/const.MASS)*p_diff_Xa_to_u
    X_alpha = (1/const.MASS)*p_diff_Xa_to_alpha
    X_q = (1/const.MASS)*p_diff_Xa_to_q - w
    X_theta = - const.GRA*np.cos(theta)
    X_delta_e = (1/const.MASS)*p_diff_Xa_to_delta_e
    X_Tm = np.sin(tilt)/const.MASS
    Z_u = (1/const.MASS)*p_diff_Za_to_u
    Z_alpha = (1/const.MASS)*p_diff_Za_to_alpha
    Z_d_alpha = u - (1/const.MASS)*p_diff_Za_to_d_alpha
    Z_q = (1/const.MASS)*p_diff_Za_to_q + u
    Z_theta = - const.GRA*np.sin(theta)
    Z_delta_e = (1/const.MASS)*p_diff_Za_to_delta_e
    Z_Tm = -(np.cos(tilt)/const.MASS)
    Z_Tr = -(1/const.MASS)
    Z_Tf = -(1/const.MASS)
    Z_u_bar = Z_u/Z_d_alpha
    Z_alpha_bar = Z_alpha/Z_d_alpha
    Z_q_bar = Z_q/Z_d_alpha
    Z_theta_bar = Z_theta/Z_d_alpha
    Z_delta_e_bar = Z_delta_e/Z_d_alpha
    Z_Tm_bar = Z_Tm/Z_d_alpha
    Z_Tr_bar = Z_Tr/Z_d_alpha
    Z_Tf_bar = Z_Tf/Z_d_alpha
    M_u = (1/const.I_YY)*p_diff_Ma_to_u
    M_alpha = (1/const.I_YY)*p_diff_Ma_to_alpha
    M_d_alpha = (1/const.I_YY)*p_diff_Ma_to_d_alpha
    M_q = (1/const.I_YY)*p_diff_Ma_to_q
    M_theta = (const.MASS*const.GRA/const.I_YY)*(const.R_G_X*np.sin(theta) - const.R_G_Z*np.cos(theta))
    M_delta_e = (1/const.I_YY)*p_diff_Ma_to_delta_e
    M_Tm = -const.LEN_M*np.cos(tilt)
    M_Tr = -const.LEN_R_X
    M_Tf = const.LEN_F
    M_u_bar = M_u + Z_u_bar*M_d_alpha
    M_alpha_bar = M_alpha + Z_alpha_bar*M_d_alpha
    M_q_bar = M_q + Z_q_bar*M_d_alpha
    M_theta_bar = M_theta + Z_theta_bar*M_d_alpha
    M_delta_e_bar = M_delta_e + Z_delta_e_bar*M_d_alpha
    M_Tm_bar = M_Tm + Z_Tm_bar*M_d_alpha
    M_Tr_bar = M_Tr + Z_Tr_bar*M_d_alpha
    M_Tf_bar = M_Tf + Z_Tf_bar*M_d_alpha

    # 状態方程式 dx = Ax + Bu
    A = np.zeros((data_size,4,4))
    B = np.zeros((data_size,4,4))

    # A,Bそれぞれの要素を代入
    # A
    A[:,0,0] = X_u
    A[:,0,1] = X_alpha
    A[:,0,2] = X_q
    A[:,0,3] = X_theta

    A[:,1,0] = Z_u_bar
    A[:,1,1] = Z_alpha_bar
    A[:,1,2] = Z_q_bar
    A[:,1,3] = Z_theta_bar

    A[:,2,0] = M_u_bar
    A[:,2,1] = M_alpha_bar
    A[:,2,2] = M_q_bar
    A[:,2,3] = M_theta_bar

    A[:,3,0] = 0
    A[:,3,1] = 0
    A[:,3,2] = 1
    A[:,3,3] = 0

    # B
    B[:,0,0] = X_delta_e
    B[:,0,1] = X_Tm
    B[:,0,2] = 0
    B[:,0,3] = 0

    B[:,1,0] = Z_delta_e_bar
    B[:,1,1] = Z_Tm_bar
    B[:,1,2] = Z_Tr_bar
    B[:,1,3] = Z_Tf_bar

    B[:,2,0] = M_delta_e_bar
    B[:,2,1] = M_Tm_bar
    B[:,2,2] = M_Tr_bar
    B[:,2,3] = M_Tf_bar

    B[:,3,0] = 0
    B[:,3,1] = 0
    B[:,3,2] = 0
    B[:,3,3] = 0

    # 固有値，固有ベクトルを計算する
    lambda_A,v_A = LA.eig(A)
    lambda_B,v_B = LA.eig(B)

    return[lambda_A,v_A,A,B,Z_d_alpha]
