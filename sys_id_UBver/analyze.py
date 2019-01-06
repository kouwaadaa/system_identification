# -*- coding: utf-8 -*-
# author: ub
# 2018/12/14 Fri. 新座標系．

#---------------------------
# モジュールのインポートなど
#---------------------------

import numpy as np
import numpy.linalg as LA
from numpy import pi
import pandas as pd

import const
import math_extention as matex

#---------------------------
# 周波数特性を調べる
#---------------------------

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

    #---------------------------
    # 状態方程式 dx = Ax + Bu の，AとBを計算する．
    #---------------------------

    # "p_diff_f_to_x" means
    # "Partial differential of f with respect to x"

    # CL
    p_diff_CL_to_u = - (1/2)*CL_q*const.MAC*np.cos(alpha)*d_theta*(1/u**2)
    p_diff_CL_to_alpha = CL_alpha
    p_diff_CL_to_d_alpha = CL_d_alpha
    p_diff_CL_to_q = const.MAC*np.cos(alpha)*CL_q / CL_delta_e
    p_diff_CL_to_delta_e = CL_delta_e

    # L
    p_diff_L_to_u = const.RHO*const.S*CL*u/(np.cos(alpha))**2 \
                    + const.RHO*const.S*u**2*p_diff_CL_to_u/(2*(np.cos(alpha))**2) \
                    + k_L/np.cos(alpha)
    p_diff_L_to_alpha = const.RHO*const.S*u**2*CL*(np.sin(alpha)/(np.cos(alpha)**3)) \
                        + const.RHO*const.S*u**2*p_diff_CL_to_alpha/(2*np.cos(alpha)**2) \
                        + k_L*u*(np.sin(alpha)/(np.cos(alpha)**2))
    p_diff_L_to_d_alpha = const.RHO*const.S*u**2*p_diff_CL_to_d_alpha/(2*np.cos(alpha)**2)
    p_diff_L_to_q = const.RHO*const.S*u**2*p_diff_CL_to_q/(2*np.cos(alpha)**2)
    p_diff_L_to_delta_e = const.RHO*const.S*u**2*p_diff_CL_to_delta_e/(2*np.cos(alpha)**2)

    # CD
    p_diff_CD_to_u = 2*kappa*CL*p_diff_CL_to_u
    p_diff_CD_to_alpha = 2*kappa*CL*p_diff_CL_to_alpha
    p_diff_CD_to_d_alpha = 2*kappa*CL*p_diff_CL_to_d_alpha
    p_diff_CD_to_q = 2*kappa*CL*p_diff_CL_to_q
    p_diff_CD_to_delta_e = 2*kappa*CL*p_diff_CL_to_delta_e

    # D
    p_diff_D_to_u = const.RHO*const.S*CD*u/(np.cos(alpha))**2 \
                    + const.RHO*const.S*u**2*p_diff_CD_to_u/(2*(np.cos(alpha))**2) \
                    + k_D/np.cos(alpha)
    p_diff_D_to_alpha = const.RHO*const.S*u**2*CD*(np.sin(alpha)/(np.cos(alpha)**3)) \
                        + const.RHO*const.S*u**2*p_diff_CD_to_alpha/(2*np.cos(alpha)**2) \
                        + k_D*u*(np.sin(alpha)/(np.cos(alpha)**2))
    p_diff_D_to_d_alpha = const.RHO*const.S*u**2*p_diff_CD_to_d_alpha/(2*np.cos(alpha)**2)
    p_diff_D_to_q = const.RHO*const.S*u**2*p_diff_CD_to_q/(2*np.cos(alpha)**2)
    p_diff_D_to_delta_e = const.RHO*const.S*u**2*p_diff_CD_to_delta_e/(2*np.cos(alpha)**2)

    # Cm
    p_diff_Cm_to_u = - (1/2)*Cm_q*const.MAC*np.cos(alpha)*d_theta*(1/u**2)
    p_diff_Cm_to_alpha = Cm_alpha
    p_diff_Cm_to_d_alpha = Cm_d_alpha
    p_diff_Cm_to_q = const.MAC*np.cos(alpha)*Cm_q / Cm_delta_e
    p_diff_Cm_to_delta_e = Cm_delta_e

    # Ma
    p_diff_Ma_to_u = const.RHO*const.S*Cm*u/(np.cos(alpha))**2 \
                    + const.RHO*const.S*u**2*p_diff_Cm_to_u/(2*(np.cos(alpha))**2) \
                    + k_m/np.cos(alpha)
    p_diff_Ma_to_alpha = const.RHO*const.S*u**2*Cm*(np.sin(alpha)/(np.cos(alpha)**3)) \
                        + const.RHO*const.S*u**2*p_diff_Cm_to_alpha/(2*np.cos(alpha)**2) \
                        + k_m*u*(np.sin(alpha)/(np.cos(alpha)**2))
    p_diff_Ma_to_d_alpha = const.RHO*const.S*u**2*p_diff_Cm_to_d_alpha/(2*np.cos(alpha)**2)
    p_diff_Ma_to_q = const.RHO*const.S*u**2*p_diff_Cm_to_q/(2*np.cos(alpha)**2)
    p_diff_Ma_to_delta_e = const.RHO*const.S*u**2*p_diff_Cm_to_delta_e/(2*np.cos(alpha)**2)

    # Xa
    p_diff_Xa_to_u = p_diff_L_to_u*np.sin(alpha) \
                     - p_diff_D_to_u*np.cos(alpha)
    p_diff_Xa_to_alpha = (p_diff_L_to_alpha + D)*np.sin(alpha) \
                         + (L - p_diff_D_to_alpha)*np.cos(alpha)
    p_diff_Xa_to_d_alpha = p_diff_L_to_d_alpha*np.sin(alpha) \
                           - p_diff_D_to_d_alpha*np.cos(alpha)
    p_diff_Xa_to_q = p_diff_L_to_q*np.sin(alpha) \
                     - p_diff_D_to_q*np.cos(alpha)
    p_diff_Xa_to_delta_e = p_diff_L_to_delta_e*np.sin(alpha) \
                           - p_diff_D_to_delta_e*np.cos(alpha)

    # Za
    p_diff_Za_to_u = - p_diff_L_to_u*np.cos(alpha) \
                     - p_diff_D_to_u*np.sin(alpha)
    p_diff_Za_to_alpha = - (p_diff_L_to_alpha + D)*np.cos(alpha) \
                         - (p_diff_D_to_alpha - L)*np.sin(alpha)
    p_diff_Za_to_d_alpha = - p_diff_L_to_d_alpha*np.cos(alpha) \
                           - p_diff_D_to_d_alpha*np.sin(alpha)
    p_diff_Za_to_q = - p_diff_L_to_q*np.cos(alpha) \
                     - p_diff_D_to_q*np.sin(alpha)
    p_diff_Za_to_delta_e = - p_diff_L_to_delta_e*np.cos(alpha) \
                           - p_diff_D_to_delta_e*np.sin(alpha)

    # 安定微係数それぞれ
    X_u = (1/const.MASS)*p_diff_Xa_to_u
    X_alpha = (1/const.MASS)*p_diff_Xa_to_alpha
    X_q = (1/const.MASS)*p_diff_Xa_to_q - w[0]
    X_delta_e = (1/const.MASS)*p_diff_Xa_to_delta_e
    Z_u = (1/const.MASS)*p_diff_Za_to_u
    Z_alpha = (1/const.MASS)*p_diff_Za_to_alpha
    Z_d_alpha = u[0] - (1/const.MASS)*p_diff_Za_to_d_alpha
    Z_q = (1/const.MASS)*p_diff_Za_to_q - u[0]
    Z_delta_e = (1/const.MASS)*p_diff_Za_to_delta_e
    Z_u_bar = Z_u/Z_d_alpha
    Z_alpha_bar = Z_alpha/Z_d_alpha
    Z_q_bar = Z_q/Z_d_alpha
    Z_delta_e_bar = Z_delta_e/Z_d_alpha
    M_u_prime = p_diff_Ma_to_u + Z_u_bar
    M_alpha_prime = p_diff_Ma_to_alpha + Z_alpha_bar
    M_q_prime = p_diff_Ma_to_q + Z_q_bar
    M_theta_prime = const.MASS*const.GRA*(const.R_G_X*np.sin(theta[0]) - const.R_G_Z*np.cos(theta[0])) \
                    - const.GRA*np.sin(theta[0])/Z_d_alpha
    M_delta_e_prime = p_diff_Ma_to_delta_e + Z_delta_e_bar
    M_Tm_prime = - (const.LEN_M + (1/Z_d_alpha))*np.cos(tilt)
    M_Tr_prime = - (const.LEN_R_X + (1/Z_d_alpha))
    M_Tf_prime = const.LEN_F - (1/Z_d_alpha)

    # 状態方程式 dx = Ax + Bu
    A = np.zeros((4,4))
    B = np.zeros((4,4))

    # A,Bそれぞれの要素を代入
    # A
    A[0,0] = X_u
    A[0,1] = X_alpha
    A[0,2] = X_q
    A[0,3] = - const.GRA*np.cos(theta[0])

    A[1,0] = Z_u_bar
    A[1,1] = Z_alpha_bar
    A[1,2] = Z_q_bar
    A[1,3] = - const.GRA*np.sin(theta[0])/Z_d_alpha

    A[2,0] = M_u_prime
    A[2,1] = M_alpha_prime
    A[2,2] = M_q_prime
    A[2,3] = M_theta_prime

    A[3,0] = 0
    A[3,1] = 0
    A[3,2] = 1
    A[3,3] = 0

    # B
    B[0,0] = X_delta_e
    B[0,1] = np.sin(tilt)/const.MASS
    B[0,2] = 0
    B[0,3] = 0

    B[1,0] = Z_delta_e_bar
    B[1,1] = - np.cos(tilt)/(const.MASS*Z_d_alpha)
    B[1,2] = - 1/(const.MASS*Z_d_alpha)
    B[1,3] = - 1/(const.MASS*Z_d_alpha)

    B[2,0] = M_delta_e_prime
    B[2,1] = - (const.LEN_M + 1/(const.MASS*Z_d_alpha))*np.cos(tilt)
    B[2,2] = - (const.LEN_R_X + 1/(const.MASS*Z_d_alpha))
    B[2,3] = const.LEN_F - 1/(const.MASS*Z_d_alpha)

    B[3,0] = 0
    B[3,1] = 0
    B[3,2] = 0
    B[3,3] = 0

    # 固有値，固有ベクトルを計算する
    lambda_A,v_A = LA.eig(A)
    lambda_B,v_B = LA.eig(B)

    print(lambda_A)
    print(lambda_B)
