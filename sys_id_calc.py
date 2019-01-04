# -*- coding: utf-8 -*-
# author: ub
# 2018/12/14 Fri. 新座標系．

#---------------------------
# モジュールのインポートなど
#---------------------------

import numpy as np
from numpy import pi
import pandas as pd
import math_extention as matex

#---------------------------
# 機体データ（定数）
#  - 新座標系のもとでの値．
#---------------------------

# 慣性モーメント [kg・m^2]
I = np.array(
    [[ 0.2484,-0.0037,-0.0078],
     [-0.0037, 0.1668, 0.0005],
     [-0.0078, 0.0005, 0.3804]]
)
I_XX = I[0,0]
I_YY = I[1,1]
I_ZZ = I[2,2]

# 各距離 [m]
# to main, rear, front, pixhawk
# 重心からの距離 -> 新座標系からの距離に修正が必要
LEN_M = 0.042 # 重心〜メインロータ
LEN_F = 0.496 # 重心〜サブロータ前
LEN_R_X = 0.232 # 重心〜サブロータ横，X軸方向
LEN_R_Y = 0.503 # 重心〜サブロータ横，Y軸方向
LEN_P = 0.353 # 重心〜Pixhawk
MAC = 0.43081 # 平均空力翼弦

# 面積
S = 0.2087*2 + 0.1202 # 主翼2枚 + 機体

# 物理量
MASS = 5.7376 # Airframe weight
GRA = 9.80665 # Gravity acceleration
RHO = 1.205 # Air density

# サブロータ推力の上限
SUB_THRUST_MAX = 9.0

#---------------------------
# システム同定に関する関数．
#---------------------------

def sys_id_LS(format_log_data):
    '''
    整理されたデータをもとに，
    最小二乗法を用いてシステム同定を行なう関数.

    Parameters
    ----------
    format_log_data: pandas.DataFrame
        ピッチ角速度, 迎角, 対気速度, エレベータ舵角, 揚力, 抗力, ピッチモーメント
        のそれぞれの実験データを含むデータ群．

    Returns
    -------
    CL_params : array-like
    CD_params : array-like
    Cm_params : array-like
        CL, CD, Cmのパラメータ同定結果のリスト．
    '''

    #---------------------------
    # 整理されたデータから値を取り出す
    #---------------------------

    data_size = len(format_log_data)
    d_theta = np.array(format_log_data['d_theta'])
    alpha = np.array(format_log_data['alpha'])
    Va = np.array(format_log_data['Va'])
    delta_e = np.array(format_log_data['delta_e'])
    L = np.array(format_log_data['L'])
    D = np.array(format_log_data['D'])
    Ma = np.array(format_log_data['Ma'])

    #---------------------------
    # システム同定（最小二乗法を用いる）
    #---------------------------
    T_CONST = 0.03
    T_DIFF = 0.02 # 時間偏差

    #---------------------------
    # 揚力
    #---------------------------

    # 既知パラメータ
    CL_0 = 0.0634
    CL_alpha = 2.68

    # n*1 揚力から計算された値のリスト
    yL = (L/((1/2)*RHO*(Va**2)*S)) - CL_0 - CL_alpha*alpha

    # n*3 リグレッサー（独立変数）や実験データのリスト
    xL = np.zeros((data_size,3))
    xL[:,0] = (MAC*d_theta)/(2*Va)
    xL[:,1] = delta_e
    xL[:,2] = 1/((1/2)*RHO*Va*S)

    # ３次ローパスフィルタをかける
    for i in range(3):
        yL_filt = matex.lp_filter(T_CONST,T_DIFF,data_size,yL)
        xL_filt = matex.lp_filter(T_CONST,T_DIFF,data_size,xL)

    # 擬似逆行列を用いた最小二乗解の計算
    # L_theta_hat = np.dot((np.linalg.pinv(xL)),yL)
    L_theta_hat = np.dot((np.linalg.pinv(xL_filt)),yL_filt)

    # 同定された未知パラメータの取り出し
    CL_q = L_theta_hat[0]
    CL_delta_e = L_theta_hat[1]
    k_L = L_theta_hat[2]

    # 同定結果から得られたCLを計算
    CL = CL_0 \
        + CL_alpha*alpha \
        + CL_q*(MAC/(2*Va))*d_theta \
        + CL_delta_e*delta_e

    #---------------------------
    # 抗力
    #---------------------------

    # 既知パラメータ
    CD_0 = 0.07887

    # n*1 抗力から計算された値のリスト
    yD = (D/((1/2)*RHO*(Va**2)*S)) - CD_0

    # n*2 リグレッサー（独立変数）や実験データのリスト
    xD = np.zeros((data_size,2))
    xD[:,0] = CL**2
    xD[:,1] = 1/((1/2)*RHO*Va*S)

    # ３次ローパスフィルタをかける
    for i in range(3):
        yD_filt = matex.lp_filter(T_CONST,T_DIFF,data_size,yD)
        xD_filt = matex.lp_filter(T_CONST,T_DIFF,data_size,xD)

    # 擬似逆行列を用いた最小二乗解の計算
    # D_theta_hat = np.dot((np.linalg.pinv(xD)),yD)
    D_theta_hat = np.dot((np.linalg.pinv(xD_filt)),yD_filt)

    # 同定された未知パラメータの取り出し
    kappa = D_theta_hat[0]
    k_D = D_theta_hat[1]

    # 同定結果から得られたCDを計算
    CD = CD_0 + kappa*(CL**2)

    #---------------------------
    # モーメント
    #---------------------------

    # n*1 空力モーメントから計算された値のリスト
    ym = Ma/((1/2)*RHO*(Va**2)*S*MAC)

    # n*5 リグレッサー（独立変数）や実験データのリスト
    xm = np.zeros((data_size,5))
    xm[:,0] = 1
    xm[:,1] = alpha
    xm[:,2] = (MAC/(2*Va))*d_theta
    xm[:,3] = delta_e
    xm[:,4] = 1/((1/2)*RHO*Va*S*MAC)

    # ３次ローパスフィルタをかける
    for i in range(3):
        ym_filt = matex.lp_filter(T_CONST,T_DIFF,data_size,ym)
        xm_filt = matex.lp_filter(T_CONST,T_DIFF,data_size,xm)

    # 擬似逆行列を用いた最小二乗解の計算
    # m_theta_hat = np.dot((np.linalg.pinv(xm)),ym)
    m_theta_hat = np.dot((np.linalg.pinv(xm_filt)),ym_filt)

    # 同定された未知パラメータの取り出し
    Cm_0 = m_theta_hat[0]
    Cm_alpha = m_theta_hat[1]
    Cm_q = m_theta_hat[2]
    Cm_delta_e = m_theta_hat[3]
    k_m = m_theta_hat[4]

    # 同定結果から得られたCDを計算
    Cm = Cm_0 \
        + Cm_alpha*alpha \
        + Cm_q*(MAC/(2*Va))*d_theta \
        + Cm_delta_e*delta_e

    #---------------------------
    # 同定結果を用いて空力を再現
    #---------------------------

    L_calc = (1/2)*RHO*S*(Va**2)*CL + k_L*Va
    D_calc = (1/2)*RHO*S*(Va**2)*CD + k_D*Va
    Ma_calc = (1/2)*RHO*S*(Va**2)*MAC*Cm + k_m*Va

    #---------------------------
    # 結果をリストにまとめて返す
    #---------------------------

    CL_params = np.array([CL_0,CL_alpha,CL_q,CL_delta_e,k_L])
    CD_params = np.array([CD_0,kappa,k_D])
    Cm_params = np.array([Cm_0,Cm_alpha,Cm_q,Cm_delta_e,k_m])
    result = np.array([CL,CD,Cm,L_calc,D_calc,Ma_calc])

    return(CL_params,CD_params,Cm_params,result)
