# -*- coding: utf-8 -*-
# author: ub
# 2018/12/14 Fri. 新座標系．

#---------------------------
# モジュールのインポートなど
#---------------------------

import numpy as np
import pandas as pd

import const
import math_extention as matex

#---------------------------
# システム同定に関する関数．
# 未知パラメータとして，すべて採用している．
#---------------------------

def sys_id_LS_ex_max(format_log_data):
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
    result: array-like
        CL，CD，Cm，揚力，抗力，ピッチモーメントのリスト．
    '''

    #---------------------------
    # 整理されたデータから値を取り出す
    #---------------------------

    data_size = len(format_log_data)
    d_theta = np.array(format_log_data['d_theta'])
    alpha = np.array(format_log_data['alpha'])
    d_alpha = np.array(format_log_data['d_alpha'])
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

    # n*1 揚力から計算された値のリスト
    yL = (L/((1/2)*const.RHO*(Va**2)*const.S))

    # n*6 リグレッサー（独立変数）や実験データのリスト
    xL = np.zeros((data_size,6))
    xL[:,0] = 1
    xL[:,1] = alpha
    xL[:,2] = d_alpha
    xL[:,3] = (const.MAC*d_theta)/(2*Va)
    xL[:,4] = delta_e
    xL[:,5] = 1/((1/2)*const.RHO*Va*const.S)

    # ３次ローパスフィルタをかける
    for i in range(3):
        yL_filt = matex.lp_filter(T_CONST,T_DIFF,data_size,yL)
        xL_filt = matex.lp_filter(T_CONST,T_DIFF,data_size,xL)

    # 擬似逆行列を用いた最小二乗解の計算
    # L_theta_hat = np.dot((np.linalg.pinv(xL)),yL)
    L_theta_hat = np.dot((np.linalg.pinv(xL_filt)),yL_filt)

    # 同定された未知パラメータの取り出し
    CL_0 = L_theta_hat[0]
    CL_alpha = L_theta_hat[1]
    CL_d_alpha = L_theta_hat[2]
    CL_q = L_theta_hat[3]
    CL_delta_e = L_theta_hat[4]
    k_L = L_theta_hat[5]

    # 同定結果から得られたCLを計算
    CL = CL_0 \
        + CL_alpha*alpha \
        + CL_d_alpha*d_alpha \
        + CL_q*(const.MAC/(2*Va))*d_theta \
        + CL_delta_e*delta_e

    #---------------------------
    # 抗力
    #---------------------------

    # n*1 抗力から計算された値のリスト
    yD = (D/((1/2)*const.RHO*(Va**2)*const.S))

    # n*3 リグレッサー（独立変数）や実験データのリスト
    xD = np.zeros((data_size,3))
    xD[:,0] = 1
    xD[:,1] = CL**2
    xD[:,2] = 1/((1/2)*const.RHO*Va*const.S)

    # ３次ローパスフィルタをかける
    for i in range(3):
        yD_filt = matex.lp_filter(T_CONST,T_DIFF,data_size,yD)
        xD_filt = matex.lp_filter(T_CONST,T_DIFF,data_size,xD)

    # 擬似逆行列を用いた最小二乗解の計算
    # D_theta_hat = np.dot((np.linalg.pinv(xD)),yD)
    D_theta_hat = np.dot((np.linalg.pinv(xD_filt)),yD_filt)

    # 同定された未知パラメータの取り出し
    CD_0 = D_theta_hat[0]
    kappa = D_theta_hat[1]
    k_D = D_theta_hat[2]

    # 同定結果から得られたCDを計算
    CD = CD_0 + kappa*(CL**2)

    #---------------------------
    # モーメント
    #---------------------------

    # n*1 空力モーメントから計算された値のリスト
    ym = Ma/((1/2)*const.RHO*(Va**2)*const.S*const.MAC)

    # n*5 リグレッサー（独立変数）や実験データのリスト
    xm = np.zeros((data_size,6))
    xm[:,0] = 1
    xm[:,1] = alpha
    xm[:,2] = d_alpha
    xm[:,3] = (const.MAC/(2*Va))*d_theta
    xm[:,4] = delta_e
    xm[:,5] = 1/((1/2)*const.RHO*Va*const.S*const.MAC)

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
    Cm_d_alpha = m_theta_hat[2]
    Cm_q = m_theta_hat[3]
    Cm_delta_e = m_theta_hat[4]
    k_m = m_theta_hat[5]

    # 同定結果から得られたCDを計算
    Cm = Cm_0 \
        + Cm_alpha*alpha \
        + Cm_d_alpha*d_alpha \
        + Cm_q*(const.MAC/(2*Va))*d_theta \
        + Cm_delta_e*delta_e

    #---------------------------
    # 同定結果を用いて空力を再現
    #---------------------------

    L_calc = (1/2)*const.RHO*const.S*(Va**2)*CL + k_L*Va
    D_calc = (1/2)*const.RHO*const.S*(Va**2)*CD + k_D*Va
    Ma_calc = (1/2)*const.RHO*const.S*(Va**2)*const.MAC*Cm + k_m*Va

    #---------------------------
    # 結果をリストにまとめて返す
    #---------------------------

    CL_params = np.zeros((data_size,6))
    CD_params = np.zeros((data_size,3))
    Cm_params = np.zeros((data_size,6))

    CL_params[:,0] = CL_0
    CL_params[:,1] = CL_alpha
    CL_params[:,2] = CL_d_alpha
    CL_params[:,3] = CL_q
    CL_params[:,4] = CL_delta_e
    CL_params[:,5] = k_L

    CD_params[:,0] = CD_0
    CD_params[:,1] = kappa
    CD_params[:,2] = k_D

    Cm_params[:,0] = Cm_0
    Cm_params[:,1] = Cm_alpha
    Cm_params[:,2] = Cm_d_alpha
    Cm_params[:,3] = Cm_q
    Cm_params[:,4] = Cm_delta_e
    Cm_params[:,5] = k_m

    result = np.array([CL,CD,Cm,L_calc,D_calc,Ma_calc])

    return [CL_params,CD_params,Cm_params,result.T]
