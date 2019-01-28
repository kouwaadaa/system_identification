# -*- coding: utf-8 -*-
# author: ub

'''
パラメータ推定に関する関数．
'''

import numpy as np
import pandas as pd

import const
import math_extention as matex


def sys_id_LS(format_df):
    '''
    整理されたデータをもとに，
    最小二乗法を用いてパラメータ推定を行なう.
    CL_0，CL_alpha，CD_0は固定．

    Parameters
    ----------
    format_df: pandas.DataFrame
        ピッチ角速度, 迎角, 対気速度, エレベータ舵角, 揚力, 抗力, ピッチモーメント
        のそれぞれの実験データを含むデータ群．

    Returns
    -------
    format_df: pandas.DataFrame
        元のデータに，同定結果を加えたデータ群．
    '''

    #---------------------------
    # 入力データから値を取り出す
    #---------------------------

    data_size = len(format_df)
    d_theta = np.array(format_df['d_theta'])
    alpha = np.array(format_df['alpha'])
    Va = np.array(format_df['Va'])
    delta_e = np.array(format_df['delta_e'])
    L = np.array(format_df['L'])
    D = np.array(format_df['D'])
    Ma = np.array(format_df['Ma'])

    #---------------------------
    # パラメータ推定（最小二乗法を用いる）
    #---------------------------
    T_CONST = 0.03

    #---------------------------
    # 揚力
    #---------------------------

    # 既知パラメータとして固定する値
    CL_0 = 0.0634
    CL_alpha = 2.68

    # n*1 揚力から計算された値のリスト
    yL = (L/((1/2)*const.RHO*(Va**2)*const.S)) - CL_0 - CL_alpha*alpha

    # n*3 リグレッサー（独立変数）や実験データのリスト
    xL = np.zeros((data_size,3))
    xL[:,0] = (const.MAC*d_theta)/(2*Va)
    xL[:,1] = delta_e
    xL[:,2] = 1/((1/2)*const.RHO*Va*const.S)

    # # ３次ローパスフィルタをかける
    # for i in range(3):
    #     yL_filt = matex.lp_filter(T_CONST,const.T_DIFF,data_size,yL)
    #     xL_filt = matex.lp_filter(T_CONST,const.T_DIFF,data_size,xL)

    # 擬似逆行列を用いた最小二乗解の計算
    L_theta_hat = np.dot((np.linalg.pinv(xL)),yL)

    # # ローパスフィルタを用いたときの最小二乗解
    # L_theta_hat = np.dot((np.linalg.pinv(xL_filt)),yL_filt)

    # 同定された未知パラメータの取り出し
    CL_q = L_theta_hat[0]
    CL_delta_e = L_theta_hat[1]
    k_L = L_theta_hat[2]

    # 同定結果から得られたCLを計算
    CL = CL_0 \
        + CL_alpha*alpha \
        + CL_q*(const.MAC/(2*Va))*d_theta \
        + CL_delta_e*delta_e

    #---------------------------
    # 抗力
    #---------------------------

    # 既知パラメータ
    CD_0 = 0.07887

    # n*1 抗力から計算された値のリスト
    yD = (D/((1/2)*const.RHO*(Va**2)*const.S)) - CD_0

    # n*2 リグレッサー（独立変数）や実験データのリスト
    xD = np.zeros((data_size,2))
    xD[:,0] = CL**2
    xD[:,1] = 1/((1/2)*const.RHO*Va*const.S)

    # # ３次ローパスフィルタをかける
    # for i in range(3):
    #     yD_filt = matex.lp_filter(T_CONST,const.T_DIFF,data_size,yD)
    #     xD_filt = matex.lp_filter(T_CONST,const.T_DIFF,data_size,xD)

    # 擬似逆行列を用いた最小二乗解の計算
    D_theta_hat = np.dot((np.linalg.pinv(xD)),yD)

    # # ローパスフィルタを用いたときの最小二乗解
    # D_theta_hat = np.dot((np.linalg.pinv(xD_filt)),yD_filt)

    # 同定された未知パラメータの取り出し
    kappa = D_theta_hat[0]
    k_D = D_theta_hat[1]

    # 同定結果から得られたCDを計算
    CD = CD_0 + kappa*(CL**2)

    #---------------------------
    # モーメント
    #---------------------------

    # n*1 空力モーメントから計算された値のリスト
    ym = Ma/((1/2)*const.RHO*(Va**2)*const.S*const.MAC)

    # n*5 リグレッサー（独立変数）や実験データのリスト
    xm = np.zeros((data_size,5))
    xm[:,0] = 1
    xm[:,1] = alpha
    xm[:,2] = (const.MAC/(2*Va))*d_theta
    xm[:,3] = delta_e
    xm[:,4] = 1/((1/2)*const.RHO*Va*const.S*const.MAC)

    # # ３次ローパスフィルタをかける
    # for i in range(3):
    #     ym_filt = matex.lp_filter(T_CONST,const.T_DIFF,data_size,ym)
    #     xm_filt = matex.lp_filter(T_CONST,const.T_DIFF,data_size,xm)

    # 擬似逆行列を用いた最小二乗解の計算
    m_theta_hat = np.dot((np.linalg.pinv(xm)),ym)

    # # ローパスフィルタを用いたときの最小二乗解
    # m_theta_hat = np.dot((np.linalg.pinv(xm_filt)),ym_filt)

    # 同定された未知パラメータの取り出し
    Cm_0 = m_theta_hat[0]
    Cm_alpha = m_theta_hat[1]
    Cm_q = m_theta_hat[2]
    Cm_delta_e = m_theta_hat[3]
    k_m = m_theta_hat[4]

    # 同定結果から得られたCDを計算
    Cm = Cm_0 \
        + Cm_alpha*alpha \
        + Cm_q*(const.MAC/(2*Va))*d_theta \
        + Cm_delta_e*delta_e

    #---------------------------
    # 同定結果を用いて空力を再現
    #---------------------------

    L_calc = (1/2)*const.RHO*const.S*(Va**2)*CL + k_L*Va
    D_calc = (1/2)*const.RHO*const.S*(Va**2)*CD + k_D*Va
    Ma_calc = (1/2)*const.RHO*const.S*(Va**2)*const.MAC*Cm + k_m*Va

    #---------------------------
    # 結果をデータファイルに書き込んで返す
    #---------------------------

    format_df['CL_0'] = CL_0
    format_df['CL_alpha'] = CL_alpha
    format_df['CL_q'] = CL_q
    format_df['CL_delta_e'] = CL_delta_e
    format_df['k_L'] = k_L

    format_df['CD_0'] = CD_0
    format_df['kappa'] = kappa
    format_df['k_D'] = k_D

    format_df['Cm_0'] = Cm_0
    format_df['Cm_alpha'] = Cm_alpha
    format_df['Cm_q'] = Cm_q
    format_df['Cm_delta_e'] = Cm_delta_e
    format_df['k_m'] = k_m

    format_df['CL'] = CL
    format_df['CD'] = CD
    format_df['Cm'] = Cm
    format_df['L_calc'] = L_calc
    format_df['D_calc'] = D_calc
    format_df['Ma_calc'] = Ma_calc

    return format_df


def sys_id_LS_ex(format_df):
    '''
    整理されたデータをもとに，
    最小二乗法を用いてパラメータ推定を行なう.
    CL_0，CL_alpha，CD_0は固定で，d_alphaの項を追加している．

    Parameters
    ----------
    format_df: pandas.DataFrame
        ピッチ角速度, 迎角, 対気速度, エレベータ舵角, 揚力, 抗力, ピッチモーメント
        のそれぞれの実験データを含むデータ群．

    Returns
    -------
    format_df: pandas.DataFrame
        元のデータに，同定結果を加えたデータ群．
    '''

    #---------------------------
    # 入力データから値を取り出す
    #---------------------------

    data_size = len(format_df)
    d_theta = np.array(format_df['d_theta'])
    alpha = np.array(format_df['alpha'])
    d_alpha = np.array(format_df['d_alpha'])
    Va = np.array(format_df['Va'])
    delta_e = np.array(format_df['delta_e'])
    L = np.array(format_df['L'])
    D = np.array(format_df['D'])
    Ma = np.array(format_df['Ma'])

    #---------------------------
    # パラメータ推定（最小二乗法を用いる）
    #---------------------------
    T_CONST = 0.03

    #---------------------------
    # 揚力
    #---------------------------

    # 既知パラメータとして固定する値
    CL_0 = 0.0634
    CL_alpha = 2.68

    # n*1 揚力から計算された値のリスト
    yL = (L/((1/2)*const.RHO*(Va**2)*const.S)) - CL_0 - CL_alpha*alpha

    # n*3 リグレッサー（独立変数）や実験データのリスト
    xL = np.zeros((data_size,4))
    xL[:,0] = d_alpha
    xL[:,1] = (const.MAC*d_theta)/(2*Va)
    xL[:,2] = delta_e
    xL[:,3] = 1/((1/2)*const.RHO*Va*const.S)

    # # ３次ローパスフィルタをかける
    # for i in range(3):
    #     yL_filt = matex.lp_filter(T_CONST,const.T_DIFF,data_size,yL)
    #     xL_filt = matex.lp_filter(T_CONST,const.T_DIFF,data_size,xL)

    # 擬似逆行列を用いた最小二乗解の計算
    L_theta_hat = np.dot((np.linalg.pinv(xL)),yL)

    # # ローパスフィルタを用いたときの最小二乗解
    # L_theta_hat = np.dot((np.linalg.pinv(xL_filt)),yL_filt)

    # 同定された未知パラメータの取り出し
    CL_d_alpha = L_theta_hat[0]
    CL_q = L_theta_hat[1]
    CL_delta_e = L_theta_hat[2]
    k_L = L_theta_hat[3]

    # 同定結果から得られたCLを計算
    CL = CL_0 \
        + CL_alpha*alpha \
        + CL_d_alpha*d_alpha \
        + CL_q*(const.MAC/(2*Va))*d_theta \
        + CL_delta_e*delta_e

    #---------------------------
    # 抗力
    #---------------------------

    # 既知パラメータ
    CD_0 = 0.07887

    # n*1 抗力から計算された値のリスト
    yD = (D/((1/2)*const.RHO*(Va**2)*const.S)) - CD_0

    # n*2 リグレッサー（独立変数）や実験データのリスト
    xD = np.zeros((data_size,2))
    xD[:,0] = CL**2
    xD[:,1] = 1/((1/2)*const.RHO*Va*const.S)

    # # ３次ローパスフィルタをかける
    # for i in range(3):
    #     yD_filt = matex.lp_filter(T_CONST,const.T_DIFF,data_size,yD)
    #     xD_filt = matex.lp_filter(T_CONST,const.T_DIFF,data_size,xD)

    # 擬似逆行列を用いた最小二乗解の計算
    D_theta_hat = np.dot((np.linalg.pinv(xD)),yD)

    # # ローパスフィルタを用いたときの最小二乗解
    # D_theta_hat = np.dot((np.linalg.pinv(xD_filt)),yD_filt)

    # 同定された未知パラメータの取り出し
    kappa = D_theta_hat[0]
    k_D = D_theta_hat[1]

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

    # # ３次ローパスフィルタをかける
    # for i in range(3):
    #     ym_filt = matex.lp_filter(T_CONST,const.T_DIFF,data_size,ym)
    #     xm_filt = matex.lp_filter(T_CONST,const.T_DIFF,data_size,xm)

    # 擬似逆行列を用いた最小二乗解の計算
    m_theta_hat = np.dot((np.linalg.pinv(xm)),ym)

    # # ローパスフィルタを用いたときの最小二乗解
    # m_theta_hat = np.dot((np.linalg.pinv(xm_filt)),ym_filt)

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
    # 結果をデータファイルに書き込んで返す
    #---------------------------

    format_df['CL_0'] = CL_0
    format_df['CL_alpha'] = CL_alpha
    format_df['CL_d_alpha'] = CL_d_alpha
    format_df['CL_q'] = CL_q
    format_df['CL_delta_e'] = CL_delta_e
    format_df['k_L'] = k_L

    format_df['CD_0'] = CD_0
    format_df['kappa'] = kappa
    format_df['k_D'] = k_D

    format_df['Cm_0'] = Cm_0
    format_df['Cm_alpha'] = Cm_alpha
    format_df['Cm_d_alpha'] = Cm_d_alpha
    format_df['Cm_q'] = Cm_q
    format_df['Cm_delta_e'] = Cm_delta_e
    format_df['k_m'] = k_m

    format_df['CL'] = CL
    format_df['CD'] = CD
    format_df['Cm'] = Cm
    format_df['L_calc'] = L_calc
    format_df['D_calc'] = D_calc
    format_df['Ma_calc'] = Ma_calc

    return format_df


def sys_id_LS_max(format_df):
    '''
    整理されたデータをもとに，
    最小二乗法を用いてパラメータ推定を行なう.
    すべて未知パラメータとして推定．d_alphaに関する項も追加．

    Parameters
    ----------
    format_df: pandas.DataFrame
        ピッチ角速度, 迎角, 対気速度, エレベータ舵角, 揚力, 抗力, ピッチモーメント
        のそれぞれの実験データを含むデータ群．

    Returns
    -------
    format_df: pandas.DataFrame
        元のデータに，同定結果を加えたデータ群．
    '''

    #---------------------------
    # 入力データから値を取り出す
    #---------------------------

    data_size = len(format_df)
    d_theta = np.array(format_df['d_theta'])
    alpha = np.array(format_df['alpha'])
    d_alpha = np.array(format_df['d_alpha'])
    Va = np.array(format_df['Va'])
    delta_e = np.array(format_df['delta_e'])
    L = np.array(format_df['L'])
    D = np.array(format_df['D'])
    Ma = np.array(format_df['Ma'])

    #---------------------------
    # パラメータ推定（最小二乗法を用いる）
    #---------------------------
    T_CONST = 0.03

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

    # # ３次ローパスフィルタをかける
    # for i in range(3):
    #     yL_filt = matex.lp_filter(T_CONST,const.T_DIFF,data_size,yL)
    #     xL_filt = matex.lp_filter(T_CONST,const.T_DIFF,data_size,xL)

    # 擬似逆行列を用いた最小二乗解の計算
    L_theta_hat = np.dot((np.linalg.pinv(xL)),yL)

    # # ローパスフィルタを用いたときの最小二乗解
    # L_theta_hat = np.dot((np.linalg.pinv(xL_filt)),yL_filt)

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

    # # ３次ローパスフィルタをかける
    # for i in range(3):
    #     yD_filt = matex.lp_filter(T_CONST,const.T_DIFF,data_size,yD)
    #     xD_filt = matex.lp_filter(T_CONST,const.T_DIFF,data_size,xD)

    # 擬似逆行列を用いた最小二乗解の計算
    D_theta_hat = np.dot((np.linalg.pinv(xD)),yD)

    # # ローパスフィルタを用いたときの最小二乗解
    # D_theta_hat = np.dot((np.linalg.pinv(xD_filt)),yD_filt)

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

    # n*6 リグレッサー（独立変数）や実験データのリスト
    xm = np.zeros((data_size,6))
    xm[:,0] = 1
    xm[:,1] = alpha
    xm[:,2] = d_alpha
    xm[:,3] = (const.MAC/(2*Va))*d_theta
    xm[:,4] = delta_e
    xm[:,5] = 1/((1/2)*const.RHO*Va*const.S*const.MAC)

    # # ３次ローパスフィルタをかける
    # for i in range(3):
    #     ym_filt = matex.lp_filter(T_CONST,const.T_DIFF,data_size,ym)
    #     xm_filt = matex.lp_filter(T_CONST,const.T_DIFF,data_size,xm)

    # 擬似逆行列を用いた最小二乗解の計算
    m_theta_hat = np.dot((np.linalg.pinv(xm)),ym)

    # # ローパスフィルタを用いたときの最小二乗解
    # m_theta_hat = np.dot((np.linalg.pinv(xm_filt)),ym_filt)

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
    # 結果をデータファイルに書き込んで返す
    #---------------------------

    format_df['CL_0'] = CL_0
    format_df['CL_alpha'] = CL_alpha
    format_df['CL_d_alpha'] = CL_d_alpha
    format_df['CL_q'] = CL_q
    format_df['CL_delta_e'] = CL_delta_e
    format_df['k_L'] = k_L

    format_df['CD_0'] = CD_0
    format_df['kappa'] = kappa
    format_df['k_D'] = k_D

    format_df['Cm_0'] = Cm_0
    format_df['Cm_alpha'] = Cm_alpha
    format_df['Cm_d_alpha'] = Cm_d_alpha
    format_df['Cm_q'] = Cm_q
    format_df['Cm_delta_e'] = Cm_delta_e
    format_df['k_m'] = k_m

    format_df['CL'] = CL
    format_df['CD'] = CD
    format_df['Cm'] = Cm
    format_df['L_calc'] = L_calc
    format_df['D_calc'] = D_calc
    format_df['Ma_calc'] = Ma_calc

    return format_df

def sys_id_LS_max_ub(format_df):
    '''
    整理されたデータをもとに，
    最小二乗法を用いてパラメータ推定を行なう.
    すべて未知パラメータとして推定．d_alphaに関する項も追加．
    モデル式を変更している．

    Parameters
    ----------
    format_df: pandas.DataFrame
        ピッチ角速度, 迎角, 対気速度, エレベータ舵角, 揚力, 抗力, ピッチモーメント
        のそれぞれの実験データを含むデータ群．

    Returns
    -------
    format_df: pandas.DataFrame
        元のデータに，同定結果を加えたデータ群．
    '''

    #---------------------------
    # 入力データから値を取り出す
    #---------------------------

    data_size = len(format_df)
    d_theta = np.array(format_df['d_theta'])
    alpha = np.array(format_df['alpha'])
    d_alpha = np.array(format_df['d_alpha'])
    Va = np.array(format_df['Va'])
    delta_e = np.array(format_df['delta_e'])
    L = np.array(format_df['L'])
    D = np.array(format_df['D'])
    Ma = np.array(format_df['Ma'])

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

    # 擬似逆行列を用いた最小二乗解の計算
    L_theta_hat = np.dot((np.linalg.pinv(xL)),yL)

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
        + CL_delta_e*delta_e \
        + k_L*(1/((1/2)*const.RHO*Va*const.S))

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

    # 擬似逆行列を用いた最小二乗解の計算
    D_theta_hat = np.dot((np.linalg.pinv(xD)),yD)

    # 同定された未知パラメータの取り出し
    CD_0 = D_theta_hat[0]
    kappa = D_theta_hat[1]
    k_D = D_theta_hat[2]

    # 同定結果から得られたCDを計算
    CD = CD_0 + kappa*(CL**2) + k_D*(1/((1/2)*const.RHO*Va*const.S))

    #---------------------------
    # モーメント
    #---------------------------

    # n*1 空力モーメントから計算された値のリスト
    ym = Ma/((1/2)*const.RHO*(Va**2)*const.S*const.MAC)

    # n*6 リグレッサー（独立変数）や実験データのリスト
    xm = np.zeros((data_size,6))
    xm[:,0] = 1
    xm[:,1] = alpha
    xm[:,2] = d_alpha
    xm[:,3] = (const.MAC/(2*Va))*d_theta
    xm[:,4] = delta_e
    xm[:,5] = 1/((1/2)*const.RHO*Va*const.S*const.MAC)

    # 擬似逆行列を用いた最小二乗解の計算
    m_theta_hat = np.dot((np.linalg.pinv(xm)),ym)

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
        + Cm_delta_e*delta_e \
        + k_m*(1/((1/2)*const.RHO*Va*const.S*const.MAC))

    #---------------------------
    # 同定結果を用いて空力を再現
    #---------------------------

    L_calc = (1/2)*const.RHO*const.S*(Va**2)*CL
    D_calc = (1/2)*const.RHO*const.S*(Va**2)*CD
    Ma_calc = (1/2)*const.RHO*const.S*(Va**2)*const.MAC*Cm

    #---------------------------
    # 結果をデータファイルに書き込んで返す
    #---------------------------

    format_df['CL_0'] = CL_0
    format_df['CL_alpha'] = CL_alpha
    format_df['CL_d_alpha'] = CL_d_alpha
    format_df['CL_q'] = CL_q
    format_df['CL_delta_e'] = CL_delta_e

    format_df['CD_0'] = CD_0
    format_df['kappa'] = kappa
    format_df['k_D'] = k_D

    format_df['Cm_0'] = Cm_0
    format_df['Cm_alpha'] = Cm_alpha
    format_df['Cm_d_alpha'] = Cm_d_alpha
    format_df['Cm_q'] = Cm_q
    format_df['Cm_delta_e'] = Cm_delta_e

    format_df['CL'] = CL
    format_df['CD'] = CD
    format_df['Cm'] = Cm
    format_df['L_calc'] = L_calc
    format_df['D_calc'] = D_calc
    format_df['Ma_calc'] = Ma_calc

    return format_df


def sys_id_LS_max_non_kv(format_df):
    '''
    整理されたデータをもとに，
    最小二乗法を用いてパラメータ推定を行なう.
    すべて未知パラメータとして推定．kVの項が無い空気力モデルを採用．

    Parameters
    ----------
    format_df: pandas.DataFrame
        ピッチ角速度, 迎角, 対気速度, エレベータ舵角, 揚力, 抗力, ピッチモーメント
        のそれぞれの実験データを含むデータ群．

    Returns
    -------
    format_df: pandas.DataFrame
        元のデータに，同定結果を加えたデータ群．
    '''

    #---------------------------
    # 入力データから値を取り出す
    #---------------------------

    data_size = len(format_df)
    d_theta = np.array(format_df['d_theta'])
    alpha = np.array(format_df['alpha'])
    d_alpha = np.array(format_df['d_alpha'])
    Va = np.array(format_df['Va'])
    delta_e = np.array(format_df['delta_e'])
    L = np.array(format_df['L'])
    D = np.array(format_df['D'])
    Ma = np.array(format_df['Ma'])

    #---------------------------
    # パラメータ推定（最小二乗法を用いる）
    #---------------------------
    T_CONST = 0.03

    #---------------------------
    # 揚力
    #---------------------------

    # n*1 揚力から計算された値のリスト
    yL = (L/((1/2)*const.RHO*(Va**2)*const.S))

    # n*5 リグレッサー（独立変数）や実験データのリスト
    xL = np.zeros((data_size,5))
    xL[:,0] = 1
    xL[:,1] = alpha
    xL[:,2] = d_alpha
    xL[:,3] = (const.MAC*d_theta)/(2*Va)
    xL[:,4] = delta_e

    # # ３次ローパスフィルタをかける
    # for i in range(3):
    #     yL_filt = matex.lp_filter(T_CONST,const.T_DIFF,data_size,yL)
    #     xL_filt = matex.lp_filter(T_CONST,const.T_DIFF,data_size,xL)

    # 擬似逆行列を用いた最小二乗解の計算
    L_theta_hat = np.dot((np.linalg.pinv(xL)),yL)

    # # ローパスフィルタを用いたときの最小二乗解
    # L_theta_hat = np.dot((np.linalg.pinv(xL_filt)),yL_filt)

    # 同定された未知パラメータの取り出し
    CL_0 = L_theta_hat[0]
    CL_alpha = L_theta_hat[1]
    CL_d_alpha = L_theta_hat[2]
    CL_q = L_theta_hat[3]
    CL_delta_e = L_theta_hat[4]

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

    # n*2 リグレッサー（独立変数）や実験データのリスト
    xD = np.zeros((data_size,2))
    xD[:,0] = 1
    xD[:,1] = CL**2

    # # ３次ローパスフィルタをかける
    # for i in range(3):
    #     yD_filt = matex.lp_filter(T_CONST,const.T_DIFF,data_size,yD)
    #     xD_filt = matex.lp_filter(T_CONST,const.T_DIFF,data_size,xD)

    # 擬似逆行列を用いた最小二乗解の計算
    D_theta_hat = np.dot((np.linalg.pinv(xD)),yD)

    # # ローパスフィルタを用いたときの最小二乗解
    # D_theta_hat = np.dot((np.linalg.pinv(xD_filt)),yD_filt)

    # 同定された未知パラメータの取り出し
    CD_0 = D_theta_hat[0]
    kappa = D_theta_hat[1]

    # 同定結果から得られたCDを計算
    CD = CD_0 + kappa*(CL**2)

    #---------------------------
    # モーメント
    #---------------------------

    # n*1 空力モーメントから計算された値のリスト
    ym = Ma/((1/2)*const.RHO*(Va**2)*const.S*const.MAC)

    # n*5 リグレッサー（独立変数）や実験データのリスト
    xm = np.zeros((data_size,5))
    xm[:,0] = 1
    xm[:,1] = alpha
    xm[:,2] = d_alpha
    xm[:,3] = (const.MAC/(2*Va))*d_theta
    xm[:,4] = delta_e

    # # ３次ローパスフィルタをかける
    # for i in range(3):
    #     ym_filt = matex.lp_filter(T_CONST,const.T_DIFF,data_size,ym)
    #     xm_filt = matex.lp_filter(T_CONST,const.T_DIFF,data_size,xm)

    # 擬似逆行列を用いた最小二乗解の計算
    m_theta_hat = np.dot((np.linalg.pinv(xm)),ym)

    # # ローパスフィルタを用いたときの最小二乗解
    # m_theta_hat = np.dot((np.linalg.pinv(xm_filt)),ym_filt)

    # 同定された未知パラメータの取り出し
    Cm_0 = m_theta_hat[0]
    Cm_alpha = m_theta_hat[1]
    Cm_d_alpha = m_theta_hat[2]
    Cm_q = m_theta_hat[3]
    Cm_delta_e = m_theta_hat[4]

    # 同定結果から得られたCDを計算
    Cm = Cm_0 \
        + Cm_alpha*alpha \
        + Cm_d_alpha*d_alpha \
        + Cm_q*(const.MAC/(2*Va))*d_theta \
        + Cm_delta_e*delta_e

    #---------------------------
    # 同定結果を用いて空力を再現
    #---------------------------

    L_calc = (1/2)*const.RHO*const.S*(Va**2)*CL
    D_calc = (1/2)*const.RHO*const.S*(Va**2)*CD
    Ma_calc = (1/2)*const.RHO*const.S*(Va**2)*const.MAC*Cm

    #---------------------------
    # 結果をデータファイルに書き込んで返す
    #---------------------------

    format_df['CL_0'] = CL_0
    format_df['CL_alpha'] = CL_alpha
    format_df['CL_d_alpha'] = CL_d_alpha
    format_df['CL_q'] = CL_q
    format_df['CL_delta_e'] = CL_delta_e

    format_df['CD_0'] = CD_0
    format_df['kappa'] = kappa

    format_df['Cm_0'] = Cm_0
    format_df['Cm_alpha'] = Cm_alpha
    format_df['Cm_d_alpha'] = Cm_d_alpha
    format_df['Cm_q'] = Cm_q
    format_df['Cm_delta_e'] = Cm_delta_e

    format_df['CL'] = CL
    format_df['CD'] = CD
    format_df['Cm'] = Cm
    format_df['L_calc'] = L_calc
    format_df['D_calc'] = D_calc
    format_df['Ma_calc'] = Ma_calc

    return format_df
