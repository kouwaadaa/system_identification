# -*- coding: utf-8 -*-
'''
author: ub
パラメータ推定に関する関数．
LS: 最小二乗法
'''

import numpy as np
import pandas as pd

import const
import math_extention as matex


def LS_yoshimura(format_df):
    '''
    2018吉村さんの設定．
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
    RHO = np.array(format_df['RHO'])

    #---------------------------
    # 揚力
    #---------------------------

    # 既知パラメータとして固定する値
    CL_0 = 0.0634
    CL_alpha = 2.68

    # n*1 揚力から計算された値のリスト
    yL = (L/((1/2)*RHO*(Va**2)*const.S)) - CL_0 - CL_alpha*alpha

    # n*3 リグレッサー（独立変数）や実験データのリスト
    xL = np.zeros((data_size,3))
    xL[:,0] = (const.MAC*d_theta)/(2*Va)
    xL[:,1] = delta_e
    xL[:,2] = 1/((1/2)*RHO*Va*const.S)

    # 擬似逆行列を用いた最小二乗解の計算
    L_theta_hat = np.dot((np.linalg.pinv(xL)),yL)

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
    yD = (D/((1/2)*RHO*(Va**2)*const.S)) - CD_0

    # n*2 リグレッサー（独立変数）や実験データのリスト
    xD = np.zeros((data_size,2))
    xD[:,0] = CL**2
    xD[:,1] = 1/((1/2)*RHO*Va*const.S)

    # 擬似逆行列を用いた最小二乗解の計算
    D_theta_hat = np.dot((np.linalg.pinv(xD)),yD)

    # 同定された未知パラメータの取り出し
    kappa = D_theta_hat[0]
    k_D = D_theta_hat[1]

    # 同定結果から得られたCDを計算
    CD = CD_0 + kappa*(CL**2)

    #---------------------------
    # モーメント
    #---------------------------

    # n*1 空力モーメントから計算された値のリスト
    ym = Ma/((1/2)*RHO*(Va**2)*const.S*const.MAC)

    # n*5 リグレッサー（独立変数）や実験データのリスト
    xm = np.zeros((data_size,5))
    xm[:,0] = 1
    xm[:,1] = alpha
    xm[:,2] = (const.MAC/(2*Va))*d_theta
    xm[:,3] = delta_e
    xm[:,4] = 1/((1/2)*RHO*Va*const.S*const.MAC)

    # 擬似逆行列を用いた最小二乗解の計算
    m_theta_hat = np.dot((np.linalg.pinv(xm)),ym)

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

    L_calc = (1/2)*RHO*const.S*(Va**2)*CL + k_L*Va
    D_calc = (1/2)*RHO*const.S*(Va**2)*CD + k_D*Va
    Ma_calc = (1/2)*RHO*const.S*(Va**2)*const.MAC*Cm + k_m*Va

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


def LS_with_dalpha(format_df):
    '''
    最小二乗法を用いてパラメータ推定を行なう.
    すべて未知パラメータとして推定．d_alphaに関する項も追加．

    Parameters
    ----------
    format_df: pandas.DataFrame
        ピッチ角速度, 迎角, 対気速度, エレベータ舵角, 揚力, 抗力, ピッチモーメント
        のそれぞれの実験データを含むデータ群．

    Returns
    -------
    format_df_return: pandas.DataFrame
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
    RHO = np.array(format_df['RHO'])

    #---------------------------
    # 揚力
    #---------------------------

    # n*1 揚力から計算された値のリスト
    yL = L/((1/2)*RHO*(Va**2)*const.S)

    # n*6 リグレッサー（独立変数）や実験データのリスト
    xL = np.zeros((data_size,6))
    xL[:,0] = 1
    xL[:,1] = alpha
    xL[:,2] = (const.MAC/(2*Va))*d_alpha
    xL[:,3] = (const.MAC/(2*Va))*d_theta
    xL[:,4] = delta_e
    xL[:,5] = 1/Va

    # 擬似逆行列を用いた最小二乗解の計算
    L_theta_hat = np.dot((np.linalg.pinv(xL)),yL)

    # 同定された未知パラメータの取り出し
    CL_0 = L_theta_hat[0]
    CL_alpha = L_theta_hat[1]
    CL_d_alpha = L_theta_hat[2]
    CL_q = L_theta_hat[3]
    CL_delta_e = L_theta_hat[4]
    CL_k = L_theta_hat[5]

    # 同定結果から得られたCLを計算
    CL = CL_0 \
        + CL_alpha*alpha \
        + CL_d_alpha*(const.MAC/(2*Va))*d_alpha \
        + CL_q*(const.MAC/(2*Va))*d_theta \
        + CL_delta_e*delta_e \
        + CL_k*(1/Va)

    #---------------------------
    # 抗力
    #---------------------------

    # n*1 抗力から計算された値のリスト
    yD = D/((1/2)*RHO*(Va**2)*const.S)

    # n*3 リグレッサー（独立変数）や実験データのリスト
    xD = np.zeros((data_size,3))
    xD[:,0] = 1
    xD[:,1] = CL**2
    xD[:,2] = 1/Va

    # 擬似逆行列を用いた最小二乗解の計算
    D_theta_hat = np.dot((np.linalg.pinv(xD)),yD)

    # 同定された未知パラメータの取り出し
    CD_0 = D_theta_hat[0]
    kappa = D_theta_hat[1]
    CD_k = D_theta_hat[2]

    # 同定結果から得られたCDを計算
    CD = CD_0 + kappa*(CL**2) + CD_k*(1/Va)

    #---------------------------
    # モーメント
    #---------------------------

    # n*1 空力モーメントから計算された値のリスト
    ym = Ma/((1/2)*RHO*(Va**2)*const.S*const.MAC)

    # n*6 リグレッサー（独立変数）や実験データのリスト
    xm = np.zeros((data_size,6))
    xm[:,0] = 1
    xm[:,1] = alpha
    xm[:,2] = (const.MAC/(2*Va))*d_alpha
    xm[:,3] = (const.MAC/(2*Va))*d_theta
    xm[:,4] = delta_e
    xm[:,5] = 1/Va

    # 擬似逆行列を用いた最小二乗解の計算
    m_theta_hat = np.dot((np.linalg.pinv(xm)),ym)

    # 同定された未知パラメータの取り出し
    Cm_0 = m_theta_hat[0]
    Cm_alpha = m_theta_hat[1]
    Cm_d_alpha = m_theta_hat[2]
    Cm_q = m_theta_hat[3]
    Cm_delta_e = m_theta_hat[4]
    Cm_k = m_theta_hat[5]

    # 同定結果から得られたCDを計算
    Cm = Cm_0 \
        + Cm_alpha*alpha \
        + Cm_d_alpha*(const.MAC/(2*Va))*d_alpha \
        + Cm_q*(const.MAC/(2*Va))*d_theta \
        + Cm_delta_e*delta_e \
        + Cm_k*(1/Va)

    #---------------------------
    # 同定結果を用いて空力を再現
    #---------------------------

    L_calc = (1/2)*RHO*const.S*(Va**2)*CL
    D_calc = (1/2)*RHO*const.S*(Va**2)*CD
    Ma_calc = (1/2)*RHO*const.S*(Va**2)*const.MAC*Cm

    #---------------------------
    # k_*を計算
    #---------------------------

    k_L = (1/2)*RHO*const.S*CL_k
    k_D = (1/2)*RHO*const.S*CD_k
    k_m = (1/2)*RHO*const.S*const.MAC*Cm_k

    #---------------------------
    # 結果をデータファイルに書き込んで返す
    #---------------------------

    format_df_return = format_df.copy()

    format_df_return['CL_0'] = CL_0
    format_df_return['CL_alpha'] = CL_alpha
    format_df_return['CL_d_alpha'] = CL_d_alpha
    format_df_return['CL_q'] = CL_q
    format_df_return['CL_delta_e'] = CL_delta_e
    format_df_return['CL_k'] = CL_k
    format_df_return['k_L'] = k_L

    format_df_return['CD_0'] = CD_0
    format_df_return['kappa'] = kappa
    format_df_return['CD_k'] = CD_k
    format_df_return['k_D'] = k_D

    format_df_return['Cm_0'] = Cm_0
    format_df_return['Cm_alpha'] = Cm_alpha
    format_df_return['Cm_d_alpha'] = Cm_d_alpha
    format_df_return['Cm_q'] = Cm_q
    format_df_return['Cm_delta_e'] = Cm_delta_e
    format_df_return['Cm_k'] = Cm_k
    format_df_return['k_m'] = k_m

    format_df_return['CL'] = CL
    format_df_return['CD'] = CD
    format_df_return['Cm'] = Cm
    format_df_return['L_calc'] = L_calc
    format_df_return['D_calc'] = D_calc
    format_df_return['Ma_calc'] = Ma_calc

    return format_df_return


def LS_non_dalpha(format_df):
    '''
    最小二乗法を用いてパラメータ推定を行なう.
    すべて未知パラメータとして推定．d_alphaは省略．

    Parameters
    ----------
    format_df: pandas.DataFrame
        ピッチ角速度, 迎角, 対気速度, エレベータ舵角, 揚力, 抗力, ピッチモーメント
        のそれぞれの実験データを含むデータ群．

    Returns
    -------
    format_df_return: pandas.DataFrame
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
    RHO = np.array(format_df['RHO'])

    #---------------------------
    # 揚力
    #---------------------------

    # n*1 揚力から計算された値のリスト
    yL = L/((1/2)*RHO*(Va**2)*const.S)

    # n*5 リグレッサー（独立変数）や実験データのリスト
    xL = np.zeros((data_size,5))
    xL[:,0] = 1
    xL[:,1] = alpha
    xL[:,2] = (const.MAC*d_theta)/(2*Va)
    xL[:,3] = delta_e
    # xL[:,4] = 1/Va

    # 擬似逆行列を用いた最小二乗解の計算
    L_theta_hat = np.dot((np.linalg.pinv(xL)),yL)

    # 同定された未知パラメータの取り出し
    CL_0 = L_theta_hat[0]
    CL_alpha = L_theta_hat[1]
    CL_q = L_theta_hat[2]
    CL_delta_e = L_theta_hat[3]
    # CL_k = L_theta_hat[4]

    # 同定結果から得られたCLを計算
    CL = CL_0 \
        + CL_alpha*alpha \
        + CL_q*(const.MAC/(2*Va))*d_theta \
        + CL_delta_e*delta_e \
        # + CL_k *(1/Va)

    #---------------------------
    # 抗力
    #---------------------------

    # n*1 抗力から計算された値のリスト
    yD = D/((1/2)*RHO*(Va**2)*const.S)

    # n*3 リグレッサー（独立変数）や実験データのリスト
    xD = np.zeros((data_size,3))
    xD[:,0] = 1
    xD[:,1] = CL**2
    xD[:,2] = 1/Va

    # 擬似逆行列を用いた最小二乗解の計算
    D_theta_hat = np.dot((np.linalg.pinv(xD)),yD)

    # 同定された未知パラメータの取り出し
    CD_0 = D_theta_hat[0]
    kappa = D_theta_hat[1]
    CD_k = D_theta_hat[2]

    # 同定結果から得られたCDを計算
    CD = CD_0 + kappa*(CL**2) + CD_k*(1/Va)

    #---------------------------
    # モーメント
    #---------------------------

    # n*1 空力モーメントから計算された値のリスト
    ym = Ma/((1/2)*RHO*(Va**2)*const.S*const.MAC)

    # n*5 リグレッサー（独立変数）や実験データのリスト
    xm = np.zeros((data_size,5))
    xm[:,0] = 1
    xm[:,1] = alpha
    xm[:,2] = (const.MAC/(2*Va))*d_theta
    xm[:,3] = delta_e
    xm[:,4] = 1/Va

    # 擬似逆行列を用いた最小二乗解の計算
    m_theta_hat = np.dot((np.linalg.pinv(xm)),ym)

    # 同定された未知パラメータの取り出し
    Cm_0 = m_theta_hat[0]
    Cm_alpha = m_theta_hat[1]
    Cm_q = m_theta_hat[2]
    Cm_delta_e = m_theta_hat[3]
    Cm_k = m_theta_hat[4]

    # 同定結果から得られたCDを計算
    Cm = Cm_0 \
        + Cm_alpha*alpha \
        + Cm_q*(const.MAC/(2*Va))*d_theta \
        + Cm_delta_e*delta_e \
        + Cm_k*(1/Va)

    #---------------------------
    # 同定結果を用いて空力を再現
    #---------------------------

    L_calc = (1/2)*RHO*const.S*(Va**2)*CL
    D_calc = (1/2)*RHO*const.S*(Va**2)*CD
    Ma_calc = (1/2)*RHO*const.S*(Va**2)*const.MAC*Cm

    #---------------------------
    # k_*を計算
    #---------------------------

    # k_L = (1/2)*RHO*const.S*CL_k
    k_D = (1/2)*RHO*const.S*CD_k
    k_m = (1/2)*RHO*const.S*const.MAC*Cm_k

    #---------------------------
    # 結果をデータファイルに書き込んで返す
    #---------------------------

    format_df_return = format_df.copy()

    format_df_return['CL_0'] = CL_0
    format_df_return['CL_alpha'] = CL_alpha
    format_df_return['CL_q'] = CL_q
    format_df_return['CL_delta_e'] = CL_delta_e
    # format_df_return['CL_k'] = CL_k
    # format_df_return['k_L'] = k_L

    format_df_return['CD_0'] = CD_0
    format_df_return['kappa'] = kappa
    format_df_return['CD_k'] = CD_k
    format_df_return['k_D'] = k_D

    format_df_return['Cm_0'] = Cm_0
    format_df_return['Cm_alpha'] = Cm_alpha
    format_df_return['Cm_q'] = Cm_q
    format_df_return['Cm_delta_e'] = Cm_delta_e
    format_df_return['Cm_k'] = Cm_k
    format_df_return['k_m'] = k_m

    format_df_return['CL'] = CL
    format_df_return['CD'] = CD
    format_df_return['Cm'] = Cm
    format_df_return['L_calc'] = L_calc
    format_df_return['D_calc'] = D_calc
    format_df_return['Ma_calc'] = Ma_calc

    return format_df_return


def LS_non_kv(format_df):
    '''
    最小二乗法を用いてパラメータ推定を行なう.
    すべて未知パラメータとして推定．kVの項が無い空気力モデルを採用．

    Parameters
    ----------
    format_df: pandas.DataFrame
        ピッチ角速度, 迎角, 対気速度, エレベータ舵角, 揚力, 抗力, ピッチモーメント
        のそれぞれの実験データを含むデータ群．

    Returns
    -------
    format_df_return: pandas.DataFrame
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
    RHO = np.array(format_df['RHO'])

    #---------------------------
    # 揚力
    #---------------------------

    # n*1 揚力から計算された値のリスト
    yL = L/((1/2)*RHO*(Va**2)*const.S)

    # n*5 リグレッサー（独立変数）や実験データのリスト
    xL = np.zeros((data_size,5))
    xL[:,0] = 1
    xL[:,1] = alpha
    xL[:,2] = (const.MAC/(2*Va))*d_alpha
    xL[:,3] = (const.MAC/(2*Va))*d_theta
    xL[:,4] = delta_e

    # 擬似逆行列を用いた最小二乗解の計算
    L_theta_hat = np.dot((np.linalg.pinv(xL)),yL)

    # 同定された未知パラメータの取り出し
    CL_0 = L_theta_hat[0]
    CL_alpha = L_theta_hat[1]
    CL_d_alpha = L_theta_hat[2]
    CL_q = L_theta_hat[3]
    CL_delta_e = L_theta_hat[4]

    # 同定結果から得られたCLを計算
    CL = CL_0 \
        + CL_alpha*alpha \
        + CL_d_alpha*(const.MAC/(2*Va))*d_alpha \
        + CL_q*(const.MAC/(2*Va))*d_theta \
        + CL_delta_e*delta_e

    #---------------------------
    # 抗力
    #---------------------------

    # n*1 抗力から計算された値のリスト
    yD = D/((1/2)*RHO*(Va**2)*const.S)

    # n*2 リグレッサー（独立変数）や実験データのリスト
    xD = np.zeros((data_size,2))
    xD[:,0] = 1
    xD[:,1] = CL**2

    # 擬似逆行列を用いた最小二乗解の計算
    D_theta_hat = np.dot((np.linalg.pinv(xD)),yD)

    # 同定された未知パラメータの取り出し
    CD_0 = D_theta_hat[0]
    kappa = D_theta_hat[1]

    # 同定結果から得られたCDを計算
    CD = CD_0 + kappa*(CL**2)

    #---------------------------
    # モーメント
    #---------------------------

    # n*1 空力モーメントから計算された値のリスト
    ym = Ma/((1/2)*RHO*(Va**2)*const.S*const.MAC)

    # n*5 リグレッサー（独立変数）や実験データのリスト
    xm = np.zeros((data_size,5))
    xm[:,0] = 1
    xm[:,1] = alpha
    xm[:,2] = (const.MAC/(2*Va))*d_alpha
    xm[:,3] = (const.MAC/(2*Va))*d_theta
    xm[:,4] = delta_e

    # 擬似逆行列を用いた最小二乗解の計算
    m_theta_hat = np.dot((np.linalg.pinv(xm)),ym)

    # 同定された未知パラメータの取り出し
    Cm_0 = m_theta_hat[0]
    Cm_alpha = m_theta_hat[1]
    Cm_d_alpha = m_theta_hat[2]
    Cm_q = m_theta_hat[3]
    Cm_delta_e = m_theta_hat[4]

    # 同定結果から得られたCDを計算
    Cm = Cm_0 \
        + Cm_alpha*alpha \
        + Cm_d_alpha*(const.MAC/(2*Va))*d_alpha \
        + Cm_q*(const.MAC/(2*Va))*d_theta \
        + Cm_delta_e*delta_e

    #---------------------------
    # 同定結果を用いて空力を再現
    #---------------------------

    L_calc = (1/2)*RHO*const.S*(Va**2)*CL
    D_calc = (1/2)*RHO*const.S*(Va**2)*CD
    Ma_calc = (1/2)*RHO*const.S*(Va**2)*const.MAC*Cm

    #---------------------------
    # 結果をデータファイルに書き込んで返す
    #---------------------------

    format_df_return = format_df.copy()

    format_df_return['CL_0'] = CL_0
    format_df_return['CL_alpha'] = CL_alpha
    format_df_return['CL_d_alpha'] = CL_d_alpha
    format_df_return['CL_q'] = CL_q
    format_df_return['CL_delta_e'] = CL_delta_e

    format_df_return['CD_0'] = CD_0
    format_df_return['kappa'] = kappa

    format_df_return['Cm_0'] = Cm_0
    format_df_return['Cm_alpha'] = Cm_alpha
    format_df_return['Cm_d_alpha'] = Cm_d_alpha
    format_df_return['Cm_q'] = Cm_q
    format_df_return['Cm_delta_e'] = Cm_delta_e

    format_df_return['CL'] = CL
    format_df_return['CD'] = CD
    format_df_return['Cm'] = Cm
    format_df_return['L_calc'] = L_calc
    format_df_return['D_calc'] = D_calc
    format_df_return['Ma_calc'] = Ma_calc

    return format_df_return


def LS_ex_with_dalpha(format_df):
    '''
    整理されたデータをもとに，
    最小二乗法を用いてパラメータ推定を行なう.
    すべて未知パラメータとして推定．d_alphaに関する項も追加．
    CDのモデル式を変更している．

    Parameters
    ----------
    format_df: pandas.DataFrame
        ピッチ角速度, 迎角, 対気速度, エレベータ舵角, 揚力, 抗力, ピッチモーメント
        のそれぞれの実験データを含むデータ群．

    Returns
    -------
    format_df_return: pandas.DataFrame
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
    RHO = np.array(format_df['RHO'])

    #---------------------------
    # 揚力
    #---------------------------

    # n*1 揚力から計算された値のリスト
    yL = L/((1/2)*RHO*(Va**2)*const.S)

    # n*6 リグレッサー（独立変数）や実験データのリスト
    xL = np.zeros((data_size,6))
    xL[:,0] = 1
    xL[:,1] = alpha
    xL[:,2] = (const.MAC/(2*Va))*d_alpha
    xL[:,3] = (const.MAC/(2*Va))*d_theta
    xL[:,4] = delta_e
    xL[:,5] = 1/Va

    # 擬似逆行列を用いた最小二乗解の計算
    L_theta_hat = np.dot((np.linalg.pinv(xL)),yL)

    # 同定された未知パラメータの取り出し
    CL_0 = L_theta_hat[0]
    CL_alpha = L_theta_hat[1]
    CL_d_alpha = L_theta_hat[2]
    CL_q = L_theta_hat[3]
    CL_delta_e = L_theta_hat[4]
    CL_k = L_theta_hat[5]

    # 同定結果から得られたCLを計算
    CL = CL_0 \
        + CL_alpha*alpha \
        + CL_d_alpha*(const.MAC/(2*Va))*d_alpha \
        + CL_q*(const.MAC/(2*Va))*d_theta \
        + CL_delta_e*delta_e \
        + CL_k*(1/Va)

    #---------------------------
    # 抗力
    #---------------------------

    # n*1 抗力から計算された値のリスト
    yD = D/((1/2)*RHO*(Va**2)*const.S)

    # n*6 リグレッサー（独立変数）や実験データのリスト
    xD = np.zeros((data_size,6))
    xD[:,0] = 1
    xD[:,1] = alpha
    xD[:,2] = (const.MAC/(2*Va))*d_alpha
    xD[:,3] = (const.MAC/(2*Va))*d_theta
    xD[:,4] = delta_e
    xD[:,5] = 1/Va

    # 擬似逆行列を用いた最小二乗解の計算
    D_theta_hat = np.dot((np.linalg.pinv(xD)),yD)

    # 同定された未知パラメータの取り出し
    CD_0 = D_theta_hat[0]
    CD_alpha = D_theta_hat[1]
    CD_d_alpha = D_theta_hat[2]
    CD_q = D_theta_hat[3]
    CD_delta_e = D_theta_hat[4]
    CD_k = D_theta_hat[5]

    # 同定結果から得られたCDを計算
    CD = CD_0 \
        + CD_alpha*alpha \
        + CD_d_alpha*(const.MAC/(2*Va))*d_alpha \
        + CD_q*(const.MAC/(2*Va))*d_theta \
        + CD_delta_e*delta_e \
        + CD_k*(1/Va)

    #---------------------------
    # モーメント
    #---------------------------

    # n*1 空力モーメントから計算された値のリスト
    ym = Ma/((1/2)*RHO*(Va**2)*const.S*const.MAC)

    # n*6 リグレッサー（独立変数）や実験データのリスト
    xm = np.zeros((data_size,6))
    xm[:,0] = 1
    xm[:,1] = alpha
    xm[:,2] = (const.MAC/(2*Va))*d_alpha
    xm[:,3] = (const.MAC/(2*Va))*d_theta
    xm[:,4] = delta_e
    xm[:,5] = 1/Va

    # 擬似逆行列を用いた最小二乗解の計算
    m_theta_hat = np.dot((np.linalg.pinv(xm)),ym)

    # 同定された未知パラメータの取り出し
    Cm_0 = m_theta_hat[0]
    Cm_alpha = m_theta_hat[1]
    Cm_d_alpha = m_theta_hat[2]
    Cm_q = m_theta_hat[3]
    Cm_delta_e = m_theta_hat[4]
    Cm_k = m_theta_hat[5]

    # 同定結果から得られたCDを計算
    Cm = Cm_0 \
        + Cm_alpha*alpha \
        + Cm_d_alpha*(const.MAC/(2*Va))*d_alpha \
        + Cm_q*(const.MAC/(2*Va))*d_theta \
        + Cm_delta_e*delta_e \
        + Cm_k*(1/Va)

    #---------------------------
    # 同定結果を用いて空力を再現
    #---------------------------

    L_calc = (1/2)*RHO*const.S*(Va**2)*CL
    D_calc = (1/2)*RHO*const.S*(Va**2)*CD
    Ma_calc = (1/2)*RHO*const.S*(Va**2)*const.MAC*Cm

    #---------------------------
    # k_*を計算
    #---------------------------

    k_L = (1/2)*RHO*const.S*CL_k
    k_D = (1/2)*RHO*const.S*CD_k
    k_m = (1/2)*RHO*const.S*const.MAC*Cm_k

    #---------------------------
    # 結果をデータファイルに書き込んで返す
    #---------------------------

    format_df_return = format_df.copy()

    format_df_return['CL_0'] = CL_0
    format_df_return['CL_alpha'] = CL_alpha
    format_df_return['CL_d_alpha'] = CL_d_alpha
    format_df_return['CL_q'] = CL_q
    format_df_return['CL_delta_e'] = CL_delta_e
    format_df_return['CL_k'] = CL_k
    format_df_return['k_L'] = k_L

    format_df_return['CD_0'] = CD_0
    format_df_return['CD_alpha'] = CD_alpha
    format_df_return['CD_d_alpha'] = CD_d_alpha
    format_df_return['CD_q'] = CD_q
    format_df_return['CD_delta_e'] = CD_delta_e
    format_df_return['CD_k'] = CD_k
    format_df_return['k_D'] = k_D

    format_df_return['Cm_0'] = Cm_0
    format_df_return['Cm_alpha'] = Cm_alpha
    format_df_return['Cm_d_alpha'] = Cm_d_alpha
    format_df_return['Cm_q'] = Cm_q
    format_df_return['Cm_delta_e'] = Cm_delta_e
    format_df_return['Cm_k'] = Cm_k
    format_df_return['k_m'] = k_m

    format_df_return['CL'] = CL
    format_df_return['CD'] = CD
    format_df_return['Cm'] = Cm
    format_df_return['L_calc'] = L_calc
    format_df_return['D_calc'] = D_calc
    format_df_return['Ma_calc'] = Ma_calc

    return format_df_return


def LS_ex_non_dalpha(format_df):
    '''
    整理されたデータをもとに，
    最小二乗法を用いてパラメータ推定を行なう.
    すべて未知パラメータとして推定．d_alphaに関する項は除外．
    CDのモデル式を変更している．

    Parameters
    ----------
    format_df: pandas.DataFrame
        ピッチ角速度, 迎角, 対気速度, エレベータ舵角, 揚力, 抗力, ピッチモーメント
        のそれぞれの実験データを含むデータ群．

    Returns
    -------
    format_df_return: pandas.DataFrame
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
    RHO = np.array(format_df['RHO'])

    #---------------------------
    # 揚力
    #---------------------------

    # n*1 揚力から計算された値のリスト
    yL = L/((1/2)*RHO*(Va**2)*const.S)

    # n*5 リグレッサー（独立変数）や実験データのリスト
    xL = np.zeros((data_size,5))
    xL[:,0] = 1
    xL[:,1] = alpha
    xL[:,2] = (const.MAC/(2*Va))*d_theta
    xL[:,3] = delta_e
    xL[:,4] = 1/Va

    # 擬似逆行列を用いた最小二乗解の計算
    L_theta_hat = np.dot((np.linalg.pinv(xL)),yL)

    # 同定された未知パラメータの取り出し
    CL_0 = L_theta_hat[0]
    CL_alpha = L_theta_hat[1]
    CL_q = L_theta_hat[2]
    CL_delta_e = L_theta_hat[3]
    CL_k = L_theta_hat[4]

    # 同定結果から得られたCLを計算
    CL = CL_0 \
        + CL_alpha*alpha \
        + CL_q*(const.MAC/(2*Va))*d_theta \
        + CL_delta_e*delta_e \
        + CL_k*(1/Va)

    #---------------------------
    # 抗力
    #---------------------------

    # n*1 抗力から計算された値のリスト
    yD = D/((1/2)*RHO*(Va**2)*const.S)

    # n*5 リグレッサー（独立変数）や実験データのリスト
    xD = np.zeros((data_size,5))
    xD[:,0] = 1
    xD[:,1] = alpha
    xD[:,2] = (const.MAC/(2*Va))*d_theta
    xD[:,3] = delta_e
    xD[:,4] = 1/Va

    # 擬似逆行列を用いた最小二乗解の計算
    D_theta_hat = np.dot((np.linalg.pinv(xD)),yD)

    # 同定された未知パラメータの取り出し
    CD_0 = D_theta_hat[0]
    CD_alpha = D_theta_hat[1]
    CD_q = D_theta_hat[2]
    CD_delta_e = D_theta_hat[3]
    CD_k = D_theta_hat[4]

    # 同定結果から得られたCDを計算
    CD = CD_0 \
        + CD_alpha*alpha \
        + CD_q*(const.MAC/(2*Va))*d_theta \
        + CD_delta_e*delta_e \
        + CD_k*(1/Va)

    #---------------------------
    # モーメント
    #---------------------------

    # n*1 空力モーメントから計算された値のリスト
    ym = Ma/((1/2)*RHO*(Va**2)*const.S*const.MAC)

    # n*5 リグレッサー（独立変数）や実験データのリスト
    xm = np.zeros((data_size,5))
    xm[:,0] = 1
    xm[:,1] = alpha
    xm[:,2] = (const.MAC/(2*Va))*d_theta
    xm[:,3] = delta_e
    xm[:,4] = 1/Va

    # 擬似逆行列を用いた最小二乗解の計算
    m_theta_hat = np.dot((np.linalg.pinv(xm)),ym)

    # 同定された未知パラメータの取り出し
    Cm_0 = m_theta_hat[0]
    Cm_alpha = m_theta_hat[1]
    Cm_q = m_theta_hat[2]
    Cm_delta_e = m_theta_hat[3]
    Cm_k = m_theta_hat[4]

    # 同定結果から得られたCDを計算
    Cm = Cm_0 \
        + Cm_alpha*alpha \
        + Cm_q*(const.MAC/(2*Va))*d_theta \
        + Cm_delta_e*delta_e \
        + Cm_k*(1/Va)

    #---------------------------
    # 同定結果を用いて空力を再現
    #---------------------------

    L_calc = (1/2)*RHO*const.S*(Va**2)*CL
    D_calc = (1/2)*RHO*const.S*(Va**2)*CD
    Ma_calc = (1/2)*RHO*const.S*(Va**2)*const.MAC*Cm

    #---------------------------
    # k_*を計算
    #---------------------------

    k_L = (1/2)*RHO*const.S*CL_k
    k_D = (1/2)*RHO*const.S*CD_k
    k_m = (1/2)*RHO*const.S*const.MAC*Cm_k

    #---------------------------
    # 結果をデータファイルに書き込んで返す
    #---------------------------

    format_df_return = format_df.copy()

    format_df_return['CL_0'] = CL_0
    format_df_return['CL_alpha'] = CL_alpha
    format_df_return['CL_q'] = CL_q
    format_df_return['CL_delta_e'] = CL_delta_e
    format_df_return['CL_k'] = CL_k
    format_df_return['k_L'] = k_L

    format_df_return['CD_0'] = CD_0
    format_df_return['CD_alpha'] = CD_alpha
    format_df_return['CD_q'] = CD_q
    format_df_return['CD_delta_e'] = CD_delta_e
    format_df_return['CD_k'] = CD_k
    format_df_return['k_D'] = k_D

    format_df_return['Cm_0'] = Cm_0
    format_df_return['Cm_alpha'] = Cm_alpha
    format_df_return['Cm_q'] = Cm_q
    format_df_return['Cm_delta_e'] = Cm_delta_e
    format_df_return['Cm_k'] = Cm_k
    format_df_return['k_m'] = k_m

    format_df_return['CL'] = CL
    format_df_return['CD'] = CD
    format_df_return['Cm'] = Cm
    format_df_return['L_calc'] = L_calc
    format_df_return['D_calc'] = D_calc
    format_df_return['Ma_calc'] = Ma_calc

    return format_df_return


def LS_ex_non_dalpha_non_clk(format_df):
    '''
    整理されたデータをもとに，
    最小二乗法を用いてパラメータ推定を行なう.
    すべて未知パラメータとして推定．d_alphaに関する項は除外．
    cl_k,K_Lも除外
    CDのモデル式を変更している．

    Parameters
    ----------
    format_df: pandas.DataFrame
        ピッチ角速度, 迎角, 対気速度, エレベータ舵角, 揚力, 抗力, ピッチモーメント
        のそれぞれの実験データを含むデータ群．

    Returns
    -------
    format_df_return: pandas.DataFrame
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
    RHO = np.array(format_df['RHO'])

    #---------------------------
    # 揚力
    #---------------------------

    # n*1 揚力から計算された値のリスト
    yL = L/((1/2)*RHO*(Va**2)*const.S)

    # n*5 リグレッサー（独立変数）や実験データのリスト
    xL = np.zeros((data_size,5))
    xL[:,0] = 1
    xL[:,1] = alpha
    xL[:,2] = (const.MAC/(2*Va))*d_theta
    xL[:,3] = delta_e
    # xL[:,4] = 1/Va

    # 擬似逆行列を用いた最小二乗解の計算
    L_theta_hat = np.dot((np.linalg.pinv(xL)),yL)

    # 同定された未知パラメータの取り出し
    CL_0 = L_theta_hat[0]
    CL_alpha = L_theta_hat[1]
    CL_q = L_theta_hat[2]
    CL_delta_e = L_theta_hat[3]
    # CL_k = L_theta_hat[4]

    # 同定結果から得られたCLを計算
    CL = CL_0 \
        + CL_alpha*alpha \
        + CL_q*(const.MAC/(2*Va))*d_theta \
        + CL_delta_e*delta_e \
        # + CL_k*(1/Va)

    #---------------------------
    # 抗力
    #---------------------------

    # n*1 抗力から計算された値のリスト
    yD = D/((1/2)*RHO*(Va**2)*const.S)

    # n*5 リグレッサー（独立変数）や実験データのリスト
    xD = np.zeros((data_size,5))
    xD[:,0] = 1
    xD[:,1] = alpha
    xD[:,2] = (const.MAC/(2*Va))*d_theta
    xD[:,3] = delta_e
    xD[:,4] = 1/Va

    # 擬似逆行列を用いた最小二乗解の計算
    D_theta_hat = np.dot((np.linalg.pinv(xD)),yD)

    # 同定された未知パラメータの取り出し
    CD_0 = D_theta_hat[0]
    CD_alpha = D_theta_hat[1]
    CD_q = D_theta_hat[2]
    CD_delta_e = D_theta_hat[3]
    CD_k = D_theta_hat[4]

    # 同定結果から得られたCDを計算
    CD = CD_0 \
        + CD_alpha*alpha \
        + CD_q*(const.MAC/(2*Va))*d_theta \
        + CD_delta_e*delta_e \
        + CD_k*(1/Va)

    #---------------------------
    # モーメント
    #---------------------------

    # n*1 空力モーメントから計算された値のリスト
    ym = Ma/((1/2)*RHO*(Va**2)*const.S*const.MAC)

    # n*5 リグレッサー（独立変数）や実験データのリスト
    xm = np.zeros((data_size,5))
    xm[:,0] = 1
    xm[:,1] = alpha
    xm[:,2] = (const.MAC/(2*Va))*d_theta
    xm[:,3] = delta_e
    xm[:,4] = 1/Va

    # 擬似逆行列を用いた最小二乗解の計算
    m_theta_hat = np.dot((np.linalg.pinv(xm)),ym)

    # 同定された未知パラメータの取り出し
    Cm_0 = m_theta_hat[0]
    Cm_alpha = m_theta_hat[1]
    Cm_q = m_theta_hat[2]
    Cm_delta_e = m_theta_hat[3]
    Cm_k = m_theta_hat[4]

    # 同定結果から得られたCDを計算
    Cm = Cm_0 \
        + Cm_alpha*alpha \
        + Cm_q*(const.MAC/(2*Va))*d_theta \
        + Cm_delta_e*delta_e \
        + Cm_k*(1/Va)

    #---------------------------
    # 同定結果を用いて空力を再現
    #---------------------------

    L_calc = (1/2)*RHO*const.S*(Va**2)*CL
    D_calc = (1/2)*RHO*const.S*(Va**2)*CD
    Ma_calc = (1/2)*RHO*const.S*(Va**2)*const.MAC*Cm

    #---------------------------
    # k_*を計算
    #---------------------------

    # k_L = (1/2)*RHO*const.S*CL_k
    k_D = (1/2)*RHO*const.S*CD_k
    k_m = (1/2)*RHO*const.S*const.MAC*Cm_k

    #---------------------------
    # 結果をデータファイルに書き込んで返す
    #---------------------------

    format_df_return = format_df.copy()

    format_df_return['CL_0'] = CL_0
    format_df_return['CL_alpha'] = CL_alpha
    format_df_return['CL_q'] = CL_q
    format_df_return['CL_delta_e'] = CL_delta_e
    # format_df_return['CL_k'] = CL_k
    # format_df_return['k_L'] = k_L

    format_df_return['CD_0'] = CD_0
    format_df_return['CD_alpha'] = CD_alpha
    format_df_return['CD_q'] = CD_q
    format_df_return['CD_delta_e'] = CD_delta_e
    format_df_return['CD_k'] = CD_k
    format_df_return['k_D'] = k_D

    format_df_return['Cm_0'] = Cm_0
    format_df_return['Cm_alpha'] = Cm_alpha
    format_df_return['Cm_q'] = Cm_q
    format_df_return['Cm_delta_e'] = Cm_delta_e
    format_df_return['Cm_k'] = Cm_k
    format_df_return['k_m'] = k_m

    format_df_return['CL'] = CL
    format_df_return['CD'] = CD
    format_df_return['Cm'] = Cm
    format_df_return['L_calc'] = L_calc
    format_df_return['D_calc'] = D_calc
    format_df_return['Ma_calc'] = Ma_calc

    return format_df_return


def LS_ex_non_kv(format_df):
    '''
    整理されたデータをもとに，
    最小二乗法を用いてパラメータ推定を行なう.
    すべて未知パラメータとして推定．d_alphaに関する項も追加．
    CDのモデル式を変更している．
    kvの項は入れていない．

    Parameters
    ----------
    format_df: pandas.DataFrame
        ピッチ角速度, 迎角, 対気速度, エレベータ舵角, 揚力, 抗力, ピッチモーメント
        のそれぞれの実験データを含むデータ群．

    Returns
    -------
    format_df_return: pandas.DataFrame
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
    RHO = np.array(format_df['RHO'])

    #---------------------------
    # 揚力
    #---------------------------

    # n*1 揚力から計算された値のリスト
    yL = L/((1/2)*RHO*(Va**2)*const.S)

    # n*4 リグレッサー（独立変数）や実験データのリスト
    xL = np.zeros((data_size,4))
    xL[:,0] = 1
    xL[:,1] = alpha
    xL[:,2] = (const.MAC/(2*Va))*d_theta
    xL[:,3] = delta_e

    # 擬似逆行列を用いた最小二乗解の計算
    L_theta_hat = np.dot((np.linalg.pinv(xL)),yL)

    # 同定された未知パラメータの取り出し
    CL_0 = L_theta_hat[0]
    CL_alpha = L_theta_hat[1]
    CL_q = L_theta_hat[2]
    CL_delta_e = L_theta_hat[3]

    # 同定結果から得られたCLを計算
    CL = CL_0 \
        + CL_alpha*alpha \
        + CL_q*(const.MAC/(2*Va))*d_theta \
        + CL_delta_e*delta_e

    #---------------------------
    # 抗力
    #---------------------------

    # n*1 抗力から計算された値のリスト
    yD = D/((1/2)*RHO*(Va**2)*const.S)

    # n*4 リグレッサー（独立変数）や実験データのリスト
    xD = np.zeros((data_size,5))
    xD[:,0] = 1
    xD[:,1] = alpha
    xD[:,2] = (const.MAC/(2*Va))*d_theta
    xD[:,3] = delta_e

    # 擬似逆行列を用いた最小二乗解の計算
    D_theta_hat = np.dot((np.linalg.pinv(xD)),yD)

    # 同定された未知パラメータの取り出し
    CD_0 = D_theta_hat[0]
    CD_alpha = D_theta_hat[1]
    CD_q = D_theta_hat[2]
    CD_delta_e = D_theta_hat[3]

    # 同定結果から得られたCDを計算
    CD = CD_0 \
        + CD_alpha*alpha \
        + CD_q*(const.MAC/(2*Va))*d_theta \
        + CD_delta_e*delta_e

    #---------------------------
    # モーメント
    #---------------------------

    # n*1 空力モーメントから計算された値のリスト
    ym = Ma/((1/2)*RHO*(Va**2)*const.S*const.MAC)

    # n*6 リグレッサー（独立変数）や実験データのリスト
    xm = np.zeros((data_size,5))
    xm[:,0] = 1
    xm[:,1] = alpha
    xm[:,2] = (const.MAC/(2*Va))*d_theta
    xm[:,3] = delta_e

    # 擬似逆行列を用いた最小二乗解の計算
    m_theta_hat = np.dot((np.linalg.pinv(xm)),ym)

    # 同定された未知パラメータの取り出し
    Cm_0 = m_theta_hat[0]
    Cm_alpha = m_theta_hat[1]
    Cm_q = m_theta_hat[2]
    Cm_delta_e = m_theta_hat[3]

    # 同定結果から得られたCDを計算
    Cm = Cm_0 \
        + Cm_alpha*alpha \
        + Cm_q*(const.MAC/(2*Va))*d_theta \
        + Cm_delta_e*delta_e

    #---------------------------
    # 同定結果を用いて空力を再現
    #---------------------------

    L_calc = (1/2)*RHO*const.S*(Va**2)*CL
    D_calc = (1/2)*RHO*const.S*(Va**2)*CD
    Ma_calc = (1/2)*RHO*const.S*(Va**2)*const.MAC*Cm

    #---------------------------
    # 結果をデータファイルに書き込んで返す
    #---------------------------

    format_df_return = format_df.copy()

    format_df_return['CL_0'] = CL_0
    format_df_return['CL_alpha'] = CL_alpha
    format_df_return['CL_q'] = CL_q
    format_df_return['CL_delta_e'] = CL_delta_e

    format_df_return['CD_0'] = CD_0
    format_df_return['CD_alpha'] = CD_alpha
    format_df_return['CD_q'] = CD_q
    format_df_return['CD_delta_e'] = CD_delta_e

    format_df_return['Cm_0'] = Cm_0
    format_df_return['Cm_alpha'] = Cm_alpha
    format_df_return['Cm_q'] = Cm_q
    format_df_return['Cm_delta_e'] = Cm_delta_e

    format_df_return['CL'] = CL
    format_df_return['CD'] = CD
    format_df_return['Cm'] = Cm
    format_df_return['L_calc'] = L_calc
    format_df_return['D_calc'] = D_calc
    format_df_return['Ma_calc'] = Ma_calc

    return format_df_return
