# -*- coding: utf-8 -*-
# author: ub
# 2018/12/14 Fri. 新座標系．

#---------------------------
# モジュールのインポートなど
#---------------------------

import numpy as np
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
    d_theta = np.array(format_log_data['d_theta'])
    alpha = np.array(format_log_data['alpha'])
    d_alpha = np.array(format_log_data['d_alpha'])
    u = np.array(format_log_data['u'])
    v = np.array(format_log_data['v'])
    w = np.array(format_log_data['w'])
    delta_e = np.array(format_log_data['delta_e'])
    CL_0 = np.array(format_log_data['CL_0'])
    CL_alpha = np.array(format_log_data['CL_alpha'])
    CL_q = np.array(format_log_data['CL_q'])
    CL_delta_e = np.array(format_log_data['CL_delta_e'])
    k_L = np.array(format_log_data['k_L'])
    CD_0 = np.array(format_log_data['CD_0'])
    kappa = np.array(format_log_data['kappa'])
    k_D = np.array(format_log_data['k_D'])
    Cm_0 = np.array(format_log_data['Cm_0'])
    Cm_alpha = np.array(format_log_data['Cm_alpha'])
    Cm_q = np.array(format_log_data['Cm_q'])
    Cm_delta_e = np.array(format_log_data['Cm_delta_e'])
    k_m = np.array(format_log_data['k_m'])
    CL = np.array(format_log_data['CL'])
    CD = np.array(format_log_data['CD'])
    Cm = np.array(format_log_data['Cm'])

    #---------------------------
    # 状態方程式 dx = Ax + Bu の，AとBを計算する．
    #---------------------------
    
