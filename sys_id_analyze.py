# -*- coding: utf-8 -*-
# author: ub
# 2018/12/14 Fri. 新座標系．

#---------------------------
# モジュールのインポートなど
#---------------------------

import const
import numpy as np
from numpy import pi
import pandas as pd
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
        ピッチ角速度, 迎角, 対気速度, エレベータ舵角, 揚力, 抗力, ピッチモーメント
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
    u = np.array(format_log_data['u'])
    v = np.array(format_log_data['v'])
    w = np.array(format_log_data['w'])
    delta_e = np.array(format_log_data['delta_e'])
