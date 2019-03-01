# -*- coding: utf-8 -*-
'''
author: ub
推力を算出する関数群．
'''

import numpy as np
import pandas as pd

import const
import file_read as fr


def calc_thrust_eff(share_per):

    #---------------------------
    # ログデータの読み込み
    #---------------------------

    # 読み込みデータ初期化
    format_df = pd.DataFrame()

    #---2017/12/27 徳島 定点ホバリング MCパラメータ変更-------------------------------------------
    format_df = fr.file_read_thrust('../log_data/Book8.csv',15.41,20.10,-2.00,1.275,0,format_df)
    format_df = fr.file_read_thrust('../log_data/Book8.csv',21.46,23.07,-2.00,1.275,0,format_df)
    format_df = fr.file_read_thrust('../log_data/Book8.csv',23.44,24.64,-2.00,1.275,0,format_df)
    format_df = fr.file_read_thrust('../log_data/Book8.csv',25.28,27.38,-2.00,1.275,0,format_df)

    format_df_filt = format_df.query('-0.2 <= theta <= 0.2')

    Tm_up = np.array(format_df_filt['Tm_up'])
    Tm_down = np.array(format_df_filt['Tm_down'])
    Tr_r = np.array(format_df_filt['Tr_r'])
    Tr_l = np.array(format_df_filt['Tr_l'])
    Tf_up = np.array(format_df_filt['Tf_up'])
    Tf_down = np.array(format_df_filt['Tf_down'])

    T_M = Tm_up + Tm_down
    T_R = Tr_r + Tr_l
    T_F = Tf_up + Tf_down
    T_SUB_TOTAL = T_R + T_F
    T_TOTAL = T_M + T_R + T_F

    T_M_mean = np.mean(T_M)
    T_R_mean = np.mean(T_R)
    T_F_mean = np.mean(T_F)
    T_SUB_TOTAL_mean = np.mean(T_SUB_TOTAL)
    T_TOTAL_mean = np.mean(T_TOTAL)

    MG = const.MASS*const.GRA

    # サブ出力割合
    SUB_THRUST_PER = 0.3

    T_E_S = MG*SUB_THRUST_PER / T_SUB_TOTAL_mean
    T_E_M = (MG - T_E_S*T_SUB_TOTAL_mean) / T_M_mean

    T_E_R = T_E_S*T_SUB_TOTAL_mean / (share_per*T_F_mean + T_R_mean)
    T_E_F = T_E_R*share_per

    T_EFF_30_array = np.array([T_E_M,T_E_R,T_E_F,T_R_mean,T_F_mean])


    # 読み込みデータ初期化
    format_df = pd.DataFrame()

    #---2017/12/27 徳島 定点ホバリング MCパラメータ変更-------------------------------------------
    format_df = fr.file_read_thrust('../log_data/Book3.csv',17.52,19.14,-4.03,1.264,0,format_df)

    Tm_up = np.array(format_df['Tm_up'])
    Tm_down = np.array(format_df['Tm_down'])
    Tr_r = np.array(format_df['Tr_r'])
    Tr_l = np.array(format_df['Tr_l'])
    Tf_up = np.array(format_df['Tf_up'])
    Tf_down = np.array(format_df['Tf_down'])

    T_M = Tm_up + Tm_down
    T_R = Tr_r + Tr_l
    T_F = Tf_up + Tf_down
    T_SUB_TOTAL = T_R + T_F
    T_TOTAL = T_M + T_R + T_F

    T_M_mean = np.mean(T_M)
    T_R_mean = np.mean(T_R)
    T_F_mean = np.mean(T_F)
    T_SUB_TOTAL_mean = np.mean(T_SUB_TOTAL)
    T_TOTAL_mean = np.mean(T_TOTAL)

    MG = const.MASS*const.GRA

    # サブ出力割合
    SUB_THRUST_PER = 0.35

    T_E_S = MG*SUB_THRUST_PER / T_SUB_TOTAL_mean
    T_E_M = (MG - T_E_S*T_SUB_TOTAL_mean) / T_M_mean

    T_E_R = T_E_S*T_SUB_TOTAL_mean / (share_per*T_F_mean + T_R_mean)
    T_E_F = T_E_R*share_per

    T_EFF_35_array = np.array([T_E_M,T_E_R,T_E_F,T_M_mean,T_R_mean,T_F_mean])

    return(T_EFF_30_array,T_EFF_35_array)
