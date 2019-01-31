# -*- coding: utf-8 -*-
# author: ub

'''
統計的数値を算出する関数群．
'''

import numpy as np
import pandas as pd

import const
import math_extention as matex

def calc_RMSE(format_df):
    '''

    '''

    data_size = len(format_df)

    CL_log = np.array(format_df['CL_log'])
    CL = np.array(format_df['CL'])

    CL_RMSE = np.sqrt(((CL-CL_log)**2).mean())

    CD_log = np.array(format_df['CD_log'])
    CD = np.array(format_df['CD'])

    CD_RMSE = np.sqrt(((CD-CD_log)**2).mean())

    Cm_log = np.array(format_df['Cm_log'])
    Cm = np.array(format_df['Cm'])

    Cm_RMSE = np.sqrt(((Cm-Cm_log)**2).mean())

    #---------------------------
    # 結果をデータファイルに書き込んで返す
    #---------------------------

    format_df_return = format_df.copy()

    format_df_return['CL_RMSE'] = CL_RMSE
    format_df_return['CD_RMSE'] = CD_RMSE
    format_df_return['Cm_RMSE'] = Cm_RMSE

    print(f'CL_RMSE:{CL_RMSE:.10f} CD_RMSE:{CD_RMSE:.10f} Cm_RMSE:{Cm_RMSE:.10f}')

    return format_df_return
