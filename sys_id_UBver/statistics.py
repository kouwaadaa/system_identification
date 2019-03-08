# -*- coding: utf-8 -*-
'''
author: ub
統計的数値を算出する関数群．
'''

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import const
import math_extention as matex


def calc_RMSE(format_df):
    '''
    RMSEを算出して，表示する関数．
    '''


    CL_log = np.array(format_df['CL_log'])
    CL = np.array(format_df['CL'])

    CL_RMSE = np.sqrt(mean_squared_error(CL_log,CL))
    CL_MAE = mean_absolute_error(CL_log,CL)
    CL_h = CL_RMSE / CL_MAE

    CD_log = np.array(format_df['CD_log'])
    CD = np.array(format_df['CD'])

    CD_RMSE = np.sqrt(mean_squared_error(CD_log,CD))
    CD_MAE = mean_absolute_error(CD_log,CD)
    CD_h = CD_RMSE / CD_MAE

    Cm_log = np.array(format_df['Cm_log'])
    Cm = np.array(format_df['Cm'])

    Cm_RMSE = np.sqrt(mean_squared_error(Cm_log,Cm))
    Cm_MAE = mean_absolute_error(Cm_log,Cm)
    Cm_h = Cm_RMSE / Cm_MAE

    #---------------------------
    # 結果を表示する．
    #---------------------------

    print(f'CL_RMSE:{CL_RMSE:.10f} CD_RMSE:{CD_RMSE:.10f} Cm_RMSE:{Cm_RMSE:.10f}')
    print(f'CL_MAE:{CL_MAE:.10f} CD_MAE:{CD_MAE:.10f} Cm_MAE:{Cm_MAE:.10f}')
    print(f'CL_h:{CL_h:.10f} CD_h:{CD_h:.10f} Cm_h:{Cm_h:.10f}\n')

    return 0
