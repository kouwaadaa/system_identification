import numpy as np
import pandas as pd

import const
import fileread as fr

#---------------------------
# ログデータの読み込み
#---------------------------

# 読み込みデータ初期化
format_df = pd.DataFrame()

format_df = fr.file_read_thrust('../log_data/Book5.csv',12.45,13.66,-4.80,1.266,0,format_df)
    
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

print(T_SUB_TOTAL_mean/T_TOTAL_mean) 


# サブ出力割合
# SUB_THRUST_PER = 0.3

# T_E_S = MG*SUB_THRUST_PER / T_SUB_TOTAL_mean
# T_E_M = (MG - T_E_S*T_SUB_TOTAL_mean) / T_M_mean

# T_E_R = T_E_S*T_SUB_TOTAL_mean / (share_per*T_F_mean + T_R_mean)
# T_E_F = T_E_R*share_per

# T_EFF_30_array = np.array([T_E_M,T_E_R,T_E_F,T_R_mean,T_F_mean])