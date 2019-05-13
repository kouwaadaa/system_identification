# -*- coding: utf-8 -*-
'''
author: ub
2019/02/25 Mon. 新座標系．
ティルト角10度の実験データを使う．
ファイル読み込み以外変更なし．
'''

#---------------------------
# モジュールのインポートなど
#---------------------------

import numpy as np
from numpy import pi
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from IPython import get_ipython

import const
import math_extention as matex
import fileread as file_read
import param_estimation as sys_id
import analyze
import statistics

#---------------------------
# matplotlibの諸設定
#---------------------------

# プロットデータを新しいウィンドウで表示する
#get_ipython().run_line_magic('matplotlib', 'qt')

# 日本語フォントの設定
# # for NotePC
# plt.rc('font', **{'family':'Gen Shin Gothic'})

# # for DeskPC
# plt.rc('font', **{'family':'YuGothic'})

plt.rcParams['font.size'] = 20
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15 # default: 12

# プロットデータのサイズ設定
plt.rcParams["figure.figsize"] = [20, 12]

#---------------------------
# ログデータの読み込み
#---------------------------

# 読み込みデータ初期化
format_df = pd.DataFrame()

# データ群ごとに線引き，綺麗でない
borderline_list = list()

#---2017/12/02 徳島 前進------------------------------------------------------
# format_df,size = file_read.file_read('../log_data/Book2.csv',54.35,59.46,-2.30,40/45,1.252,10,format_df)
# borderline_list.append(size)
# format_df,size = file_read.file_read('../log_data/Book2.csv',70.68,83.38,-2.30,40/45,1.252,10,format_df)
# borderline_list.append(size+borderline_list[-1])

#---2018/01/14 徳島 エレベータ（ピッチアップ）------------------------------------------
format_df,size = file_read.file_read(600,'../log_data/Book6.csv',17.47,19.08,-2.00,40/47,1.282,10,format_df)
borderline_list.append(size)
# borderline_list.append(size+borderline_list[-1])
format_df,size = file_read.file_read(600,'../log_data/Book6.csv',21.36,24.67,-2.00,40/47,1.282,10,format_df)
borderline_list.append(size+borderline_list[-1])
format_df,size = file_read.file_read(600,'../log_data/Book6.csv',63.34,68.20,-2.00,40/47,1.282,10,format_df)
borderline_list.append(size+borderline_list[-1])

#---2018/01/14 徳島 エレベータ（制御orピッチアップ）------------------------------------
# format_df,size = file_read.file_read(700,'../log_data/Book7.csv',13.93,15.36,-1.25,40/47,1.281,10,format_df)
# borderline_list.append(size+borderline_list[-1])
# format_df,size = file_read.file_read(700,'../log_data/Book7.csv',55.40,56.85,-1.25,40/47,1.281,10,format_df)
# borderline_list.append(size+borderline_list[-1])
# format_df,size = file_read.file_read(700,'../log_data/Book7.csv',62.60,64.83,-1.25,40/47,1.281,10,format_df)
# borderline_list.append(size+borderline_list[-1])

#---2018/01/14 徳島 エレベータ（ピッチ運動＆前進）------------------------------------------
format_df,size = file_read.file_read(801,'../log_data/Book8.csv',43.24,45.90,-2.00,40/47,1.275,10,format_df)
borderline_list.append(size+borderline_list[-1])
format_df,size = file_read.file_read(802,'../log_data/Book8.csv',61.44,64.48,-2.00,40/47,1.275,10,format_df)
borderline_list.append(size+borderline_list[-1])
format_df,size = file_read.file_read(803,'../log_data/Book8.csv',71.60,80.56,-2.00,40/47,1.275,10,format_df)
borderline_list.append(size+borderline_list[-1])
format_df,size = file_read.file_read(804,'../log_data/Book8.csv',101.9,109.6,-2.00,40/47,1.275,10,format_df)
borderline_list.append(size+borderline_list[-1])

#---2018/01/26 徳島 前進＆エレベータ制御---------------------------------------------------
format_df,size = file_read.file_read(1000,'../log_data/Book10.csv',15.56,17.40,-3.277,40/48,1.260,10,format_df)
borderline_list.append(size+borderline_list[-1])
format_df,size = file_read.file_read(1000,'../log_data/Book10.csv',94.13,101.5,-3.277,40/48,1.260,10,format_df)
borderline_list.append(size+borderline_list[-1])
format_df,size = file_read.file_read(1000,'../log_data/Book10.csv',103.6,105.3,-3.277,40/48,1.260,10,format_df)
borderline_list.append(size+borderline_list[-1])
#---------------------------------------------------------

#---------------------------
# データの整理
#---------------------------

# indexの振り直し
# ここで新たに"index"という列が生成されるが，
# これを残しておけばログデータごとのプロットがしやすい．
format_df = format_df.reset_index()

#---------------------------
# パラメータ推定の結果を計算し，取得
#---------------------------

df_non_dalpha = sys_id.LS_non_dalpha(format_df)
df_with_dalpha = sys_id.LS_with_dalpha(format_df)
df_non_kv = sys_id.LS_non_kv(format_df)

df_ex_non_dalpha = sys_id.LS_ex_non_dalpha(format_df)
df_ex_with_dalpha = sys_id.LS_ex_with_dalpha(format_df)
df_ex_non_kv = sys_id.LS_ex_non_kv(format_df)

#---------------------------
# 機体の状態方程式から固有振動数を解析する
#---------------------------

# anly_result = analyze.linearlize(df_with_dalpha)
anly_result = analyze.linearlize_non_d_alpha(df_non_dalpha)

#---------------------------
# 統計データ算出
#---------------------------

df_non_dalpha = statistics.calc_RMSE(df_non_dalpha)
df_with_dalpha = statistics.calc_RMSE(df_with_dalpha)
df_non_kv = statistics.calc_RMSE(df_non_kv)
df_ex_non_dalpha = statistics.calc_RMSE(df_ex_non_dalpha)
df_ex_with_dalpha = statistics.calc_RMSE(df_ex_with_dalpha)
df_ex_non_kv = statistics.calc_RMSE(df_ex_non_kv)

#---------------------------
# データの取り出し
#---------------------------

data_size = len(format_df) # 合計のデータサイズを取得

#---------------------------
# 結果をプロット
#---------------------------

# df5_V_filter = df_with_dalpha.query('4.5 <= Va <= 5.5')

Tr_r = np.array(df_with_dalpha['Tr_r'])
Tr_l = np.array(df_with_dalpha['Tr_l'])
Tf_up = np.array(df_with_dalpha['Tf_up'])
Tf_down = np.array(df_with_dalpha['Tf_down'])

T_R_mean = np.mean(Tr_r+Tr_l)
T_F_mean = np.mean(Tf_up+Tf_down)

# プロットするときに実験データごとに見られるようにする．
grouped_df = df_ex_non_kv.groupby('id')
#
# #------------------------------------------------------------------
# df_with_dalpha[['alpha_deg']].plot.line()
# for j in borderline_list:
#     plt.axvline(x=j, color="black",linestyle="--") # 実験データの境目で線を引く
# df_with_dalpha[['CD_log','CD','Va']].plot.line(x='Va', style='o')
# df_with_dalpha[['Cm_log','Cm','Va']].plot.line(x='Va', style='o')
# plt.tight_layout()
#------------------------------------------------------------------
# Va横軸で空力係数比較
Va = np.array(df_ex_non_kv['Va'])
CD_log = np.array(df_ex_non_kv['CD_log'])

# plt.figure(figsize=(12,10))
plt.subplot(111)
plt.scatter(Va,CD_log,color="#333333")

for count,id in enumerate(grouped_df.groups):
    # if id == 600 or id == 1000:
    #     a = 1
    # else:
       d = grouped_df.get_group(id)
       v_array = np.array(d['Va'])
       d_array = np.array(d['CD'])
       plt.scatter(v_array,d_array,color=cm.Set1(count/9)) # Set1は9色まで

# plt.xlabel(r'$V_a \mathrm{[m s^{-1}]}$')
# plt.ylabel(r'$C_D$')
plt.tight_layout()
#------------------------------------------------------------------
# # Va横軸で空力係数比較
# Va = np.array(df_with_dalpha['Va'])
# CD_log = np.array(df_with_dalpha['CD_log'])
# CD_non_kv = np.array(df_non_kv['CD']) # non kV
# CD_with_dalpha = np.array(df_with_dalpha['CD']) # max
# # CD_non_dalpha = np.array(df_non_dalpha['CD']) # non d_alpha
#
# # plt.figure(figsize=(12,10))
# plt.subplot(111)
# plt.scatter(Va,CD_log,label="Data1: log data",linewidth="3")
# plt.scatter(Va,CD_non_kv,label=r"Data2: model without $k_DV_a$")
# plt.scatter(Va,CD_with_dalpha,label=r"Data3: model with $k_DV_a$")
# # plt.scatter(Va,CD_non_dalpha,label=r"Model:No $\dot{\alpha}$")
# plt.legend()
#
# plt.xlabel(r'$V_a \mathrm{[m s^{-1}]}$')
# plt.ylabel(r'$C_D$')
# plt.tight_layout()
#------------------------------------------------------------------
# # 各空気力の再現性を見る
# L = np.array(df_with_dalpha['L_calc'])
# L_log = np.array(df_with_dalpha['L'])
# D = np.array(df_with_dalpha['D_calc'])
# D_log = np.array(df_with_dalpha['D'])
# Ma = np.array(df_with_dalpha['Ma_calc'])
# Ma_log = np.array(df_with_dalpha['Ma'])
#
# # plt.figure(figsize=(12,10))
# plt.subplot(111)
# plt.plot(D_log,label=r"$D_{log}$")
# plt.plot(D,label=r"$D_{calc}$")
# plt.legend()
#
# for j in borderline_list:
#     plt.axvline(x=j, color="black",linestyle="--") # 実験データの境目で線を引く
#
# plt.xlabel('Data Number')
# plt.ylabel(r'Lift$\mathrm{[N]}$')
# plt.tight_layout()
#----------------------------------------------------------------
# lambda_A_abs = np.abs(anly_result[0])
#
# xxx = np.arange(data_size)
# y = lambda_A_abs[:,0]
# yy = lambda_A_abs[:,1]
# yyy = lambda_A_abs[:,2]
# yyyy = lambda_A_abs[:,3]
#
# plt.subplot(111)
# plt.scatter(xxx,y,label="")
# plt.scatter(xxx,yy,label="")
# plt.scatter(xxx,yyy,label="")
# plt.scatter(xxx,yyyy,label="")
#
# for j in borderline_list:
#     plt.axvline(x=j, color="black", linestyle="--") # 実験データの境目で線を引く
#
# # plt.title('固有値散布図')
# plt.xlabel('Data Number')
# plt.ylabel('Absolute eigenvalue')
# plt.tight_layout()
