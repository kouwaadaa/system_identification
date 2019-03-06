# -*- coding: utf-8 -*-
'''
author: ub
2018/12/14 Mon. 新座標系に変更．
ティルト角0度の実験データを使う．
ファイル読み込み以外変更なし．
'''

#---------------------------
# モジュールのインポートなど
#---------------------------

import numpy as np
from numpy import pi
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib.font_manager
from IPython import get_ipython

import const
import math_extention as matex
import file_read
import calc_thrust
import sys_id
import analyze
import statistics
import plot

#---------------------------
# matplotlibの諸設定
#---------------------------

# プロットデータを新しいウィンドウで表示する
get_ipython().run_line_magic('matplotlib', 'qt')

# 日本語フォントの設定
# 使用できるフォントを確認したいときは，次の行のコメントアウトを外して実行
# print([f.name for f in matplotlib.font_manager.fontManager.ttflist])

# # for NotePC
# plt.rc('font', **{'family':'Gen Shin Gothic'})

# # for DeskPC
# plt.rc('font', **{'family':'YuGothic'})

# plt.rcParams['font.size'] = 20
# plt.rcParams['xtick.labelsize'] = 15
# plt.rcParams['ytick.labelsize'] = 15 # default: 12
#
# プロットデータのサイズ設定
plt.rcParams["figure.figsize"] = [20, 12]

#---------------------------
# 推力効率係数の算出
#---------------------------

T_EFF_30 = calc_thrust.calc_thrust_eff(1.0)[0]
T_EFF_35 = calc_thrust.calc_thrust_eff(1.0)[1]

#---------------------------
# ログデータの読み込み
#---------------------------

# 読み込みデータ初期化
format_df = pd.DataFrame()

# データ群ごとに線引き，綺麗でない
borderline_list = list()

#---2017/12/27 徳島 定点ホバリング MCパラメータ変更------------------------------------------------------
# format_df,size = file_read.file_read(300,'../log_data/Book3.csv',17.52,19.14,-4.03,T_EFF_35,1.264,0,format_df)
# borderline_list.append(size)
#
# #---2017/12/27 徳島 エレベータ（ピッチアップ） MCパラメータ変更------------------------------------------
# format_df,size = file_read.file_read(400,'../log_data/Book4.csv',11.97,13.30,-5.05,T_EFF_35,1.264,0,format_df)
format_df,size = file_read.file_read(400,'../log_data/Book4.csv',18.66,21.08,-5.05,T_EFF_35,1.264,0,format_df)
# borderline_list.append(size+borderline_list[-1])
#
# #---2017/12/27 徳島 エレベータ（ピッチダウン） MCパラメータ変更-------------------------------------------
format_df,size = file_read.file_read(500,'../log_data/Book5.csv',12.45,13.66,-4.80,T_EFF_35,1.266,0,format_df)
# borderline_list.append(size+borderline_list[-1])
format_df,size = file_read.file_read(500,'../log_data/Book5.csv',16.07,17.03,-4.80,T_EFF_35,1.266,0,format_df)
# borderline_list.append(size+borderline_list[-1])
format_df,size = file_read.file_read(500,'../log_data/Book5.csv',18.95,22.88,-4.80,T_EFF_35,1.266,0,format_df)
# borderline_list.append(size+borderline_list[-1])

# #---2018/01/14 徳島 ピッチ運動-----------------------------------------------------------
format_df,size = file_read.file_read(800,'../log_data/Book8.csv',15.41,20.10,-2.00,T_EFF_30,1.275,0,format_df)
# # borderline_list.append(size+borderline_list[-1])
format_df,size = file_read.file_read(800,'../log_data/Book8.csv',21.46,23.07,-2.00,T_EFF_30,1.275,0,format_df)
# # borderline_list.append(size+borderline_list[-1])
format_df,size = file_read.file_read(800,'../log_data/Book8.csv',23.44,24.64,-2.00,T_EFF_30,1.275,0,format_df)
# # borderline_list.append(size+borderline_list[-1])
format_df,size = file_read.file_read(800,'../log_data/Book8.csv',25.28,27.38,-2.00,T_EFF_30,1.275,0,format_df)
# # borderline_list.append(size+borderline_list[-1])
#
# #---2018/01/26 神戸 前進＆エレベータ制御-------------------------------------------------
format_df,size = file_read.file_read(900,'../log_data/Book9.csv',20.73,30.28,-2.647,T_EFF_30,1.251,0,format_df)
# # borderline_list.append(size+borderline_list[-1])
format_df,size = file_read.file_read(900,'../log_data/Book9.csv',98.05,104.1,-2.647,T_EFF_30,1.251,0,format_df)
# # borderline_list.append(size+borderline_list[-1])
format_df,size = file_read.file_read(900,'../log_data/Book9.csv',104.9,107.1,-2.647,T_EFF_30,1.251,0,format_df)
# # borderline_list.append(size+borderline_list[-1])
format_df,size = file_read.file_read(900,'../log_data/Book9.csv',107.7,109.7,-2.647,T_EFF_30,1.251,0,format_df)
# # borderline_list.append(size+borderline_list[-1])
#
# #---2018/01/26 神戸 前進-----------------------------------------------------------------
format_df,size = file_read.file_read(1100,'../log_data/Book11.csv',19.86,25.27,-1.467,T_EFF_30,1.268,0,format_df)
# format_df,size = file_read.file_read(1100,'../log_data/Book11.csv',26.43,29.83,-1.467,T_EFF_30,1.268,0,format_df)
# #---------------------------------------------------------

#---------------------------
# データの整理
#---------------------------

# indexの振り直し
format_df = format_df.reset_index()

#---------------------------
# パラメータ推定の結果を計算し，取得
#---------------------------

df_non_dalpha = sys_id.sys_id_LS(format_df)
df_with_dalpha = sys_id.sys_id_LS_with_dalpha(format_df)
df_non_kv = sys_id.sys_id_LS_non_kv(format_df)

df_ex_non_dalpha = sys_id.sys_id_LS_ex_non_dalpha(format_df)
df_ex_with_dalpha = sys_id.sys_id_LS_ex_with_dalpha(format_df)
df_ex_non_kv = sys_id.sys_id_LS_ex_non_kv(format_df)

#---------------------------
# 機体の状態方程式から固有振動数を解析する
#---------------------------

# anly_result = analyze.linearlize(df_with_dalpha)
anly_result = analyze.linearlize_non_d_alpha(df_non_dalpha)

#---------------------------
# 統計データ算出
#---------------------------

# df_non_dalpha = statistics.calc_RMSE(df_non_dalpha)
# df_with_dalpha = statistics.calc_RMSE(df_with_dalpha)
# df_non_kv = statistics.calc_RMSE(df_non_kv)
df_ex_non_dalpha = statistics.calc_RMSE(df_ex_non_dalpha)
# df_ex_with_dalpha = statistics.calc_RMSE(df_ex_with_dalpha)
df_ex_non_kv = statistics.calc_RMSE(df_ex_non_kv)

#---------------------------
# データ取り出し作業
#---------------------------

# 合計のデータサイズを取得
data_size = len(format_df)

#---------------------------
# 結果をプロット
#---------------------------

# dfdf = format_df[format_df['Time_DIFF'] >= 0.03]
# print(dfdf['index'])

df_ex_with_dalpha[['alpha','theta']].plot.line()
df_ex_with_dalpha[['d_alpha','d_theta']].plot.line()
# df_non_kv[['D','D_calc']].plot.line()
# df_non_kv[['Ma','Ma_calc']].plot.line()
#
# df_with_dalpha[['L','L_calc']].plot.line()
# df_with_dalpha[['D','D_calc']].plot.line()
# df_with_dalpha[['Ma','Ma_calc']].plot.line()
#
# df_ex_non_dalpha[['CL_log','CL','alpha_deg']].plot.line(x='alpha_deg', style='o', title='CL')
# df_ex_non_dalpha[['CD_log','CD','alpha_deg']].plot.line(x='alpha_deg', style='o', title='CD')
# df_ex_non_dalpha[['Cm_log','Cm','alpha_deg']].plot.line(x='alpha_deg', style='o', title='Cm')

# df_with_dalpha[['CL_log','CL']].plot.line(title='CL_max')
# df_with_dalpha[['CD_log','CD']].plot.line(title='CD_max')
# df_with_dalpha[['Cm_log','Cm']].plot.line(title='Cm_max')

# df_non_dalpha[['CL_log','CL']].plot.line(title='CL_nonda')
# df_non_dalpha[['CD_log','CD']].plot.line(title='CD_nonda')
# df_non_dalpha[['Cm_log','Cm']].plot.line(title='Cm_nonda')

# format_df7[['CL_log','CL']].plot.line(title='CL_complete')
# format_df7[['CD_log','CD']].plot.line(title='CD_complete')
# format_df7[['Cm_log','Cm']].plot.line(title='Cm_complete')

# format_df[['L_calc','alpha_deg']].plot.line(x='alpha_deg', style=['o'])
# format_df[['D_calc','alpha_deg']].plot.line(x='alpha_deg', style=['o'])
# format_df[['Ma_calc','alpha_deg']].plot.line(x='alpha_deg', style=['o'])

#----------------------------------------------------------------

# ax = fig.add_subplot(2,1,2)
#
# ax.plot(xxx,d_alpha)

# df_with_dalpha[['Va']].plot.line()

# d_theta = np.array(df_with_dalpha['d_theta'])
# d_theta_filt = np.array(df_with_dalpha['d_theta_filt'])
#
# plt.subplot(111)
# plt.plot(d_theta,label="Raw data")
# plt.plot(d_theta_filt,label="Filtered data",linewidth="3")
# plt.legend()
#
# plt.xlabel('Data Number')
# plt.ylabel('Pitch Rate [rad/s]')
# plt.show()

#----------------------------------------------------------------

# L = np.array(df_with_dalpha['L'])
# L_calc = np.array(df_with_dalpha['L_calc'])
# D = np.array(df_with_dalpha['D'])
# D_calc = np.array(df_with_dalpha['D_calc'])
# Ma = np.array(df_with_dalpha['Ma'])
# Ma_calc = np.array(df_with_dalpha['Ma_calc'])
#
# plt.subplot(111)
# plt.plot(Ma,label=r"$M_{a_{log}}$")
# plt.plot(Ma_calc,label=r"$M_{a_{calc}}$")
# plt.legend()
#
# for j in borderline_list:
#     plt.axvline(x=j, color="black",linestyle="--") # 実験データの境目で線を引く
#
# plt.xlabel('Data Number')
# plt.ylabel(r'Pitch moment$\mathrm{[N \cdot m]}$')
# plt.show()
#----------------------------------------------------------------
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
# #----------------------------------------------------------------

# スラスト出力確認
# Tr_r = np.array(df_with_dalpha['Tr_r'])
# Tr_l = np.array(df_with_dalpha['Tr_l'])
# Tf_up = np.array(df_with_dalpha['Tf_up'])
# Tf_down = np.array(df_with_dalpha['Tf_down'])
#
# T_R_mean = np.mean(Tr_r+Tr_l)
# T_F_mean = np.mean(Tf_up+Tf_down)

#----------------------------------------------------------------
# Va横軸で空力係数を比較
# plot.plot_CL_compare_model(df_ex_non_kv, df_ex_non_dalpha)
# plot.plot_CD_compare_model(df_ex_non_kv, df_ex_non_dalpha)
# plot.plot_Cm_compare_model(df_ex_non_kv, df_ex_non_dalpha)

# CFDと同定結果との比較用
# plot.plot_CL_compare_CFD(df_ex_non_dalpha)
# plot.plot_CD_compare_CFD(df_ex_non_dalpha)
# plot.plot_Cm_compare_CFD(df_ex_non_dalpha)

# 固有値の絶対値
# plot.plot_eigen_abs(anly_result)
