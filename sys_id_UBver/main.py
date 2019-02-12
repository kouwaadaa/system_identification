# -*- coding: utf-8 -*-
# author: ub
# 2018/12/14 Fri. 新座標系．

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
import sys_id
import analyze
import statistics

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
plt.rc('font', **{'family':'YuGothic'})

plt.rcParams['font.size'] = 20
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15 # default: 12

# プロットデータのサイズ設定
plt.rcParams["figure.figsize"] = [20, 12]

#---------------------------
# ログデータの読み込み
#---------------------------

format_df = pd.DataFrame()
borderline_list = list()

format_df,size = file_read.file_read('../log_data/Book3.csv',17.52,19.14,-4.03,40/48,0,format_df)
borderline_list.append(size)
# format_df = file_read.file_read('../log_data/Book4.csv',11.97,13.30,-5.05,40/45,0,format_df)
format_df,size = file_read.file_read('../log_data/Book4.csv',18.66,21.08,-5.05,40/45,0,format_df)
borderline_list.append(size+borderline_list[-1])

format_df,size = file_read.file_read('../log_data/Book5.csv',12.45,13.66,-4.80,40/48,0,format_df)
borderline_list.append(size+borderline_list[-1])
format_df,size = file_read.file_read('../log_data/Book5.csv',16.07,17.03,-4.80,40/48,0,format_df)
borderline_list.append(size+borderline_list[-1])
format_df,size = file_read.file_read('../log_data/Book5.csv',18.95,22.88,-4.80,40/48,0,format_df)
borderline_list.append(size+borderline_list[-1])

format_df,size = file_read.file_read('../log_data/Book8.csv',15.41,20.10,-2.00,40/47,0,format_df)
borderline_list.append(size+borderline_list[-1])
format_df,size = file_read.file_read('../log_data/Book8.csv',21.46,23.07,-2.00,40/47,0,format_df)
borderline_list.append(size+borderline_list[-1])
format_df,size = file_read.file_read('../log_data/Book8.csv',23.44,24.64,-2.00,40/47,0,format_df)
borderline_list.append(size+borderline_list[-1])
format_df,size = file_read.file_read('../log_data/Book8.csv',25.28,27.38,-2.00,40/47,0,format_df)
borderline_list.append(size+borderline_list[-1])

format_df,size = file_read.file_read('../log_data/Book9.csv',20.73,30.28,-2.647,40/48,0,format_df)
borderline_list.append(size+borderline_list[-1])
format_df,size = file_read.file_read('../log_data/Book9.csv',98.05,104.1,-2.647,40/48,0,format_df)
borderline_list.append(size+borderline_list[-1])
format_df,size = file_read.file_read('../log_data/Book9.csv',104.9,107.1,-2.647,40/48,0,format_df)
borderline_list.append(size+borderline_list[-1])
format_df,size = file_read.file_read('../log_data/Book9.csv',107.7,109.7,-2.647,40/48,0,format_df)
borderline_list.append(size+borderline_list[-1])
# format_df,size = file_read.file_read('../log_data/Book11.csv',19.86,25.27,-1.467,40/48,0,format_df)

# format_df = file_read.file_read('../log_data/Book11.csv',26.43,29.83,-1.467,40/48,0,format_df)

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

# format_df1 = sys_id.sys_id_LS(format_df)
# format_df2 = sys_id.sys_id_LS_ex(format_df)
format_df4 = sys_id.sys_id_LS_max_non_kv(format_df)
format_df5 = sys_id.sys_id_LS_max_ub(format_df)
format_df6 = sys_id.sys_id_LS_non_d_alpha_ub(format_df)

#---------------------------
# 機体の状態方程式から固有振動数を解析する
#---------------------------

# anly_result = analyze.linearlize(format_df5)

#---------------------------
# データの取り出し
#---------------------------

data_size = len(format_df) # 合計のデータサイズを取得

#---------------------------
# 結果をプロット
#---------------------------

format_df4 = statistics.calc_RMSE(format_df4)
format_df5 = statistics.calc_RMSE(format_df5)
format_df6 = statistics.calc_RMSE(format_df6)

# df5_V_filter = format_df5.query('4.5 <= Va <= 5.5')

# format_df[['L_total','alpha_deg']].plot.line(x='alpha_deg', style=['o'])
# format_df[['D_total','alpha_deg']].plot.line(x='alpha_deg', style=['o'])
# format_df[['M','alpha_deg']].plot.line(x='alpha_deg', style=['o'])

# format_df[['L','alpha_deg']].plot.line(x='alpha_deg', style=['o'])
# format_df[['Time_DIFF','D']].plot.line(subplots='True', layout=(2,1))

# dfdf = format_df[format_df['Time_DIFF'] >= 0.03]
# print(dfdf['index'])

# format_df[['D_total','alpha_deg']].plot.line(x='alpha_deg', style=['o'])
# format_df[['M','alpha_deg']].plot.line(x='alpha_deg', style=['o'])

# format_df[['D','D_calc']].plot.line()
# format_df[['Ma','Ma_calc']].plot.line()

# format_df[['theta']].plot.line()
# format_df[['d_theta']].plot.line()
# format_df4[['D','D_calc']].plot.line()
# format_df4[['Ma','Ma_calc']].plot.line()
#
# format_df5[['L','L_calc']].plot.line()
# format_df5[['D','D_calc']].plot.line()
# format_df5[['Ma','Ma_calc']].plot.line()

# format_df6[['L','L_calc']].plot.line()
# format_df6[['D','D_calc']].plot.line()
# format_df6[['Ma','Ma_calc']].plot.line()

# fq = np.fft.fftfreq(data_size,d=0.02)
# format_df['fq'] = fq
# format_df[['alpha_fft','fq']].plot.line(x='fq')

# df5_V_filter[['CL_log','CL','alpha_deg']].plot.line(x='alpha_deg', style='o', title='CL alpha dot x')
# df5_V_filter[['CD_log','CD','alpha_deg']].plot.line(x='alpha_deg', style='o', title='CD alpha dot x')
# df5_V_filter[['Cm_log','Cm','alpha_deg']].plot.line(x='alpha_deg', style='o', title='Cm alpha dot x')

# format_df5[['CL_log','CL','Va']].plot.line(x='Va', style=['o','p'], title='CL')
# format_df5[['CD_log','CD','Va']].plot.line(x='Va', style=['o','p'], title='CD')
# format_df5[['Cm_log','Cm','Va']].plot.line(x='Va', style=['o','p'], title='Cm')

# format_df[['L_calc','alpha_deg']].plot.line(x='alpha_deg', style=['o'])
# format_df[['D_calc','alpha_deg']].plot.line(x='alpha_deg', style=['o'])
# format_df[['Ma_calc','alpha_deg']].plot.line(x='alpha_deg', style=['o'])

# format_df[['CL_kawano','Va']].plot.line(x='Va', style=['o'])
# format_df[['CD_kawano','Va']].plot.line(x='Va', style=['o'])
# format_df[['Cm_kawano','Va']].plot.line(x='Va', style=['o'])

# window = np.hamming(data_size)
# manual_T3 = window * manual_T3

# # 固有値の絶対値をとる．
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
# #
# #
# for j in borderline_list:
#     plt.axvline(x=j, color="black") # 実験データの境目で線を引く
#
# # plt.title('固有値散布図')
# plt.xlabel('Data Number')
# plt.ylabel('Eigenvalue')
# plt.show()
# #
# ax = fig.add_subplot(2,1,2)
#
# ax.plot(xxx,d_alpha)

# format_df5[['d_theta','d_theta_filt']].plot.line()

# d_theta = np.array(format_df5['d_theta'])
# d_theta_filt = np.array(format_df5['d_theta_filt'])
#
# plt.subplot(111)
# plt.plot(d_theta,label="Native data")
# plt.plot(d_theta_filt,label="Filtered data",linewidth="3")
# plt.legend()
#
# plt.xlabel('Data Number')
# plt.ylabel('Pitch Rate [rad/s]')
# plt.show()

# Va = np.array(format_df5['Va'])
# CL_log = np.array(format_df5['CL_log'])
# CL_4 = np.array(format_df4['CL']) # non kV
# CL_5 = np.array(format_df5['CL']) # max
# # CL_6 = np.array(format_df6['CL']) # non d_alpha
#
# plt.subplot(111)
# plt.scatter(Va,CL_log,label="Log data",linewidth="3")
# plt.scatter(Va,CL_4,label=r"None $k_LV_a$ model")
# plt.scatter(Va,CL_5,label="Postulate model")
# # plt.scatter(Va,CL_6,label=r"Model:No $\dot{\alpha}$")
# plt.legend()
#
# plt.xlabel(r'$V_a \mathrm{[m s^{-1}]}$')
# plt.ylabel(r'$C_L$')
# plt.show()

# f_up_pwm = np.array(format_df5['f_up_pwm'])
#
# # 高速フーリエ変換(FFT)
# F = np.fft.fft(f_up_pwm) #
#
# # FFTの複素数結果を絶対に変換
# F_abs = np.abs(F)
# # 振幅をもとの信号に揃える
# F_abs_am = F_abs / data_size * 2 # 交流成分はデータ数で割って2倍
# F_abs_amp = F_abs_am / (1/0.02)
#
# # 周波数軸のデータ作成
# fq = np.linspace(1, 1.0/0.02, data_size) # 周波数軸　linspace(開始,終了,分割数)
#
# fig = plt.figure()
#
# # 時間軸
# ax1 = fig.add_subplot(121)
# plt.xlabel('Data Number')
# plt.ylabel('PWM value')
# plt.plot(f_up_pwm,label="Sub rotor command") # ナイキスト定数まで表示
# plt.legend()
#
# # FFTのグラフ（周波数軸）
# ax2 = fig.add_subplot(122)
# plt.xlabel('Freqency[Hz]')
# plt.ylabel('Amplitude')
# plt.plot(fq[1:int(data_size/2)], F_abs_amp[1:int(data_size/2)],label="Sub rotor command") # ナイキスト定数まで表示
# plt.legend()
# plt.show()

L = np.array(format_df5['L'])
L_calc = np.array(format_df5['L_calc'])
D = np.array(format_df5['D'])
D_calc = np.array(format_df5['D_calc'])
Ma = np.array(format_df5['Ma'])
Ma_calc = np.array(format_df5['Ma_calc'])

plt.subplot(111)
plt.plot(Ma,label=r"$M_{a_{log}}$")
plt.plot(Ma_calc,label=r"$M_{a_{calc}}$")
plt.legend()

for j in borderline_list:
    plt.axvline(x=j, color="black",linestyle="--") # 実験データの境目で線を引く

plt.xlabel('Data Number')
plt.ylabel(r'Pitch moment$\mathrm{[N \cdot m]}$')
plt.show()

# CL = np.array(format_df5['CL'])
# CL_log = np.array(format_df5['CL_log'])
# CD = np.array(format_df5['CD'])
# CD_log = np.array(format_df5['CD_log'])
# Cm = np.array(format_df5['Cm'])
# Cm_log = np.array(format_df5['Cm_log'])
#
# plt.subplot(111)
# plt.plot(CD_log,label=r"$D_{log}$")
# plt.plot(CD,label=r"$D_{calc}$")
# plt.legend()
#
# for j in borderline_list:
#     plt.axvline(x=j, color="black",linestyle="--") # 実験データの境目で線を引く
#
# plt.xlabel('Data Number')
# plt.ylabel(r'Drag$\mathrm{[N]}$')
# plt.show()
