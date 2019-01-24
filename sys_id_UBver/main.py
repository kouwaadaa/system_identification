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
# from scipy import signal

import const
import math_extention as matex
import file_read
import calc
import calc_ex
import calc_ex_max
import calc_kawano
import analyze

#---------------------------
# matplotlibの諸設定
#---------------------------

# プロットデータを新しいウィンドウで表示する
get_ipython().run_line_magic('matplotlib', 'qt')

# 日本語フォントの設定
# 使用できるフォントを確認したいときは，次の行のコメントアウトを外して実行
# print([f.name for f in matplotlib.font_manager.fontManager.ttflist])

# for NotePC
plt.rc('font', **{'family':'Gen Shin Gothic'})

# for DeskPC
# plt.rc('font', **{'family':'YuGothic'})

plt.rcParams['font.size'] = 20
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15 # default: 12

# プロットデータのサイズ設定
plt.rcParams["figure.figsize"] = [20, 12]

#---------------------------
# ログデータの読み込み
#---------------------------

init_df = pd.DataFrame()

format_log_data = file_read.file_read('../log_data/Book3.csv',17.52,19.14,-4.03,40/48,0,init_df)

format_log_data = file_read.file_read('../log_data/Book4.csv',11.97,13.30,-5.05,40/45,0,format_log_data)
format_log_data = file_read.file_read('../log_data/Book4.csv',18.66,21.08,-5.05,40/45,0,format_log_data)

format_log_data = file_read.file_read('../log_data/Book5.csv',12.45,13.66,-4.80,40/48,0,format_log_data)
format_log_data = file_read.file_read('../log_data/Book5.csv',16.07,17.03,-4.80,40/48,0,format_log_data)
format_log_data = file_read.file_read('../log_data/Book5.csv',18.95,22.88,-4.80,40/48,0,format_log_data)

format_log_data = file_read.file_read('../log_data/Book8.csv',15.41,20.10,-2.00,40/47,0,format_log_data)
format_log_data = file_read.file_read('../log_data/Book8.csv',21.46,23.07,-2.00,40/47,0,format_log_data)
format_log_data = file_read.file_read('../log_data/Book8.csv',23.44,24.64,-2.00,40/47,0,format_log_data)
format_log_data = file_read.file_read('../log_data/Book8.csv',25.28,27.38,-2.00,40/47,0,format_log_data)

format_log_data = file_read.file_read('../log_data/Book9.csv',20.73,30.28,-2.647,40/48,0,format_log_data)
format_log_data = file_read.file_read('../log_data/Book9.csv',98.05,104.1,-2.647,40/48,0,format_log_data)
format_log_data = file_read.file_read('../log_data/Book9.csv',104.9,107.1,-2.647,40/48,0,format_log_data)
format_log_data = file_read.file_read('../log_data/Book9.csv',107.7,109.7,-2.647,40/48,0,format_log_data)

format_log_data = file_read.file_read('../log_data/Book11.csv',19.86,25.27,-1.467,40/48,0,format_log_data)
format_log_data = file_read.file_read('../log_data/Book11.csv',26.43,29.83,-1.467,40/48,0,format_log_data)

#---------------------------
# パラメータ推定の結果を計算し，取得
#---------------------------

sys_id_result = calc.sys_id_LS(format_log_data)
# sys_id_result = calc_ex.sys_id_LS_ex(format_log_data)
# sys_id_result = calc_ex_max.sys_id_LS_ex_max(format_log_data)
# sys_id_result = calc_kawano.sys_id_LS_kawano(format_log_data)

#---------------------------
# 推定結果の値もデータ群に格納する
#---------------------------

# d_alphaを含まない場合
if sys_id_result[0].shape[1] == 5:
    format_log_data['CL_0'] = sys_id_result[0][:,0]
    format_log_data['CL_alpha'] = sys_id_result[0][:,1]
    format_log_data['CL_q'] = sys_id_result[0][:,2]
    format_log_data['CL_delta_e'] = sys_id_result[0][:,3]
    format_log_data['k_L'] = sys_id_result[0][:,4]

    format_log_data['CD_0'] = sys_id_result[1][:,0]
    format_log_data['kappa'] = sys_id_result[1][:,1]
    format_log_data['k_D'] = sys_id_result[1][:,2]

    format_log_data['Cm_0'] = sys_id_result[2][:,0]
    format_log_data['Cm_alpha'] = sys_id_result[2][:,1]
    format_log_data['Cm_q'] = sys_id_result[2][:,2]
    format_log_data['Cm_delta_e'] = sys_id_result[2][:,3]
    format_log_data['k_m'] = sys_id_result[2][:,4]

    format_log_data['CL'] = sys_id_result[3][:,0]
    format_log_data['CD'] = sys_id_result[3][:,1]
    format_log_data['Cm'] = sys_id_result[3][:,2]
    format_log_data['L_calc'] = sys_id_result[3][:,3]
    format_log_data['D_calc'] = sys_id_result[3][:,4]
    format_log_data['Ma_calc'] = sys_id_result[3][:,5]

# d_alphaを含む場合
elif sys_id_result[0].shape[1] == 6:
    format_log_data['CL_0'] = sys_id_result[0][:,0]
    format_log_data['CL_alpha'] = sys_id_result[0][:,1]
    format_log_data['CL_d_alpha'] = sys_id_result[0][:,2]
    format_log_data['CL_q'] = sys_id_result[0][:,3]
    format_log_data['CL_delta_e'] = sys_id_result[0][:,4]
    format_log_data['k_L'] = sys_id_result[0][:,5]

    format_log_data['CD_0'] = sys_id_result[1][:,0]
    format_log_data['kappa'] = sys_id_result[1][:,1]
    format_log_data['k_D'] = sys_id_result[1][:,2]

    format_log_data['Cm_0'] = sys_id_result[2][:,0]
    format_log_data['Cm_alpha'] = sys_id_result[2][:,1]
    format_log_data['Cm_d_alpha'] = sys_id_result[2][:,2]
    format_log_data['Cm_q'] = sys_id_result[2][:,3]
    format_log_data['Cm_delta_e'] = sys_id_result[2][:,4]
    format_log_data['k_m'] = sys_id_result[2][:,5]

    format_log_data['CL'] = sys_id_result[3][:,0]
    format_log_data['CD'] = sys_id_result[3][:,1]
    format_log_data['Cm'] = sys_id_result[3][:,2]
    format_log_data['L_calc'] = sys_id_result[3][:,3]
    format_log_data['D_calc'] = sys_id_result[3][:,4]
    format_log_data['Ma_calc'] = sys_id_result[3][:,5]

# indexの振り直し
# ここで新たに"index"という列が生成されるが，
# これを残しておけばログデータごとのプロットがしやすい．
format_log_data = format_log_data.reset_index()

#---------------------------
# 機体の状態方程式から固有振動数を解析する
#---------------------------

# anly_result = analyze.linearlize(format_log_data)

#---------------------------
# データの取り出し
#---------------------------

data_size = len(format_log_data) # 合計のデータサイズを取得

#---------------------------
# 結果をプロット
#---------------------------

format_log_data[['L','L_calc']].plot()

format_log_data.plot.line(x='Va', y=['Ma','Ma_calc'], style=['o','o'])

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

# plt.subplot(111)
# plt.scatter(xxx,y)
# plt.scatter(xxx,yy)
# plt.scatter(xxx,yyy)
# plt.scatter(xxx,yyyy)
#
#
# for j in range(FILE_NUM-1):
#     plt.axvline(x=borderline_data_num[j], color="black") # 実験データの境目で線を引く
#
# plt.title('固有値散布図')
# plt.xlabel('データ番号')
# plt.ylabel('固有値')
#
# # ax = fig.add_subplot(2,1,2)
# #
# # ax.plot(xxx,d_alpha)
#

#---------------------------
# フーリエ変換
#---------------------------

# # 周波数軸のデータ作成
# fq = np.fft.fftfreq(data_size,d=0.02)
#
# # FFT
# F_d_theta = matex.fft_set_amp(d_theta,0.02,data_size)
#
# F_manual_T1 = matex.fft_set_amp(manual_T1,0.02,data_size)
# F_manual_T2 = matex.fft_set_amp(manual_T2,0.02,data_size)
# F_manual_T3 = matex.fft_set_amp(manual_T3,0.02,data_size)
# F_manual_T4 = matex.fft_set_amp(manual_T4,0.02,data_size)
# F_manual_T5 = matex.fft_set_amp(manual_T5,0.02,data_size)
# F_manual_T6 = matex.fft_set_amp(manual_T6,0.02,data_size)
# F_manual_elevon_r = matex.fft_set_amp(manual_elevon_r,0.02,data_size)
# F_manual_elevon_l = matex.fft_set_amp(manual_elevon_l,0.02,data_size)
# F_manual_pitch = matex.fft_set_amp(manual_pitch,0.02,data_size)
# F_manual_thrust = matex.fft_set_amp(manual_thrust,0.02,data_size)
# F_manual_tilt = matex.fft_set_amp(manual_tilt,0.02,data_size)
#
# # ３次ローパスフィルタをかける
# for i in range(3):
#     theta_filt = matex.lp_filter(0.03,0.02,data_size,theta)
