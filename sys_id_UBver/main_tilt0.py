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
import fileread as fr
import thrust
import param_estimation as prme
import analyze
import statistics as stts
import plot

#---------------------------
# matplotlibの諸設定
#---------------------------

# プロットデータを新しいウィンドウで表示する．
# Spyderなどで実行する場合に必要．
#get_ipython().run_line_magic('matplotlib', 'qt')

# 日本語フォントの設定 font_managerのimportが必要．
# 使用できるフォントを確認したいときは，次の行のコメントアウトを外して実行．
# print([f.name for f in matplotlib.font_manager.fontManager.ttflist])

# 日本語フォントを使用したい場合は，このように指定する．
# plt.rc('font', **{'family':'YuGothic'})

# プロットデータのサイズ設定
# すべて変更されてしまうため注意．
# plt.rcParams['font.size'] = 28
# plt.rcParams['xtick.labelsize'] = 24
# plt.rcParams['ytick.labelsize'] = 24 # default: 12
# plt.rcParams["figure.figsize"] = [20, 12]

plt.rcParams['font.size'] = 28
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12 # default: 12
plt.rcParams["figure.figsize"] = [16, 10]

#---------------------------
# 推力効率係数の算出
#---------------------------

# サブ推力30%，35%
T_EFF_30 = thrust.calc_thrust_eff(1.0)[0]
T_EFF_35 = thrust.calc_thrust_eff(1.0)[1]

#---------------------------
# ログデータの読み込み
#---------------------------

# 読み込みデータ初期化
format_df = pd.DataFrame()

#---2017/12/27 徳島 定点ホバリング MCパラメータ変更------------------------------------------------------
# format_df = fr.file_read(300,'../log_data/Book3.csv',17.52,19.14,-4.03,T_EFF_35,1.264,0,format_df)
#
# #---2017/12/27 徳島 エレベータ（ピッチアップ） MCパラメータ変更------------------------------------------
format_df = fr.file_read(400,'../log_data/Book4.csv',11.97,13.30,-5.05,T_EFF_35,1.264,0,format_df)
format_df = fr.file_read(400,'../log_data/Book4.csv',18.66,21.08,-5.05,T_EFF_35,1.264,0,format_df)

# #---2017/12/27 徳島 エレベータ（ピッチダウン） MCパラメータ変更-------------------------------------------
format_df = fr.file_read(500,'../log_data/Book5.csv',12.45,13.66,-4.80,T_EFF_35,1.266,0,format_df)
format_df = fr.file_read(500,'../log_data/Book5.csv',16.07,17.03,-4.80,T_EFF_35,1.266,0,format_df)
format_df = fr.file_read(500,'../log_data/Book5.csv',18.95,22.88,-4.80,T_EFF_35,1.266,0,format_df)

# #---2018/01/14 徳島 ピッチ運動-----------------------------------------------------------
format_df = fr.file_read(800,'../log_data/Book8.csv',15.41,20.10,-2.00,T_EFF_30,1.275,0,format_df)
format_df = fr.file_read(800,'../log_data/Book8.csv',21.46,23.07,-2.00,T_EFF_30,1.275,0,format_df)
format_df = fr.file_read(800,'../log_data/Book8.csv',23.44,24.64,-2.00,T_EFF_30,1.275,0,format_df)
format_df = fr.file_read(800,'../log_data/Book8.csv',25.28,27.38,-2.00,T_EFF_30,1.275,0,format_df)

# #---2018/01/26 神戸 前進＆エレベータ制御-------------------------------------------------
format_df = fr.file_read(900,'../log_data/Book9.csv',20.73,30.28,-2.647,T_EFF_30,1.251,0,format_df)
format_df = fr.file_read(900,'../log_data/Book9.csv',98.05,104.1,-2.647,T_EFF_30,1.251,0,format_df)
format_df = fr.file_read(900,'../log_data/Book9.csv',104.9,107.1,-2.647,T_EFF_30,1.251,0,format_df)
format_df = fr.file_read(900,'../log_data/Book9.csv',107.7,109.7,-2.647,T_EFF_30,1.251,0,format_df)
#
# #---2018/01/26 神戸 前進-----------------------------------------------------------------
format_df = fr.file_read(1100,'../log_data/Book11.csv',19.86,25.27,-1.467,T_EFF_30,1.268,0,format_df)
format_df = fr.file_read(1100,'../log_data/Book11.csv',26.43,29.83,-1.467,T_EFF_30,1.268,0,format_df)
# #---------------------------------------------------------

#---------------------------
# データの整理
#---------------------------

# indexの振り直し
format_df = format_df.reset_index()

#---------------------------
# パラメータ推定の結果を計算し，取得
#---------------------------

df_non_dalpha = prme.LS_non_dalpha(format_df)
df_with_dalpha = prme.LS_with_dalpha(format_df)
df_non_kv = prme.LS_non_kv(format_df)

df_ex_non_dalpha = prme.LS_ex_non_dalpha(format_df)
df_ex_with_dalpha = prme.LS_ex_with_dalpha(format_df)
df_ex_non_kv = prme.LS_ex_non_kv(format_df)

#---------------------------
# 機体の状態方程式から固有振動数を解析する
#---------------------------

# anly_result = analyze.linearlize(df_with_dalpha)
anly_result = analyze.linearlize_non_d_alpha(df_non_dalpha)

#---------------------------
# 統計データ算出
#---------------------------

# stts.calc_RMSE(df_non_dalpha)
# stts.calc_RMSE(df_with_dalpha)
# stts.calc_RMSE(df_non_kv)
stts.calc_RMSE(df_ex_non_dalpha)
# stts.calc_RMSE(df_ex_with_dalpha)
stts.calc_RMSE(df_ex_non_kv)

#---------------------------
# データ取り出し作業
#---------------------------

# 合計のデータサイズを取得
data_size = len(format_df)

#---------------------------
# 結果をプロット
#---------------------------

'''
・時間軸に対してどんな動きのデータかざっくりみたいとき
ex) ピッチ角速度
df[['d_theta']].plot.line()

cf) 複数データを同時に見る
df[['d_theta','d_phi']].plot.line()


・横軸を指定してデータをみたいとき
ex) Va横軸で迎角を見る
df[['Va','alpha']].plot.line(x='Va')

cf) この場合，線でつながれて見にくいので
df[['Va','alpha']].plot.line(x='Va',style='o')
として散布図にする．


．ある条件でデータを切り出してプロットする
ex) Vaが5以上のデータのみ
dff = df[df['Va'] >= 5]
としてdffをプロット．


Jupyterで試してみて，記録に残しておくか，
plot.pyに関数として残して再利用しやすくすると良い．
'''

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
