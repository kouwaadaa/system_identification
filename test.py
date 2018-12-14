# -*- coding: utf-8 -*-

#---------------------------
# Import library or package
#---------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import get_ipython

import math_extention as matex

from numpy import pi

#---------------------------
# matplotlibの諸設定
#---------------------------

# プロットデータを新しいウィンドウで表示する
get_ipython().run_line_magic('matplotlib', 'qt')

# 日本語フォントの設定
# 使用できるフォントを確認したいときは，次の行のコメントアウトを外して実行
# print([f.name for f in matplotlib.font_manager.fontManager.ttflist])
# plt.rc('font', **{'family':'Gen Shin Gothic'})
plt.rc('font', **{'family':'YuGothic'})
plt.rcParams['font.size'] = 20
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15 # default: 12

# プロットデータのサイズ設定
plt.rcParams["figure.figsize"] = [20, 12]

#---------------------------
# テスト空間
#---------------------------
read_log_data = pd.read_csv(
    filepath_or_buffer='./log_data/Book3.csv',
    encoding='ASCII',
    sep=',',
    header=None
)

df = read_log_data[read_log_data.duplicated(subset=390)]

# # 重複データの削除
# read_log_data = read_log_data.drop_duplicates(subset=390)

# # 時間データを[秒]に変換
# read_log_data['Time_ST'] = read_log_data.at[0,390]
# read_log_data['Time_Conv'] = (read_log_data[390] - read_log_data['Time_ST'])/1000000

# # 実験時間のみ切り取り
# read_log_data = read_log_data.query(
#     '17.52 <= Time_Conv <= 19.14'
# )
#
# # Aファイルに依存する値（風速，推力効率，ティルト角）
# V_W = -4.03
# THRUST_EF = 40/48
# GAMMA = 0

# #------
# # 計算
# #------
#
# d_theta = np.array(read_log_data.values[:,4])
#
# # 時間
# time = np.array(read_log_data.values[:,392])
#
# # 時間偏差（サンプリング間隔）
# DT = 0.02
#
# # データサイズの取得（列方向）
# data_size = len(read_log_data)
#
# # 高速フーリエ変換（FFT）
# F_d_theta = np.fft.fft(d_theta)
#
# # FFTの複素数結果を絶対変換
# F_d_theta_abs = np.abs(F_d_theta)
#
# # 振れ幅をもとの信号に揃える
# F_d_theta_abs = F_d_theta_abs / data_size * 2 # 交流成分
# F_d_theta_abs[0] = F_d_theta_abs[0] / 2 # 直流成分
#
# # 周波数軸のデータ作成
# fq = np.linspace(0, 1.0/DT, data_size) # 周波数軸　linspace(開始,終了,分割数)
#
# # 振幅強度でフィルタリング処理
# F2_d_theta = np.copy(F_d_theta) # FFT結果コピー
# ac = 0.03 # 振幅強度の閾値
# F2_d_theta[(F_d_theta_abs < ac)] = 0 # 振幅が閾値未満はゼロにする（ノイズ除去）
#
# # 振幅でフィルタリング処理した結果の確認
# # FFTの複素数結果を絶対値に変換
# F2_d_theta_abs = np.abs(F2_d_theta)
# # 振幅をもとの信号に揃える
# F2_d_theta_abs = F2_d_theta_abs / data_size * 2 # 交流成分はデータ数で割って2倍
# F2_d_theta_abs[0] = F2_d_theta_abs[0] / 2 # 直流成分（今回は扱わないけど）は2倍不要
#
# # 逆変換IFFT
# Fi_d_theta = np.fft.ifft(F_d_theta) # IFFT
# Fi_d_theta_real = Fi_d_theta.real # 実数部
#
# F2i_d_theta = np.fft.ifft(F2_d_theta) # IFFT
# F2i_d_theta_real = F2i_d_theta.real # 実数部

# #-----
# # プロット
# #-----
#
# #-------------------
#
# plt.figure()
# # 余白を設定
# plt.subplots_adjust(wspace=0.4, hspace=0.6)
#
# plt.plot(time, F2i_d_theta_real)
# plt.plot(time, d_theta)
#
# #-------------------
#
# plt.figure()
# # 余白を設定
# plt.subplots_adjust(wspace=0.4, hspace=0.6)
#
# plt.xlabel('frequency[Hz]')
# plt.ylabel('d_theta[rad]')
# plt.plot(fq, F_d_theta_abs)
#
# #-------------------
#
# plt.figure()
# # 余白を設定
# plt.subplots_adjust(wspace=0.4, hspace=0.6)
#
# plt.xlabel('freqency(Hz)')
# plt.ylabel('amplitude')
# plt.plot(fq, F2_d_theta_abs, c='orange')
#
# #-------------------
#
# plt.show()
