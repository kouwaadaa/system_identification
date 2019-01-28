# -*- coding: utf-8 -*-

#---------------------------
# Import library or package
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

#---------------------------
# matplotlibの諸設定
#---------------------------

# プロットデータを新しいウィンドウで表示する
get_ipython().run_line_magic('matplotlib', 'qt')

# 日本語フォントの設定
# 使用できるフォントを確認したいときは，次の行のコメントアウトを外して実行
# print([f.name for f in matplotlib.font_manager.fontManager.ttflist])
plt.rc('font', **{'family':'Gen Shin Gothic'})
# plt.rc('font', **{'family':'YuGothic'})
plt.rcParams['font.size'] = 20
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15 # default: 12

# プロットデータのサイズ設定
plt.rcParams["figure.figsize"] = [20, 12]

#---------------------------
# テスト空間
#---------------------------
format_df = pd.DataFrame()

format_df = file_read.file_read('../log_data/Book3.csv',17.52,19.14,-4.03,40/48,0,format_df)
#
format_df = file_read.file_read('../log_data/Book4.csv',11.97,13.30,-5.05,40/45,0,format_df)
format_df = file_read.file_read('../log_data/Book4.csv',18.66,21.08,-5.05,40/45,0,format_df)

format_df = file_read.file_read('../log_data/Book5.csv',12.45,13.66,-4.80,40/48,0,format_df)
format_df = file_read.file_read('../log_data/Book5.csv',16.07,17.03,-4.80,40/48,0,format_df)
format_df = file_read.file_read('../log_data/Book5.csv',18.95,22.88,-4.80,40/48,0,format_df)
#
# format_df = file_read.file_read('../log_data/Book8.csv',15.41,20.10,-2.00,40/47,0,format_df)
# format_df = file_read.file_read('../log_data/Book8.csv',21.46,23.07,-2.00,40/47,0,format_df)
# format_df = file_read.file_read('../log_data/Book8.csv',23.44,24.64,-2.00,40/47,0,format_df)
# format_df = file_read.file_read('../log_data/Book8.csv',25.28,27.38,-2.00,40/47,0,format_df)
#
# format_df = file_read.file_read('../log_data/Book9.csv',20.73,30.28,-2.647,40/48,0,format_df)
# format_df = file_read.file_read('../log_data/Book9.csv',98.05,104.1,-2.647,40/48,0,format_df)
# format_df = file_read.file_read('../log_data/Book9.csv',104.9,107.1,-2.647,40/48,0,format_df)
# format_df = file_read.file_read('../log_data/Book9.csv',107.7,109.7,-2.647,40/48,0,format_df)
#
# format_df = file_read.file_read('../log_data/Book11.csv',19.86,25.27,-1.467,40/48,0,format_df)
# format_df = file_read.file_read('../log_data/Book11.csv',26.43,29.83,-1.467,40/48,0,format_df)


format_df = format_df.reset_index()

d_theta = np.array(format_df['d_theta'])
data_size = len(format_df)
window = np.hamming(data_size)
d_theta_window = d_theta*window

fft_d_theta = np.fft.fft(d_theta)
fft_d_theta_window = np.fft.fft(d_theta_window)

fq_axis = np.linspace(0, 1.0/const.T_DIFF, data_size)

fq = np.fft.fftfreq(data_size, const.T_DIFF)
fft_d_theta[(fq >= 5)] = 0
fft_d_theta[(fq <= -5)] = 0

d_theta_filt = (np.fft.ifft(fft_d_theta)).real

a = np.fft.fft(d_theta_filt)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(d_theta, label='Pitch rate')
ax.plot(d_theta_filt, c='C1', label='Pitch rate filtered')

ax.legend()
ax.set_xlabel('Data Number')
ax.set_ylabel('[rad/s]')

#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# # ax.plot(fq_axis[1:int(data_size/2)], np.abs(a/(data_size/2))[1:int(data_size/2)], label='Pitch Rate freq filtered')
# # ax.plot(fq_axis[1:int(data_size/2)], np.abs(fft_d_theta/(data_size/2))[1:int(data_size/2)], label='Pitch Rate freq filtered')
# ax.plot(fq_axis, np.abs(a/(data_size/2)), label='Pitch Rate freq filtered')
# ax.plot(fq_axis, np.abs(fft_d_theta/(data_size/2)), label='Pitch Rate freq filtered')
#
# ax.legend()
# ax.set_xlabel('frequency[Hz]')
# ax.set_ylabel('amplitude')

#
# fq_axis = np.linspace(0, 1.0/const.T_DIFF, data_size)
#
# plt.plot(fq_axis[1:int(data_size/2)], fft_d_theta_amp[1:int(data_size/2)])
# plt.plot(fq_axis[1:int(data_size/2)], fft_d_theta_window_amp[1:int(data_size/2)])
#
# plt.show()
