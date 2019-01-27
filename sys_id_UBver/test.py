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
format_df = file_read.file_read('../log_data/Book5.csv',18.95,22.88,-4.80,40/48,0,format_df)

# format_df[['d_theta']].plot.line()

d_theta = np.array(format_df['d_theta'])
data_size = len(format_df)
window = np.hamming(data_size)
d_theta_window = d_theta*window

fft_d_theta = np.fft.fft(d_theta)
fft_d_theta_window = np.fft.fft(d_theta_window)

fft_d_theta_amp = np.abs(fft_d_theta/(data_size/2))
fft_d_theta_window_amp = np.abs(fft_d_theta_window/(data_size/2))

fq_axis = np.linspace(0, 1.0/const.T_DIFF, data_size)

plt.plot(fq_axis[1:int(data_size/2)], fft_d_theta_amp[1:int(data_size/2)])
plt.plot(fq_axis[1:int(data_size/2)], fft_d_theta_window_amp[1:int(data_size/2)])

plt.show()
