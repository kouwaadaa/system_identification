# -*- coding: utf-8 -*-
'''
author: ub
ログデータの閲覧用．
'''

#---------------------------
# モジュールのインポートなど
#---------------------------

import numpy as np
from numpy import pi
import pandas as pd
import matplotlib.pyplot as plt
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

plt.rcParams['font.size'] = 28
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24 # default: 12
#
# プロットデータのサイズ設定
plt.rcParams["figure.figsize"] = [20, 12]

#---------------------------
# ファイルの読み込み
#---------------------------

# CSVファイルの読み込み
df = pd.read_csv(
    filepath_or_buffer="../log_data/Book8.csv",
    encoding='ASCII',
    sep=',',
    header=0,
    usecols=['ATT_Roll',
             'ATT_Pitch',
             # 'ATT_Yaw',
             # 'ATT_RollRate',
             'ATT_PitchRate',
             # 'ATT_YawRate',
             # 'LPOS_X',
             # 'LPOS_Y',
             # 'LPOS_Z',
             'LPOS_VX',
             'LPOS_VY',
             'LPOS_VZ',
             'GPS_Alt',
             # 'OUT0_Out0',
             # 'OUT0_Out1',
             # 'OUT0_Out2',
             # 'OUT0_Out3',
             # 'OUT0_Out4',
             # 'OUT0_Out5',
             # 'OUT1_Out0',
             # 'OUT1_Out1',
             # 'AIRS_TrueSpeed',
             # 'MAN_pitch',
             # 'MAN_thrust',
             # 'VTOL_Tilt',
             'TIME_StartTime'
             ]
)

# 空白行を削除
df = df.dropna(how='all')
df = df.reset_index(drop=True)

# 時間データを[秒]に変換
df['Time_ST'] = df.at[0,'TIME_StartTime']
df['Time_sec'] = (df['TIME_StartTime'] - df['Time_ST'])/1000000

df[['GPS_Alt','Time_sec']].plot.line(x='Time_sec')
