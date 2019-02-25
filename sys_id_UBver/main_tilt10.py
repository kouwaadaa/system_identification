# -*- coding: utf-8 -*-
# author: ub
# 2019/02/25 Mon. 新座標系．

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
import sys_id
import analyze
import statistics

#---------------------------
# matplotlibの諸設定
#---------------------------

# プロットデータを新しいウィンドウで表示する
get_ipython().run_line_magic('matplotlib', 'qt')

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

format_df = pd.DataFrame()
borderline_list = list()

format_df,size = file_read.file_read('../log_data/Book3.csv',17.52,19.14,-4.03,40/48,0,format_df)
borderline_list.append(size)
# format_df = file_read.file_read('../log_data/Book4.csv',11.97,13.30,-5.05,40/45,0,format_df)
format_df,size = file_read.file_read('../log_data/Book4.csv',18.66,21.08,-5.05,40/45,0,format_df)
borderline_list.append(size+borderline_list[-1])

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

format_df4 = sys_id.sys_id_LS_max_non_kv(format_df)

#---------------------------
# 機体の状態方程式から固有振動数を解析する
#---------------------------

anly_result = analyze.linearlize_non_d_alpha(format_df6)

#---------------------------
# データの取り出し
#---------------------------

data_size = len(format_df) # 合計のデータサイズを取得

#---------------------------
# 結果をプロット
#---------------------------
