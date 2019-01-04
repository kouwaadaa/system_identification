# -*- coding: utf-8 -*-
# author: ub
# 2018/12/14 Fri. 新座標系．

#---------------------------
# モジュールのインポートなど
#---------------------------

import numpy as np
from numpy import pi
import pandas as pd
import math_extention as matex

#---------------------------
# 機体データ（定数）
#  - 新座標系のもとでの値．
#---------------------------

# 慣性モーメント [kg・m^2]
I = np.array(
    [[ 0.2484,-0.0037,-0.0078],
     [-0.0037, 0.1668, 0.0005],
     [-0.0078, 0.0005, 0.3804]]
)
I_XX = I[0,0]
I_YY = I[1,1]
I_ZZ = I[2,2]

# 各距離 [m]
# to main, rear, front, pixhawk
# 重心からの距離 -> 新座標系からの距離に修正が必要
LEN_M = 0.042 # 重心〜メインロータ
LEN_F = 0.496 # 重心〜サブロータ前
LEN_R_X = 0.232 # 重心〜サブロータ横，X軸方向
LEN_R_Y = 0.503 # 重心〜サブロータ横，Y軸方向
LEN_P = 0.353 # 重心〜Pixhawk
MAC = 0.43081 # 平均空力翼弦

# 面積
S = 0.2087*2 + 0.1202 # 主翼2枚 + 機体

# 物理量
MASS = 5.7376 # Airframe weight
GRA = 9.80665 # Gravity acceleration
RHO = 1.205 # Air density

# サブロータ推力の上限
SUB_THRUST_MAX = 9.0
