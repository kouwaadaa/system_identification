#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 16:36:52 2018

@author: cs22Mac
"""

#import csv
#
#csv_file = open("./log_data/Book1.csv", "r", encoding="ASCII", errors="", newline="" )
##リスト形式
#f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
#for row in f:
#    #rowはList
#    #row[0]で必要な項目を取得することができる
#    print(row)

import numpy as np
from numpy import pi
import pymap3d as pm
import pandas as pd

csv_input = pd.read_csv(filepath_or_buffer="./log_data/Book1.csv", encoding="ASCII", sep=",") # sepはデフォルトで,なので不要
#値を二次元配列形式?で返却します。
#返却される型は、numpy.ndarray
#print(csv_input.values)

#行インデックス、カラムインデックスの順番で指定して項目の値を取得できます。
# Velocity
dot_x_position = np.array(csv_input.values[:,58])
dot_y_position = np.array(csv_input.values[:,59])
dot_z_position = np.array(csv_input.values[:,60])

# for文の練習
# data_size = len(csv_input)
#
# w_theta = []
# for i in range(data_size):
# 	w_theta.append(theta[i]*theta[i])
#
# print(w_theta)

# 行列の四則演算の練習
# 要素ごとに一気に計算する場合
# print(phi + theta)
# print((phi + theta)/2)
#
# print(phi**2 + phi*phi)

# Velocity
pixhawk_ground_velocity = []
pixhawk_airframe_system_velocity = []
airframe_system_velocity = [] # pixhawk -> center
airframe_system_wind_velocity = []

# Caliculate velocity
pixhawk_ground_velocity = np.sqrt(dot_x_position**2 + dot_y_position**2 + dot_z_position**2)
pixhawk_airframe_system_velocity = 
