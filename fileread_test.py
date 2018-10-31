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
import pandas as pd
import math_extention

read_log_data = pd.read_csv(filepath_or_buffer="./log_data/Book1.csv", encoding="ASCII", sep=",", header=None) # sepはデフォルトで,なので不要
#値を二次元配列形式?で返却します。
#返却される型は、numpy.ndarray

print(read_log_data.duplicated(subset=390))
print(read_log_data.duplicated(subset=390).value_counts())

read_log_data = read_log_data.drop_duplicates(subset=390)

print(read_log_data.duplicated(subset=390))
print(read_log_data.duplicated(subset=390).value_counts())

# read_log_data = read_log_data.T
# read_log_data = read_log_data.drop_duplicates()

#行インデックス、カラムインデックスの順番で指定して項目の値を取得できます。

phi = np.array(read_log_data.values[:,0])
theta = np.array(read_log_data.values[:,1])
psi = np.array(read_log_data.values[:,2])

# # Velocity
dot_x_position = np.array(read_log_data.values[:,58])
dot_y_position = np.array(read_log_data.values[:,59])
dot_z_position = np.array(read_log_data.values[:,60])
#
# dot_x_position = dot_x_position[:,None]
# dot_y_position = dot_y_position[:,None]
# dot_z_position = dot_z_position[:,None]
#
# dot_xyz_position = np.concatenate([dot_x_position,dot_y_position,dot_z_position], axis=1)

# Velocity
# ned_velocity = []
#
# for i in range(data_size):
# 	ned_velocity.append(math_extention.bc2ned(phi[i],theta[i],psi[i],dot_x_position[i],dot_y_position[i],dot_z_position[i]))
#
# ned_velocity = np.array(ned_velocity)
#
# print(ned_velocity)
