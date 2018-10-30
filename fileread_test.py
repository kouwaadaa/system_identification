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

import pandas as pd

csv_input = pd.read_csv(filepath_or_buffer="./log_data/Book1.csv", encoding="ASCII", sep=",") # sepはデフォルトで,なので不要
#値を二次元配列形式?で返却します。
#返却される型は、numpy.ndarray
#print(csv_input.values)

#行インデックス、カラムインデックスの順番で指定して項目の値を取得できます。
theta = csv_input.values[:, 1]

# for文の練習
data_size = len(csv_input)

w_theta = []
for i in range(data_size):
	w_theta.append(theta[i]*theta[i])

print(w_theta)
