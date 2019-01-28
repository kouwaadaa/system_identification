# -*- coding: utf-8 -*-
# author: ub

'''
周波数領域に関する関数．
'''

import numpy as np
import pandas as pd

import const
import math_extention as matex


def fourier_filter(data, dt, N, fc):
    '''
    フーリエ変換を用いて，時系列データを周波数領域でフィルタリング処理する．

    Parameters
    ----------
    data : array-like
        処理を施したいファイル．
    dt : float64
        サンプリング間隔[s]
    N : int
        分割数（データサイズ）
    fc : float64
        カットオフ周波数[Hz]

    Returns
    -------
    filt_data : array-like
        フィルタリング後のデータ．
    '''

    # 周波数
    fq = np.fft.fftfreq(N, dt)

    # FFT
    fft = np.fft.fft(data)

    # フィルタリング，エイリアシングした部分は残す．
    fft[(fq >= fc)] = 0
    fft[(fq <= -fc)] = 0

    # IFFT 実部だけ取り出す．
    filt_data = (np.fft.ifft(fft)).real

    return filt_data
