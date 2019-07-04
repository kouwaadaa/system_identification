# -*- coding: utf-8 -*-
'''
author: ub
周波数領域に関する関数．
'''

import numpy as np
import pandas as pd
import pywt
from statistics import median
from scipy import interpolate

import const
import math_extention as matex
import matplotlib.pyplot as plt


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

    # plt.figure()
    # plt.subplot(111)
    # plt.plot(data,label="data",linewidth="3")
    # plt.plot(filt_data,label="filt_data")
    # # plt.legend(fontsize='25')
    # # plt.tick_params(labelsize='28')

    # # plt.xlabel(r'$V_a \mathrm{[m s^{-1}]}$',fontsize='36')
    # # plt.ylabel(r'$C_L^{\prime}$',fontsize='36')
    # plt.tight_layout()
    # plt.show()

    return filt_data

def wavelet_filter(data, N):
    '''
    ウェーブレット変換を用いて，時系列データを時間周波数領域でフィルタリング処理する．

    Parameters
    ----------
    data : array-likes
        処理を施したいファイル．
    N : int
        データサイズ
    wavelet : str

    Returns
    -------
    filt_data : array-like
        フィルタリング後のデータ．
    '''
    #ゼロパッディング(cannot run)
    # while True:
    #     count = 1
    #     if(N<=2**count):
    #         data = np.pad(data, [0,2**count-N], 'constant')
    #         break
    #     else:
    #         count += 1

    #マザーウェーブレットの設定
    wavelet = 'sym6'
    
    #変換
    coeff = pywt.wavedec(data,wavelet)

    #閾値設定
    uthresh = (1/0.6745) * median(abs(abs(coeff[-1])-median(abs(coeff[-1])))) * np.sqrt(2*np.log(N))

    #フィルタリング処理
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

    #逆変換
    filt_data = pywt.waverec(coeff,wavelet)
    # filt_data = np.delete(filt_data, np.s_[-(2**count-N):])
    
    #データ数の調整
    #よくわかっていない
    if(len(data)<len(filt_data)):
        filt_data = np.delete(filt_data, -1)
    
    # plt.figure()
    # plt.subplot(111)
    # plt.plot(data,label="data",linewidth="3")
    # plt.plot(filt_data,label="filt_data")
    # # plt.legend(fontsize='25')
    # # plt.tick_params(labelsize='28')

    # # plt.xlabel(r'$V_a \mathrm{[m s^{-1}]}$',fontsize='36')
    # # plt.ylabel(r'$C_L^{\prime}$',fontsize='36')
    # plt.tight_layout()
    # plt.show()

    return filt_data
