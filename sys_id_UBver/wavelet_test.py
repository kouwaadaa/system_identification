# -*- coding: utf-8 -*-
import frequency as freq
import numpy as np
import pywt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

original = np.loadtxt('./Doppler1Norm.dat')
data = np.loadtxt('./Doppler1Ns12.dat')

for wave in pywt.wavelist(kind='discrete'):
    filt_data = freq.wavelet_filter(data, 1024, wave)

    RMSE = np.sqrt(mean_squared_error(original,filt_data))
    print(wave)
    print(f'RMSE:{RMSE:.10f}')

# filt_data = freq.wavelet_filter(data, 1024, "sym6")

# RMSE = np.sqrt(mean_squared_error(original,filt_data))
# print(f'RMSE:{RMSE:.10f}')

    