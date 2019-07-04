# -*- coding: utf-8 -*-
import frequency as freq
import numpy as np
import pywt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

original = np.loadtxt('../wavelet_test/Doppler1Norm.dat')
data = np.loadtxt('../wavelet_test/Doppler1Ns12.dat')

filt_data = freq.wavelet_filter(data, 1024)

RMSE = np.sqrt(mean_squared_error(original,filt_data))
print(f'RMSE:{RMSE:.10f}')