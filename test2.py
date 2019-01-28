import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython

#---------------------------
# matplotlibの諸設定
#---------------------------

# プロットデータを新しいウィンドウで表示する
get_ipython().run_line_magic('matplotlib', 'qt')

# 日本語フォントの設定
# 使用できるフォントを確認したいときは，次の行のコメントアウトを外して実行
# print([f.name for f in matplotlib.font_manager.fontManager.ttflist])
plt.rc('font', **{'family':'Gen Shin Gothic'})
# plt.rc('font', **{'family':'YuGothic'})
plt.rcParams['font.size'] = 20
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15 # default: 12

# プロットデータのサイズ設定
plt.rcParams["figure.figsize"] = [20, 12]


# 簡単な信号の作成
N = 20 # サンプル数
dt = 0.01 # サンプリング周期(sec):100ms =>サンプリング周波数100Hz
freq1 = 10 # 周波数(10Hz) =>正弦波の周期0.1sec
amp1 = 1 # 振幅
freq2 = 15 # 周波数(10Hz) =>正弦波の周期0.1sec
amp2 = 1 # 振幅

t = np.arange(0, N*dt, dt) # 時間軸
f = amp1 * np.sin(2*np.pi*freq1*t) + amp2 * np.sin(2*np.pi*freq2*t) # 信号
# # グラフ表示
# plt.xlabel('time(sec)', fontsize=14)
# plt.ylabel('signal', fontsize=14)
# plt.plot(t, f)

# 高速フーリエ変換(FFT)
F = np.fft.fft(f)
# FFTの複素数結果を絶対に変換
F_abs = np.abs(F)
# 振幅をもとの信号に揃える
F_abs_amp = F_abs / N * 2 # 交流成分はデータ数で割って2倍
F_abs_amp[0] = F_abs_amp[0] / 2 # 直流成分（今回は扱わないけど）は2倍不要

# 周波数軸のデータ作成
fq = np.linspace(0, 1.0/dt, N) # 周波数軸　linspace(開始,終了,分割数)

F_ifft = (np.fft.ifft(F)).real

F2 = np.fft.fft(F_ifft)
F2_abs = np.abs(F2)
F2_abs_amp = F2_abs / N * 2 # 交流成分はデータ数で割って2倍
F2_abs_amp[0] = F2_abs_amp[0] / 2 # 直流成分（今回は扱わないけど）は2倍不要

# グラフ表示（FFT解析結果）
plt.xlabel('freqency(Hz)')
plt.ylabel('amplitude')
plt.plot(fq, F_abs_amp)
plt.plot(fq, F_abs_amp)
