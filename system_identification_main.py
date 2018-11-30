# -*- coding: utf-8 -*-

#---------------------------
# モジュールのインポートなど
#---------------------------

import numpy as np
from numpy import pi
import pandas as pd

import matplotlib.pyplot as plt
# import matplotlib.font_manager
from IPython import get_ipython

import math_extention as matex

#---------------------------
# matplotlibの諸設定
#---------------------------

# プロットデータを新しいウィンドウで表示する
get_ipython().run_line_magic('matplotlib', 'qt')

# 日本語フォントの設定
# 使用できるフォントを確認したいときは，次の行のコメントアウトを外して実行
# print([f.name for f in matplotlib.font_manager.fontManager.ttflist])
plt.rc('font', **{'family':'YuGothic'})
plt.rcParams['font.size'] = 20
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15 # default: 12

# プロットデータのサイズ設定
plt.rcParams["figure.figsize"] = [20, 12]

#---------------------------
# 機体データ（定数）
#---------------------------

# 慣性モーメント [kg/m^2]
I = np.array(
    [[ 0.2484,-0.0037,-0.0078],
     [-0.0037, 0.1668, 0.0005],
     [-0.0078, 0.0005, 0.3804]]
)
I_XX = I[0,0] # X軸
I_YY = I[1,1] # Y軸
I_ZZ = I[2,2] # Z軸

# 各距離 [m]
LEN_M = 0.042 # 重心〜メインロータ
LEN_F = 0.496 # 重心〜サブロータ前
LEN_S_X = 0.232 # 重心〜サブロータ横，X軸方向
LEN_S_Y = 0.503 # 重心〜サブロータ横，Y軸方向
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

#---------------------------
# ログデータの読み込み
#---------------------------

# 読み込むファイル数
FILE_NUM = 6
for file_number in range(FILE_NUM):

    #---------------------------
    # CSVデータを読み込んで整理する
    #---------------------------

    if file_number == 0:
        read_log_data = pd.read_csv(
            filepath_or_buffer='./log_data/Book3.csv',
            encoding='ASCII',
            sep=',',
            header=None
        )

        # 重複データの削除
        read_log_data = read_log_data.drop_duplicates(subset=390)

        # 時間データを[秒]に変換
        read_log_data['Time_ST'] = read_log_data.at[0,390]
        read_log_data['Time_Conv'] = (read_log_data[390] - read_log_data['Time_ST'])/1000000

        # 実験時間のみ切り取り
        read_log_data = read_log_data.query(
            '17.52 <= Time_Conv <= 19.14'
        )

        # Aファイルに依存する値（風速，推力効率，ティルト角）
        V_W = -4.03
        THRUST_EF = 40/48
        GAMMA = 0

    elif file_number == 1:
        read_log_data = pd.read_csv(
            filepath_or_buffer='./log_data/Book4.csv',
            encoding='ASCII',
            sep=',',
            header=None
        )

        # 重複データの削除
        read_log_data = read_log_data.drop_duplicates(subset=390)

        # 時間データを[秒]に変換
        read_log_data['Time_ST'] = read_log_data.at[0,390]
        read_log_data['Time_Conv'] = (read_log_data[390] - read_log_data['Time_ST'])/1000000

        # 実験時間のみ切り取り
        read_log_data = read_log_data.query(
            '11.97 <= Time_Conv <= 13.30 \
            | 18.66 <= Time_Conv <= 21.08'
        )

        # ファイルに依存する値（風速，推力効率，ティルト角）
        V_W = -5.05
        THRUST_EF = 40/45
        GAMMA = 0

    elif file_number == 2:
        read_log_data = pd.read_csv(
            filepath_or_buffer='./log_data/Book5.csv',
            encoding='ASCII',
            sep=',',
            header=None
        )

        # 重複データの削除
        read_log_data = read_log_data.drop_duplicates(subset=390)

        # 時間データを[秒]に変換
        read_log_data['Time_ST'] = read_log_data.at[0,390]
        read_log_data['Time_Conv'] = (read_log_data[390] - read_log_data['Time_ST'])/1000000

        # 実験時間のみ切り取り
        read_log_data = read_log_data.query(
            '12.45 <= Time_Conv <= 13.66 \
            | 16.07 <= Time_Conv <= 17.03 \
            | 18.95 <= Time_Conv <= 22.88'
        )

        # ファイルに依存する値（風速，推力効率，ティルト角）
        V_W = -4.80
        THRUST_EF = 40/48
        GAMMA = 0

    elif file_number == 3:
        read_log_data = pd.read_csv(
            filepath_or_buffer='./log_data/Book8.csv',
            encoding='ASCII',
            sep=',',
            header=None
        )

        # 重複データの削除
        read_log_data = read_log_data.drop_duplicates(subset=390)

        # 時間データを[秒]に変換
        read_log_data['Time_ST'] = read_log_data.at[0,390]
        read_log_data['Time_Conv'] = (read_log_data[390] - read_log_data['Time_ST'])/1000000

        # 実験時間のみ切り取り
        read_log_data = read_log_data.query(
            '15.41 <= Time_Conv <= 20.10 \
            | 21.46 <= Time_Conv <= 23.07 \
            | 23.44 <= Time_Conv <= 24.64 \
            | 25.28 <= Time_Conv <= 27.38'
        )

        # ファイルに依存する値（風速，推力効率，ティルト角）
        V_W = -2.0
        THRUST_EF = 40/47
        GAMMA = 0

    elif file_number == 4:
        read_log_data = pd.read_csv(
            filepath_or_buffer='./log_data/Book9.csv',
            encoding='ASCII',
            sep=',',
            header=None
        )

        # 重複データの削除
        read_log_data = read_log_data.drop_duplicates(subset=390)

        # 時間データを[秒]に変換
        read_log_data['Time_ST'] = read_log_data.at[0,390]
        read_log_data['Time_Conv'] = (read_log_data[390] - read_log_data['Time_ST'])/1000000

        # 実験時間のみ切り取り
        read_log_data = read_log_data.query(
            '20.73 <= Time_Conv <= 30.28 \
            | 98.05 <= Time_Conv <= 104.1 \
            | 104.9 <= Time_Conv <= 107.1 \
            | 107.7 <= Time_Conv <= 109.7'
        )

        # ファイルに依存する値（風速，推力効率，ティルト角）
        V_W = -2.647
        THRUST_EF = 40/48
        GAMMA = 0

    elif file_number == 5:
        read_log_data = pd.read_csv(
            filepath_or_buffer='./log_data/Book11.csv',
            encoding='ASCII',
            sep=',',
            header=None
        )

        # 重複データの削除
        read_log_data = read_log_data.drop_duplicates(subset=390)

        # 時間データを[秒]に変換
        read_log_data['Time_ST'] = read_log_data.at[0,390]
        read_log_data['Time_Conv'] = (read_log_data[390] - read_log_data['Time_ST'])/1000000

        # 実験時間のみ切り取り
        read_log_data = read_log_data.query(
            '19.86 <= Time_Conv <= 25.27 \
            | 26.43 <= Time_Conv <= 29.83'
        )

        # ファイルに依存する値（風速，推力効率，ティルト角）
        V_W = -1.467
        THRUST_EF = 40/48
        GAMMA = 0

    #---------------------------
    # 各データを取り出す
    #---------------------------
    # 角度
    phi = np.array(read_log_data.values[:,0])
    theta = np.array(read_log_data.values[:,1])
    psi = np.array(read_log_data.values[:,2])

    # 角速度
    d_phi = np.array(read_log_data.values[:,3])
    d_theta = np.array(read_log_data.values[:,4])
    d_psi = np.array(read_log_data.values[:,5])

    # 位置
    position_x = np.array(read_log_data.values[:,53])
    position_y = np.array(read_log_data.values[:,54])
    position_z = np.array(read_log_data.values[:,55])

    # 速度
    d_position_x = np.array(read_log_data.values[:,58])
    d_position_y = np.array(read_log_data.values[:,59])
    d_position_z = np.array(read_log_data.values[:,60])

    # GPS高度
    gps_altitude = np.array(read_log_data.values[:,79])

    # ピトー管から得た対気速度
    measurement_airspeed = np.array(read_log_data.values[:,133])

    # ロータ指令値
    m_up_pwm = np.array(read_log_data.values[:,116]) # T1
    m_down_pwm = np.array(read_log_data.values[:,117]) # T2
    s_r_pwm = np.array(read_log_data.values[:,118]) # T3
    s_l_pwm = np.array(read_log_data.values[:,119]) # T4
    f_up_pwm = np.array(read_log_data.values[:,120]) # T5
    f_down_pwm = np.array(read_log_data.values[:,121]) # T6

    # エレボン指令値(command 0 ~ 1)
    delta_e_r_command = np.array(read_log_data.values[:,124])
    delta_e_l_command = np.array(read_log_data.values[:,125])

    # マニュアル操作量
    manual_pitch = np.array(read_log_data.values[:,374])
    manual_thrust = np.array(read_log_data.values[:,377])
    manual_tilt = np.array(read_log_data.values[:,389])

    # 時間
    time = np.array(read_log_data.values[:,392])

    # データサイズの取得（列方向）
    data_size = len(read_log_data)

    #---------------------------
    # 計算の必要がある値
    #---------------------------

    # ロータ推力
    Tm_up = THRUST_EF*0.5*GRA*(9.5636* 10**(-3)*m_up_pwm - 12.1379)
    Tm_down = THRUST_EF*0.5*GRA*(9.5636* 10**(-3)*m_down_pwm - 12.1379)
    Ts_r = GRA*(1.5701* 10**(-6) *(s_r_pwm)**2 - 3.3963*10**(-3)*s_r_pwm + 1.9386)
    Ts_l = GRA*(1.5701* 10**(-6) *(s_l_pwm)**2 - 3.3963*10**(-3)*s_l_pwm + 1.9386)
    Tf_up = GRA*(1.5701* 10**(-6) *(f_up_pwm)**2 - 3.3963*10**(-3)*f_up_pwm + 1.9386)
    Tf_down = GRA*(1.5701* 10**(-6) *(f_down_pwm)**2 - 3.3963*10**(-3)*f_down_pwm + 1.9386)

    # ロータ推力に制限をかける
    Tm_up[Tm_up < 0] = 0
    Tm_down[Tm_down < 0] = 0
    Ts_r[Ts_r > SUB_THRUST_MAX] = SUB_THRUST_MAX
    Ts_l[Ts_l > SUB_THRUST_MAX] = SUB_THRUST_MAX
    Tf_up[Tf_up > SUB_THRUST_MAX] = SUB_THRUST_MAX
    Tf_down[Tf_down > SUB_THRUST_MAX] = SUB_THRUST_MAX

    # エレボン舵角
    delta_e_r = ((delta_e_r_command*400 + 1500)/8 - 1500/8)*pi/180
    delta_e_l = ((delta_e_l_command*400 + 1500)/8 - 1500/8)*pi/180

    # エレベータ舵角，エルロン舵角に分ける
    elevator = (delta_e_l - delta_e_r)/2
    aileron = (delta_e_l + delta_e_r)/2

    # 速度
    Vg_pixhawk = []
    Vi = []
    Vi_wind = []
    Va = []

    # 機体速度（対地）の計算
    Vg_pixhawk = np.sqrt(
        d_position_x**2 \
        + d_position_y**2 \
        + d_position_z**2
    )

    # 機体速度と風速を慣性座標系へ変換
    for i in range(data_size):
        Vi.append(
            matex.bc2ic(phi[i],theta[i],psi[i],d_position_x[i],d_position_y[i],d_position_z[i])
        )
        Vi_wind.append(
            matex.bc2ic(phi[i],theta[i],0,V_W,0,0)
        )

    # リストからnumpy配列に変換
    Vi = np.array(Vi)
    Vi_wind = np.array(Vi_wind)

    # Pixhawk位置から重心位置の速度に変換
    Vi[:,2] = Vi[:,2] + d_theta*LEN_P

    # 対気速度を計算
    Va = Vi - Vi_wind
    Va_mag = np.sqrt(
        Va[:,0]**2
        + Va[:,1]**2
        + Va[:,2]**2
    )

    # 迎え角を計算[rad]
    alpha = np.arctan2(Va[:,2],Va[:,0])

    # 時間偏差を計算
    # サイズがひとつ小さくなるので，最後の値をそのまま一番うしろに付け足す
    time_diff = np.diff(time)
    time_diff = np.append(time_diff, time_diff[data_size-2])

    # 加速度
    d_Va_list = []
    dd_phi = []
    dd_theta = []
    dd_psi = []

    # 加速度を計算
    # 各軸ごとに計算してまとめ，最後にそれぞれを分割している
    d_Va_list = np.array(
    	matex.central_diff(Vi[:,0],time)
    ) # x軸
    d_Va_list = np.append(
        d_Va_list,
        matex.central_diff(Vi[:,1],time)
    ) # y軸
    d_Va_list = np.append(
        d_Va_list,
        matex.central_diff(Vi[:,2],time)
    ) # z軸
    d_Va =  np.reshape(d_Va_list,(data_size,3),order='F') # Unit

    dd_phi = matex.central_diff(d_phi,time)
    dd_theta = matex.central_diff(d_theta,time)
    dd_psi = matex.central_diff(d_psi,time)

    # どちらにティルトしているか
    tilt_switch = np.diff(manual_tilt)
    tilt_switch[np.isnan(tilt_switch)] = 0 # Nan -> 0

    for i in range(np.size(tilt_switch)):
        if tilt_switch[i] > 0: # MC -> FW
            tilt_switch[i] = 1
            continue
        elif tilt_switch[i] < 0: # FW -> MC
            tilt_switch[i] = -1
            continue
        elif tilt_switch[i] == 0:
            tilt_switch[i] = tilt_switch[i-1]
            continue

    tilt_switch = np.append(tilt_switch,tilt_switch[data_size-2])

    # ティルト角
    tilt = []

    # ティルト角を計算
    # ０度〜設定したティルト角の上限値に制限する
    for i in range(np.size(tilt_switch)):
        if tilt_switch[i] == 1:
            tilt = np.append(tilt,tilt[i-1] + (90/4.0)*(pi/180)*time_diff[i])
            if tilt[i] >= GAMMA:
                tilt[i] = GAMMA
                continue
        elif tilt_switch[i] == -1:
            tilt = np.append(tilt,tilt[i-1] - (90/4.0)*(pi/180)*time_diff[i])
            if tilt[i] < 0.0:
                tilt[i] = 0
                continue
        elif tilt_switch[i] == 0:
            tilt = np.append(tilt,0)

    # 空力の計算
    F_x = MASS * (d_Va[:,0] + d_theta*Vi[:,2]) \
                        + MASS * GRA * np.sin(theta)
    F_z = MASS * (d_Va[:,2] - d_theta*Vi[:,0]) \
                        - MASS * GRA * np.cos(theta)
    T_x = (Tm_up + Tm_down) * np.sin(tilt)
    T_z = - (Tm_up + Tm_down) * np.cos(tilt) \
                          - (Ts_r + Ts_l + Tf_up + Tf_down)
    A_x = F_x - T_x
    A_z = F_z - T_z

    # 揚力と抗力（実験値）
    L = A_x * np.sin(alpha) - A_z * np.cos(alpha)
    D = - A_x * np.cos(alpha) - A_z * np.sin(alpha)

    # 空力モーメントを計算
    M = I_YY * dd_theta # 全軸モーメント
    tau = LEN_F*(Tf_up + Tf_down) \
        - LEN_M*(Tm_up + Tm_down)*np.cos(tilt) \
        - LEN_S_X*(Ts_l + Ts_r) # ロータ推力によるモーメント
    Ma = M - tau

    #---------------------------
    # データを一つにまとめる
    #---------------------------

    # 最初のデータのためのイニット
    if file_number == 0:
        format_log_data = pd.DataFrame()

    # 同定に使うデータをまとめる
    format_log_data = pd.concat([format_log_data, pd.DataFrame({
        'phi' : phi,
        'theta' : theta,
        'psi' : psi,
        'd_phi' : d_phi,
        'd_theta' : d_theta,
        'd_psi' : d_psi,
        'dd_phi' : dd_phi,
        'dd_theta' : dd_theta,
        'dd_psi' : dd_psi,
        'position_x' : position_x,
        'position_y' : position_y,
        'position_z' : position_z,
        'U' : Va[:,0],
        'V' : Va[:,1],
        'W' : Va[:,2],
        'Va' : Va_mag,
        'Tm_up' : Tm_up,
        'Tm_down' : Tm_down,
        'Ts_r' : Ts_r,
        'Ts_l' : Ts_l,
        'Tf_up' : Tf_up,
        'Tf_down' : Tf_down,
        'alpha' : alpha,
        'delta_e' : elevator,
        'delta_a' : aileron,
        'F_x' : F_x,
        'T_x' : T_x,
        'F_z' : F_z,
        'T_z' : T_z,
        'L' : L,
        'D' : D,
        'M' : M,
        'tau' : tau,
        'Ma' : Ma,
    })])

#---------------------------
# 整理されたデータから値を取り出す
#---------------------------

data_size = len(format_log_data)
d_theta = np.array(format_log_data['d_theta'])
alpha = np.array(format_log_data['alpha'])
Va = np.array(format_log_data['Va'])
delta_e = np.array(format_log_data['delta_e'])
L = np.array(format_log_data['L'])
D = np.array(format_log_data['D'])
Ma = np.array(format_log_data['Ma'])

#---------------------------
# システム同定（最小二乗法を用いる）
#---------------------------

T_CONST = input('時定数の値を入力してください: ')
T_CONST = float(T_CONST)
T_DIFF = 0.02 # 時間偏差

#---------------------------
# 揚力
#---------------------------

# 既知パラメータ
CL_0 = 0.0634
CL_alpha = 2.68

# n*1 揚力から計算された値のリスト
yL = (L/((1/2)*RHO*(Va**2)*S)) - CL_0 - CL_alpha*alpha

# n*3 リグレッサー（独立変数）や実験データのリスト
xL = np.zeros((data_size,3))
xL[:,0] = (MAC*d_theta)/(2*Va)
xL[:,1] = delta_e
xL[:,2] = 1/((1/2)*RHO*Va*S)

# ３次ローパスフィルタをかける
for i in range(3):
    yL_filt = matex.lp_filter(T_CONST,T_DIFF,data_size,yL)
    xL_filt = matex.lp_filter(T_CONST,T_DIFF,data_size,xL)

# 擬似逆行列を用いた最小二乗解の計算
L_theta_hat = np.dot((np.linalg.pinv(xL)),yL)

# 同定された未知パラメータの取り出し
CL_q = L_theta_hat[0]
CL_delta_e = L_theta_hat[1]
k_L = L_theta_hat[2]

# 同定結果から得られたCLを計算
CL = CL_0 \
    + CL_alpha*alpha \
    + CL_q*(MAC/(2*Va))*d_theta \
    + CL_delta_e*delta_e

#---------------------------
# 抗力
#---------------------------

# 既知パラメータ
CD_0 = 0.07887

# n*1 抗力から計算された値のリスト
yD = (D/((1/2)*RHO*(Va**2)*S)) - CD_0

# n*2 リグレッサー（独立変数）や実験データのリスト
xD = np.zeros((data_size,2))
xD[:,0] = CL**2
xD[:,1] = 1/((1/2)*RHO*Va*S)

# ３次ローパスフィルタをかける
for i in range(3):
    yD_filt = matex.lp_filter(T_CONST,T_DIFF,data_size,yD)
    xD_filt = matex.lp_filter(T_CONST,T_DIFF,data_size,xD)

# 擬似逆行列を用いた最小二乗解の計算
D_theta_hat = np.dot((np.linalg.pinv(xD)),yD)

# 同定された未知パラメータの取り出し
kappa = D_theta_hat[0]
k_D = D_theta_hat[1]

# 同定結果から得られたCDを計算
CD = CD_0 + kappa*(CL**2)

#---------------------------
# モーメント
#---------------------------

# n*1 空力モーメントから計算された値のリスト
ym = Ma/((1/2)*RHO*(Va**2)*S*MAC)

# n*5 リグレッサー（独立変数）や実験データのリスト
xm = np.zeros((data_size,5))
xm[:,0] = 1
xm[:,1] = alpha
xm[:,2] = (MAC/(2*Va))*d_theta
xm[:,3] = delta_e
xm[:,4] = 1/((1/2)*RHO*Va*S*MAC)

# ３次ローパスフィルタをかける
for i in range(3):
    ym_filt = matex.lp_filter(T_CONST,T_DIFF,data_size,ym)
    xm_filt = matex.lp_filter(T_CONST,T_DIFF,data_size,xm)

# 擬似逆行列を用いた最小二乗解の計算
m_theta_hat = np.dot((np.linalg.pinv(xm)),ym)

# 同定された未知パラメータの取り出し
Cm_0 = m_theta_hat[0]
Cm_alpha = m_theta_hat[1]
Cm_q = m_theta_hat[2]
Cm_delta_e = m_theta_hat[3]
k_m = m_theta_hat[4]

# 同定結果から得られたCDを計算
Cm = Cm_0 \
    + Cm_alpha*alpha \
    + Cm_q*(MAC/(2*Va))*d_theta \
    + Cm_delta_e*delta_e

#---------------------------
# 同定結果を用いて計算
#---------------------------

L_calc = (1/2)*RHO*S*(Va**2)*CL + k_L*Va
D_calc = (1/2)*RHO*S*(Va**2)*CD + k_D*Va
Ma_calc = (1/2)*RHO*S*(Va**2)*MAC*Cm + k_m*Va

#---------------------------
# 結果
#---------------------------
print(L_theta_hat)
print(D_theta_hat)
print(m_theta_hat)

#---------------------------
plt.figure(1)

# 余白を設定
plt.subplots_adjust(wspace=0.4, hspace=0.6)

plt.subplot(3,1,1)
plt.plot(L)
plt.plot(L_calc)
plt.grid()
plt.xlabel('データ番号')
plt.ylabel('揚力')

plt.subplot(3,1,2)
plt.plot(D)
plt.plot(D_calc)
plt.grid()
plt.xlabel('データ番号')
plt.ylabel('抗力')

plt.subplot(3,1,3)
plt.plot(Ma)
plt.plot(Ma_calc)
plt.grid()
plt.xlabel('データ番号')
plt.ylabel('モーメント')

#---------------------------
plt.figure(2)

# 余白を設定
plt.subplots_adjust(wspace=0.4, hspace=0.6)

plt.subplot(3,1,1)
plt.plot(yL, 'blue')
plt.plot(yL_filt, 'red')
plt.grid()
plt.xlabel('データ番号')
plt.ylabel('揚力')

plt.subplot(3,1,2)
plt.plot(yD, 'blue')
plt.plot(yD_filt, 'red')
plt.grid()
plt.xlabel('データ番号')
plt.ylabel('抗力')

plt.subplot(3,1,3)
plt.plot(ym, 'blue')
plt.plot(ym_filt, 'red')
plt.grid()
plt.xlabel('データ番号')
plt.ylabel('モーメント')

#---------------------------
plt.show()
