# -*- coding: utf-8 -*-
# author: ub
# 2018/12/14 Fri. 新座標系．

#---------------------------
# モジュールのインポートなど
#---------------------------

import numpy as np
from numpy import pi
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib.font_manager
from IPython import get_ipython
# from scipy import signal

import const
import math_extention as matex
import calc
import calc_ex
import calc_ex_max
import calc_kawano
import analyze

#---------------------------
# matplotlibの諸設定
#---------------------------

# プロットデータを新しいウィンドウで表示する
get_ipython().run_line_magic('matplotlib', 'qt')

# 日本語フォントの設定
# 使用できるフォントを確認したいときは，次の行のコメントアウトを外して実行
# print([f.name for f in matplotlib.font_manager.fontManager.ttflist])

# for NotePC
plt.rc('font', **{'family':'Gen Shin Gothic'})

# for DeskPC
# plt.rc('font', **{'family':'YuGothic'})

plt.rcParams['font.size'] = 20
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15 # default: 12

# プロットデータのサイズ設定
plt.rcParams["figure.figsize"] = [20, 12]

#---------------------------
# ログデータの読み込み
#---------------------------

FILE_NUM = 6 # 読み込むファイル数
borderline_data_num = np.zeros(FILE_NUM-1) # 読み込むファイルごとの境目

# ファイルを一つずつ読み込む
for file_number in range(FILE_NUM):

    #---------------------------
    # CSVデータを読み込んで整理する
    #---------------------------

    if file_number == 0:
        read_log_data = pd.read_csv(
            filepath_or_buffer='../log_data/Book3.csv',
            encoding='ASCII',
            sep=',',
            header=0,
            usecols=['ATT_Roll',
                     'ATT_Pitch',
                     'ATT_Yaw',
                     'ATT_RollRate',
                     'ATT_PitchRate',
                     'ATT_YawRate',
                     'LPOS_X',
                     'LPOS_Y',
                     'LPOS_Z',
                     'LPOS_VX',
                     'LPOS_VY',
                     'LPOS_VZ',
                     'GPS_Alt',
                     'OUT0_Out0',
                     'OUT0_Out1',
                     'OUT0_Out2',
                     'OUT0_Out3',
                     'OUT0_Out4',
                     'OUT0_Out5',
                     'OUT1_Out0',
                     'OUT1_Out1',
                     'AIRS_TrueSpeed',
                     'MAN_pitch',
                     'MAN_thrust',
                     'VTOL_Tilt',
                     'TIME_StartTime'
                     ]
        )

        # 空白行を削除
        read_log_data = read_log_data.dropna(how='all')
        read_log_data = read_log_data.reset_index(drop=True)

        # 時間データを[秒]に変換
        read_log_data['Time_ST'] = read_log_data.at[0,'TIME_StartTime']
        read_log_data['Time_Conv'] = (read_log_data['TIME_StartTime'] - read_log_data['Time_ST'])/1000000

        # 重複データの削除
        read_log_data = read_log_data.drop_duplicates(subset='TIME_StartTime')

        # 実験時間のみ切り取り
        read_log_data = read_log_data.query(
            '17.52 <= Time_Conv <= 19.14'
        )

        # ファイルに依存する値（風速，推力効率，ティルト角）
        V_W = -4.03
        THRUST_EF = 40/48
        GAMMA = 0

    elif file_number == 1:
        read_log_data = pd.read_csv(
            filepath_or_buffer='../log_data/Book4.csv',
            encoding='ASCII',
            sep=',',
            header=0,
            usecols=['ATT_Roll',
                     'ATT_Pitch',
                     'ATT_Yaw',
                     'ATT_RollRate',
                     'ATT_PitchRate',
                     'ATT_YawRate',
                     'LPOS_X',
                     'LPOS_Y',
                     'LPOS_Z',
                     'LPOS_VX',
                     'LPOS_VY',
                     'LPOS_VZ',
                     'GPS_Alt',
                     'OUT0_Out0',
                     'OUT0_Out1',
                     'OUT0_Out2',
                     'OUT0_Out3',
                     'OUT0_Out4',
                     'OUT0_Out5',
                     'OUT1_Out0',
                     'OUT1_Out1',
                     'AIRS_TrueSpeed',
                     'MAN_pitch',
                     'MAN_thrust',
                     'VTOL_Tilt',
                     'TIME_StartTime'
                     ]
        )

        # 空白行を削除
        read_log_data = read_log_data.dropna(how='all')
        read_log_data = read_log_data.reset_index(drop=True)

        # 時間データを[秒]に変換
        read_log_data['Time_ST'] = read_log_data.at[0,'TIME_StartTime']
        read_log_data['Time_Conv'] = (read_log_data['TIME_StartTime'] - read_log_data['Time_ST'])/1000000

        # 重複データの削除
        read_log_data = read_log_data.drop_duplicates(subset='TIME_StartTime')

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
            filepath_or_buffer='../log_data/Book5.csv',
            encoding='ASCII',
            sep=',',
            header=0,
            usecols=['ATT_Roll',
                     'ATT_Pitch',
                     'ATT_Yaw',
                     'ATT_RollRate',
                     'ATT_PitchRate',
                     'ATT_YawRate',
                     'LPOS_X',
                     'LPOS_Y',
                     'LPOS_Z',
                     'LPOS_VX',
                     'LPOS_VY',
                     'LPOS_VZ',
                     'GPS_Alt',
                     'OUT0_Out0',
                     'OUT0_Out1',
                     'OUT0_Out2',
                     'OUT0_Out3',
                     'OUT0_Out4',
                     'OUT0_Out5',
                     'OUT1_Out0',
                     'OUT1_Out1',
                     'AIRS_TrueSpeed',
                     'MAN_pitch',
                     'MAN_thrust',
                     'VTOL_Tilt',
                     'TIME_StartTime'
                     ]
        )

        # 空白行を削除
        read_log_data = read_log_data.dropna(how='all')
        read_log_data = read_log_data.reset_index(drop=True)

        # 時間データを[秒]に変換
        read_log_data['Time_ST'] = read_log_data.at[0,'TIME_StartTime']
        read_log_data['Time_Conv'] = (read_log_data['TIME_StartTime'] - read_log_data['Time_ST'])/1000000

        # 重複データの削除
        read_log_data = read_log_data.drop_duplicates(subset='TIME_StartTime')

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
            filepath_or_buffer='../log_data/Book8.csv',
            encoding='ASCII',
            sep=',',
            header=0,
            usecols=['ATT_Roll',
                     'ATT_Pitch',
                     'ATT_Yaw',
                     'ATT_RollRate',
                     'ATT_PitchRate',
                     'ATT_YawRate',
                     'LPOS_X',
                     'LPOS_Y',
                     'LPOS_Z',
                     'LPOS_VX',
                     'LPOS_VY',
                     'LPOS_VZ',
                     'GPS_Alt',
                     'OUT0_Out0',
                     'OUT0_Out1',
                     'OUT0_Out2',
                     'OUT0_Out3',
                     'OUT0_Out4',
                     'OUT0_Out5',
                     'OUT1_Out0',
                     'OUT1_Out1',
                     'AIRS_TrueSpeed',
                     'MAN_pitch',
                     'MAN_thrust',
                     'VTOL_Tilt',
                     'TIME_StartTime'
                     ]
        )

        # 空白行を削除
        read_log_data = read_log_data.dropna(how='all')
        read_log_data = read_log_data.reset_index(drop=True)

        # 時間データを[秒]に変換
        read_log_data['Time_ST'] = read_log_data.at[0,'TIME_StartTime']
        read_log_data['Time_Conv'] = (read_log_data['TIME_StartTime'] - read_log_data['Time_ST'])/1000000

        # 重複データの削除
        read_log_data = read_log_data.drop_duplicates(subset='TIME_StartTime')

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
            filepath_or_buffer='../log_data/Book9.csv',
            encoding='ASCII',
            sep=',',
            header=0,
            usecols=['ATT_Roll',
                     'ATT_Pitch',
                     'ATT_Yaw',
                     'ATT_RollRate',
                     'ATT_PitchRate',
                     'ATT_YawRate',
                     'LPOS_X',
                     'LPOS_Y',
                     'LPOS_Z',
                     'LPOS_VX',
                     'LPOS_VY',
                     'LPOS_VZ',
                     'GPS_Alt',
                     'OUT0_Out0',
                     'OUT0_Out1',
                     'OUT0_Out2',
                     'OUT0_Out3',
                     'OUT0_Out4',
                     'OUT0_Out5',
                     'OUT1_Out0',
                     'OUT1_Out1',
                     'AIRS_TrueSpeed',
                     'MAN_pitch',
                     'MAN_thrust',
                     'VTOL_Tilt',
                     'TIME_StartTime'
                     ]
        )

        # 空白行を削除
        read_log_data = read_log_data.dropna(how='all')
        read_log_data = read_log_data.reset_index(drop=True)

        # 時間データを[秒]に変換
        read_log_data['Time_ST'] = read_log_data.at[0,'TIME_StartTime']
        read_log_data['Time_Conv'] = (read_log_data['TIME_StartTime'] - read_log_data['Time_ST'])/1000000

        # 重複データの削除
        read_log_data = read_log_data.drop_duplicates(subset='TIME_StartTime')

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
            filepath_or_buffer='../log_data/Book11.csv',
            encoding='ASCII',
            sep=',',
            header=0,
            usecols=['ATT_Roll',
                     'ATT_Pitch',
                     'ATT_Yaw',
                     'ATT_RollRate',
                     'ATT_PitchRate',
                     'ATT_YawRate',
                     'LPOS_X',
                     'LPOS_Y',
                     'LPOS_Z',
                     'LPOS_VX',
                     'LPOS_VY',
                     'LPOS_VZ',
                     'GPS_Alt',
                     'OUT0_Out0',
                     'OUT0_Out1',
                     'OUT0_Out2',
                     'OUT0_Out3',
                     'OUT0_Out4',
                     'OUT0_Out5',
                     'OUT1_Out0',
                     'OUT1_Out1',
                     'AIRS_TrueSpeed',
                     'MAN_pitch',
                     'MAN_thrust',
                     'VTOL_Tilt',
                     'TIME_StartTime'
                     ]
        )

        # 空白行を削除
        read_log_data = read_log_data.dropna(how='all')
        read_log_data = read_log_data.reset_index(drop=True)

        # 時間データを[秒]に変換
        read_log_data['Time_ST'] = read_log_data.at[0,'TIME_StartTime']
        read_log_data['Time_Conv'] = (read_log_data['TIME_StartTime'] - read_log_data['Time_ST'])/1000000

        # 重複データの削除
        read_log_data = read_log_data.drop_duplicates(subset='TIME_StartTime')

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
    phi = np.array(read_log_data['ATT_Roll'])
    theta = np.array(read_log_data['ATT_Pitch'])
    psi = np.array(read_log_data['ATT_Yaw'])

    # 角速度
    d_phi = np.array(read_log_data['ATT_RollRate'])
    d_theta = np.array(read_log_data['ATT_PitchRate'])
    d_psi = np.array(read_log_data['ATT_YawRate'])

    # 位置
    position_x = np.array(read_log_data['LPOS_X'])
    position_y = np.array(read_log_data['LPOS_Y'])
    position_z = np.array(read_log_data['LPOS_Z'])

    # 速度
    d_position_x = np.array(read_log_data['LPOS_VX'])
    d_position_y = np.array(read_log_data['LPOS_VY'])
    d_position_z = np.array(read_log_data['LPOS_VZ'])

    # GPS高度
    gps_altitude = np.array(read_log_data['GPS_Alt'])

    # ピトー管から得た対気速度
    measurement_airspeed = np.array(read_log_data['AIRS_TrueSpeed'])

    # ロータ指令値
    m_up_pwm = np.array(read_log_data['OUT0_Out0']) # T1
    m_down_pwm = np.array(read_log_data['OUT0_Out1']) # T2
    r_r_pwm = np.array(read_log_data['OUT0_Out2']) # T3
    r_l_pwm = np.array(read_log_data['OUT0_Out3']) # T4
    f_up_pwm = np.array(read_log_data['OUT0_Out4']) # T5
    f_down_pwm = np.array(read_log_data['OUT0_Out5']) # T6

    # エレボン指令値(command 0 ~ 1)
    delta_e_r_command = np.array(read_log_data['OUT1_Out0'])
    delta_e_l_command = np.array(read_log_data['OUT1_Out1'])

    # マニュアル操作量
    manual_pitch = np.array(read_log_data['MAN_pitch'])
    manual_thrust = np.array(read_log_data['MAN_thrust'])
    manual_tilt = np.array(read_log_data['VTOL_Tilt'])

    # 時間
    time = np.array(read_log_data['Time_Conv'])

    # データサイズの取得（列方向）
    data_size = len(read_log_data)

    #
    if(file_number == 0):
        borderline_data_num[file_number] = data_size
    elif(0 < file_number < FILE_NUM-1):
        borderline_data_num[file_number] = data_size + borderline_data_num[file_number-1]

    #---------------------------
    # 計算の必要がある値
    #---------------------------

    # ロータ推力
    Tm_up = THRUST_EF*0.5*const.GRA*(9.5636* 10**(-3)*m_up_pwm - 12.1379)
    Tm_down = THRUST_EF*0.5*const.GRA*(9.5636* 10**(-3)*m_down_pwm - 12.1379)
    Tr_r = const.GRA*(1.5701* 10**(-6) *(r_r_pwm)**2 - 3.3963*10**(-3)*r_r_pwm + 1.9386)
    Tr_l = const.GRA*(1.5701* 10**(-6) *(r_l_pwm)**2 - 3.3963*10**(-3)*r_l_pwm + 1.9386)
    Tf_up = const.GRA*(1.5701* 10**(-6) *(f_up_pwm)**2 - 3.3963*10**(-3)*f_up_pwm + 1.9386)
    Tf_down = const.GRA*(1.5701* 10**(-6) *(f_down_pwm)**2 - 3.3963*10**(-3)*f_down_pwm + 1.9386)

    # ロータ推力に制限をかける
    Tm_up[Tm_up < 0] = 0
    Tm_down[Tm_down < 0] = 0
    Tr_r[Tr_r > const.SUB_THRUST_MAX] = const.SUB_THRUST_MAX
    Tr_l[Tr_l > const.SUB_THRUST_MAX] = const.SUB_THRUST_MAX
    Tf_up[Tf_up > const.SUB_THRUST_MAX] = const.SUB_THRUST_MAX
    Tf_down[Tf_down > const.SUB_THRUST_MAX] = const.SUB_THRUST_MAX

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

    # センサー位置の補正
    Vi[:,2] = Vi[:,2] + d_theta*const.LEN_P

    # 対気速度を計算
    Va = Vi - Vi_wind
    Va_mag = np.sqrt(
        Va[:,0]**2
        + Va[:,1]**2
        + Va[:,2]**2
    )

    # 迎角を計算[rad]
    alpha = np.arctan2(Va[:,2],Va[:,0])

    # 迎角の一階時間微分を計算[rad]
    d_alpha = matex.central_diff(alpha, time)

    # 時間偏差を計算
    # サイズがひとつ小さくなるので，最後の値をそのまま一番うしろに付け足す
    time_diff = np.diff(time)
    time_diff = np.append(time_diff, time_diff[data_size-2])

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
    F_x = const.MASS * (d_Va[:,0] + d_theta*Vi[:,2]) \
                        + const.MASS * const.GRA * np.sin(theta)
    F_z = const.MASS * (d_Va[:,2] - d_theta*Vi[:,0]) \
                        - const.MASS * const.GRA * np.cos(theta)
    T_x = (Tm_up + Tm_down) * np.sin(tilt)
    T_z = - (Tm_up + Tm_down) * np.cos(tilt) \
                          - (Tr_r + Tr_l + Tf_up + Tf_down)
    A_x = F_x - T_x
    A_z = F_z - T_z

    # 揚力と抗力（実験値）
    L = A_x * np.sin(alpha) - A_z * np.cos(alpha)
    D = - A_x * np.cos(alpha) - A_z * np.sin(alpha)

    # 空力モーメントを計算
    M = const.I_YY * dd_theta # 全軸モーメント
    Mt = const.LEN_F*(Tf_up + Tf_down) \
        - const.LEN_M*(Tm_up + Tm_down)*np.cos(tilt) \
        - const.LEN_R_X*(Tr_l + Tr_r) # ロータ推力によるモーメント

    # 重力によるモーメント
    Lg = const.R_G_Z*const.MASS*const.GRA*np.cos(theta)*np.sin(phi)
    Mg = - const.R_G_Z*const.MASS*const.GRA*np.sin(theta) \
         - const.R_G_X*const.MASS*const.GRA*np.cos(theta)*np.cos(phi)
    Ng = const.R_G_X*const.MASS*const.GRA*np.cos(theta)*np.sin(phi)

    Ma = M - Mt - Mg

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
        'u' : Va[:,0],
        'v' : Va[:,1],
        'w' : Va[:,2],
        'Va' : Va_mag,
        'Tm_up' : Tm_up,
        'Tm_down' : Tm_down,
        'Tr_r' : Tr_r,
        'Tr_l' : Tr_l,
        'Tf_up' : Tf_up,
        'Tf_down' : Tf_down,
        'alpha' : alpha,
        'd_alpha' : d_alpha,
        'delta_e' : elevator,
        'delta_a' : aileron,
        'tilt' : tilt,
        'F_x' : F_x,
        'T_x' : T_x,
        'F_z' : F_z,
        'T_z' : T_z,
        'L' : L,
        'D' : D,
        'M' : M,
        'Mt' : Mt,
        'Mg' : Mg,
        'Ma' : Ma,
        'pitot_Va' : measurement_airspeed,
        'manual_T1' : m_up_pwm,
        'manual_T2' : m_down_pwm,
        'manual_T3' : r_r_pwm,
        'manual_T4' : r_l_pwm,
        'manual_T5' : f_up_pwm,
        'manual_T6' : f_down_pwm,
        'manual_elevon_r' : delta_e_r_command,
        'manual_elevon_l' : delta_e_l_command,
        'manual_pitch' : manual_pitch,
        'manual_thrust' : manual_thrust,
        'manual_tilt' : manual_tilt,
        })
    ])

#---------------------------
# パラメータ推定の結果を計算し，取得
#---------------------------

sys_id_result = calc.sys_id_LS(format_log_data)
# sys_id_result = calc_ex.sys_id_LS_ex(format_log_data)
# sys_id_result = calc_ex_max.sys_id_LS_ex_max(format_log_data)
# sys_id_result = calc_kawano.sys_id_LS_kawano(format_log_data)

#---------------------------
# 推定結果の値もデータ群に格納する
#---------------------------

# d_alphaを含まない場合
if sys_id_result[0].shape[1] == 5:
    format_log_data['CL_0'] = sys_id_result[0][:,0]
    format_log_data['CL_alpha'] = sys_id_result[0][:,1]
    format_log_data['CL_q'] = sys_id_result[0][:,2]
    format_log_data['CL_delta_e'] = sys_id_result[0][:,3]
    format_log_data['k_L'] = sys_id_result[0][:,4]

    format_log_data['CD_0'] = sys_id_result[1][:,0]
    format_log_data['kappa'] = sys_id_result[1][:,1]
    format_log_data['k_D'] = sys_id_result[1][:,2]

    format_log_data['Cm_0'] = sys_id_result[2][:,0]
    format_log_data['Cm_alpha'] = sys_id_result[2][:,1]
    format_log_data['Cm_q'] = sys_id_result[2][:,2]
    format_log_data['Cm_delta_e'] = sys_id_result[2][:,3]
    format_log_data['k_m'] = sys_id_result[2][:,4]

    format_log_data['CL'] = sys_id_result[3][:,0]
    format_log_data['CD'] = sys_id_result[3][:,1]
    format_log_data['Cm'] = sys_id_result[3][:,2]
    format_log_data['L_calc'] = sys_id_result[3][:,3]
    format_log_data['D_calc'] = sys_id_result[3][:,4]
    format_log_data['Ma_calc'] = sys_id_result[3][:,5]

# d_alphaを含む場合
elif sys_id_result[0].shape[1] == 6:
    format_log_data['CL_0'] = sys_id_result[0][:,0]
    format_log_data['CL_alpha'] = sys_id_result[0][:,1]
    format_log_data['CL_d_alpha'] = sys_id_result[0][:,2]
    format_log_data['CL_q'] = sys_id_result[0][:,3]
    format_log_data['CL_delta_e'] = sys_id_result[0][:,4]
    format_log_data['k_L'] = sys_id_result[0][:,5]

    format_log_data['CD_0'] = sys_id_result[1][:,0]
    format_log_data['kappa'] = sys_id_result[1][:,1]
    format_log_data['k_D'] = sys_id_result[1][:,2]

    format_log_data['Cm_0'] = sys_id_result[2][:,0]
    format_log_data['Cm_alpha'] = sys_id_result[2][:,1]
    format_log_data['Cm_d_alpha'] = sys_id_result[2][:,2]
    format_log_data['Cm_q'] = sys_id_result[2][:,3]
    format_log_data['Cm_delta_e'] = sys_id_result[2][:,4]
    format_log_data['k_m'] = sys_id_result[2][:,5]

    format_log_data['CL'] = sys_id_result[3][:,0]
    format_log_data['CD'] = sys_id_result[3][:,1]
    format_log_data['Cm'] = sys_id_result[3][:,2]
    format_log_data['L_calc'] = sys_id_result[3][:,3]
    format_log_data['D_calc'] = sys_id_result[3][:,4]
    format_log_data['Ma_calc'] = sys_id_result[3][:,5]

#---------------------------
# 機体の状態方程式から固有振動数を解析する
#---------------------------

# anly_result = analyze.linearlize(format_log_data)

#---------------------------
# データの取り出し
#---------------------------
data_size = len(format_log_data) # 合計のデータサイズを取得
theta = np.array(format_log_data['theta'])
d_theta = np.array(format_log_data['d_theta'])
alpha = np.array(format_log_data['alpha'])
Va = np.array(format_log_data['Va'])
pitot_Va = np.array(format_log_data['pitot_Va'])
delta_e = np.array(format_log_data['delta_e'])
L = np.array(format_log_data['L'])
D = np.array(format_log_data['D'])
Ma = np.array(format_log_data['Ma'])


CD = np.array(format_log_data['CD'])

manual_T1 = np.array(format_log_data['manual_T1'])
manual_T2 = np.array(format_log_data['manual_T2'])
manual_T3 = np.array(format_log_data['manual_T3'])
manual_T4 = np.array(format_log_data['manual_T4'])
manual_T5 = np.array(format_log_data['manual_T5'])
manual_T6 = np.array(format_log_data['manual_T6'])
manual_elevon_r = np.array(format_log_data['manual_elevon_r'])
manual_elevon_l = np.array(format_log_data['manual_elevon_l'])
manual_pitch = np.array(format_log_data['manual_pitch'])
manual_thrust = np.array(format_log_data['manual_thrust'])
manual_tilt = np.array(format_log_data['manual_tilt'])


# window = np.hamming(data_size)
# manual_T3 = window * manual_T3

# # 固有値の絶対値をとる．
# lambda_A_abs = np.abs(anly_result[0])
#
# xxx = np.arange(data_size)
# y = lambda_A_abs[:,0]
# yy = lambda_A_abs[:,1]
# yyy = lambda_A_abs[:,2]
# yyyy = lambda_A_abs[:,3]

# plt.subplot(111)
# plt.scatter(xxx,y)
# plt.scatter(xxx,yy)
# plt.scatter(xxx,yyy)
# plt.scatter(xxx,yyyy)
#
#
# for j in range(FILE_NUM-1):
#     plt.axvline(x=borderline_data_num[j], color="black") # 実験データの境目で線を引く
#
# plt.title('固有値散布図')
# plt.xlabel('データ番号')
# plt.ylabel('固有値')
#
# # ax = fig.add_subplot(2,1,2)
# #
# # ax.plot(xxx,d_alpha)
#

#---------------------------
# フーリエ変換
#---------------------------

# 周波数軸のデータ作成
fq = np.fft.fftfreq(data_size,d=0.02)

# FFT
F_d_theta = matex.fft_set_amp(d_theta,0.02,data_size)

F_manual_T1 = matex.fft_set_amp(manual_T1,0.02,data_size)
F_manual_T2 = matex.fft_set_amp(manual_T2,0.02,data_size)
F_manual_T3 = matex.fft_set_amp(manual_T3,0.02,data_size)
F_manual_T4 = matex.fft_set_amp(manual_T4,0.02,data_size)
F_manual_T5 = matex.fft_set_amp(manual_T5,0.02,data_size)
F_manual_T6 = matex.fft_set_amp(manual_T6,0.02,data_size)
F_manual_elevon_r = matex.fft_set_amp(manual_elevon_r,0.02,data_size)
F_manual_elevon_l = matex.fft_set_amp(manual_elevon_l,0.02,data_size)
F_manual_pitch = matex.fft_set_amp(manual_pitch,0.02,data_size)
F_manual_thrust = matex.fft_set_amp(manual_thrust,0.02,data_size)
F_manual_tilt = matex.fft_set_amp(manual_tilt,0.02,data_size)

# ３次ローパスフィルタをかける
for i in range(3):
    theta_filt = matex.lp_filter(0.03,0.02,data_size,theta)

#---------------------------
# 結果
#---------------------------

## ピッチ角，角速度，迎角をプロット
# fig1 = plt.figure()
#
# ax1 = fig1.add_subplot(311)
# ax2 = fig1.add_subplot(312)
# ax3 = fig1.add_subplot(313)
#
# ax1.plot(theta)
# ax2.plot(d_theta)
# ax3.plot(alpha)
#
# ax1.set_title("Pitch")
# ax2.set_title("Pitch Rate")
# ax3.set_title("Angle of attack")
#
# ax1.set_ylabel("[rad]")
# ax2.set_ylabel("[rad]")
# ax3.set_ylabel("[rad]")
#
# fig1.tight_layout()


# plt.subplot(111)
# plt.plot(fq[1:int(data_size/2)], F_manual_T3[1:int(data_size/2)])
# plt.xlabel('frequency [Hz]')
# plt.ylabel('amplitude')

#---------------------------
# plt.show()
