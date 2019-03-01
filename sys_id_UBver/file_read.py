# -*- coding: utf-8 -*-
'''
author: ub
ファイル読み込みに関する関数．
'''

import numpy as np
from numpy import pi
import pandas as pd

import const
import math_extention as matex
import fourier_filter as ffilt


def file_read(id,filename, section_ST, section_ED, V_W, T_EF, RHO, GAMMA, input_log_data):
    '''
    CSVファイルを読み込み，それぞれ必要な計算をして，DataFrameにまとめる.

    Parameters
    ----------
    id : int
        読み込むファイルに番号をつける．
        他と重複していなければなんでもいい．
    filename : str
        読み込むCSVファイル.
    section_ST : float64
        ファイル切り取り区間の始め[s].
    section_ED : float64
        ファイル切り取り区間の終わり[s].
    V_W : float64
        風速．
    T_EF : float64
        推力効率係数．メインとサブで均等に分配．
    RHO : float64
        大気密度．実験ごとに測定された気温などから計算．
    GAMMA: int
        ティルト角．
    input_log_data: pandas.DataFrame
        書き込むデータファイル．

    Returns
    -------
    format_log_data : pandas.DataFrame
        整理後のデータファイル．
    data_size : int
        各データファイルのサイズ．
    '''


    #---------------------------
    # ファイルの読み込み
    #---------------------------

    # CSVファイルの読み込み
    df = pd.read_csv(
        filepath_or_buffer=filename,
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
    df = df.dropna(how='all')
    df = df.reset_index(drop=True)

    # 時間データを[秒]に変換
    df['Time_ST'] = df.at[0,'TIME_StartTime']
    df['Time_Conv'] = (df['TIME_StartTime'] - df['Time_ST'])/1000000

    # 重複データの削除
    df = df.drop_duplicates(subset='TIME_StartTime')

    # 実験時間のみ切り取り
    df = df[(section_ST <= df['Time_Conv']) & (df['Time_Conv'] <= section_ED)]

    #---------------------------
    # 各データを取り出す
    #---------------------------

    # 角度
    phi = np.array(df['ATT_Roll'])
    theta = np.array(df['ATT_Pitch'])
    psi = np.array(df['ATT_Yaw'])

    # 角速度 p,q,r
    d_phi = np.array(df['ATT_RollRate'])
    d_theta = np.array(df['ATT_PitchRate'])
    d_psi = np.array(df['ATT_YawRate'])

    # 位置 x,y,z
    position_x = np.array(df['LPOS_X'])
    position_y = np.array(df['LPOS_Y'])
    position_z = np.array(df['LPOS_Z'])

    # 速度
    d_position_x = np.array(df['LPOS_VX'])
    d_position_y = np.array(df['LPOS_VY'])
    d_position_z = np.array(df['LPOS_VZ'])
    Vp = np.array([d_position_x, d_position_y, d_position_z])

    # GPS高度
    gps_altitude = np.array(df['GPS_Alt'])

    # ピトー管から得た対気速度
    measurement_airspeed = np.array(df['AIRS_TrueSpeed'])

    # ロータ指令値
    m_up_pwm = np.array(df['OUT0_Out0']) # T1
    m_down_pwm = np.array(df['OUT0_Out1']) # T2
    r_r_pwm = np.array(df['OUT0_Out2']) # T3
    r_l_pwm = np.array(df['OUT0_Out3']) # T4
    f_up_pwm = np.array(df['OUT0_Out4']) # T5
    f_down_pwm = np.array(df['OUT0_Out5']) # T6

    # エレボン指令値(command 0 ~ 1)
    delta_e_r_command = np.array(df['OUT1_Out0'])
    delta_e_l_command = np.array(df['OUT1_Out1'])

    # マニュアル操作量
    manual_pitch = np.array(df['MAN_pitch'])
    manual_thrust = np.array(df['MAN_thrust'])
    manual_tilt = np.array(df['VTOL_Tilt'])

    # 時間
    time = np.array(df['Time_Conv'])

    # データサイズの取得（列方向）
    data_size = len(df)

    #---------------------------
    # 計算の必要がある値
    #---------------------------

    # メイン，サブロータ両方に等しくかかる推力効率係数T_Eを算出
    X_HOVER = 40 / T_EF
    T_E = const.MASS*const.GRA / ((const.MASS*const.GRA-40)+X_HOVER)

    # サブロータ出力35%のとき
    # T_E = 57/70

    # サブ前とサブ左右の係数比重を変えたとき
    T_E_R = T_E * 10/18
    T_E_F = T_E * 8/18

    # ロータ推力 推算値，要修正？
    Tm_up = T_E*0.5*const.GRA*(9.5636* 10**(-3)*m_up_pwm - 12.1379)
    Tm_down = T_E*0.5*const.GRA*(9.5636* 10**(-3)*m_down_pwm - 12.1379)
    Tr_r = T_E_R*const.GRA*(1.5701* 10**(-6) *(r_r_pwm)**2 - 3.3963*10**(-3)*r_r_pwm + 1.9386)
    Tr_l = T_E_R*const.GRA*(1.5701* 10**(-6) *(r_l_pwm)**2 - 3.3963*10**(-3)*r_l_pwm + 1.9386)
    Tf_up = T_E_F*const.GRA*(1.5701* 10**(-6) *(f_up_pwm)**2 - 3.3963*10**(-3)*f_up_pwm + 1.9386)
    Tf_down = T_E_F*const.GRA*(1.5701* 10**(-6) *(f_down_pwm)**2 - 3.3963*10**(-3)*f_down_pwm + 1.9386)

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
    delta_e = (delta_e_l - delta_e_r)/2
    delta_a = (delta_e_l + delta_e_r)/2

    # 速度
    Vg = []
    Vg_wind = []

    # 機体速度と風速を慣性座標系へ変換
    for i in range(data_size):
        Vg.append(
            matex.bc2ic(phi[i],theta[i],psi[i],d_position_x[i],d_position_y[i],d_position_z[i])
        )
        Vg_wind.append(
            matex.bc2ic(phi[i],theta[i],0,V_W,0,0) # 風に対してヨー角はずれていないと仮定
        )

    # リストからnumpy配列に変換
    Vg = np.array(Vg)
    Vg_wind = np.array(Vg_wind)

    # センサー位置の補正
    Vg[:,0] = Vg[:,0] + d_theta*const.LEN_PZ
    Vg[:,1] = Vg[:,1] - d_psi*const.LEN_PX - d_phi*const.LEN_PZ
    Vg[:,2] = Vg[:,2] + d_theta*const.LEN_PX

    # 対気速度を計算
    Va = Vg - Vg_wind
    Va_mag = np.sqrt(
        Va[:,0]**2
        + Va[:,1]**2
        + Va[:,2]**2
    )

    # 迎角を計算[rad]
    alpha = np.arctan2(Va[:,2],Va[:,0])
    alpha_deg = alpha*(180/pi)

    # 迎角の一階時間微分を計算[rad]
    d_alpha = matex.central_diff(alpha, time)

    # 時間偏差を計算
    # サイズがひとつ小さくなるので，最後の値をそのまま一番うしろに付け足す
    time_diff = np.diff(time)
    time_diff = np.append(time_diff, time_diff[data_size-2])

    # 加速度を計算
    # 各軸ごとに計算してまとめ，最後にそれぞれを分割している
    d_Va_list = np.array(
    	matex.central_diff(Vg[:,0],time)
    ) # x軸
    d_Va_list = np.append(
        d_Va_list,
        matex.central_diff(Vg[:,1],time)
    ) # y軸
    d_Va_list = np.append(
        d_Va_list,
        matex.central_diff(Vg[:,2],time)
    ) # z軸
    d_Va =  np.reshape(d_Va_list,(data_size,3),order='F') # Unit

    dd_phi = matex.central_diff(d_phi,time)
    dd_theta = matex.central_diff(d_theta,time)
    dd_psi = matex.central_diff(d_psi,time)

    # ティルト角固定
    tilt = GAMMA * (pi/180)

    # # どちらにティルトしているか
    # tilt_switch = np.diff(manual_tilt)
    # tilt_switch[np.isnan(tilt_switch)] = 0 # Nan -> 0
    #
    # for i in range(np.size(tilt_switch)):
    #     if tilt_switch[i] > 0: # MC -> FW
    #         tilt_switch[i] = 1
    #         continue
    #     elif tilt_switch[i] < 0: # FW -> MC
    #         tilt_switch[i] = -1
    #         continue
    #     elif tilt_switch[i] == 0:
    #         tilt_switch[i] = tilt_switch[i-1]
    #         continue
    #
    # tilt_switch = np.append(tilt_switch,tilt_switch[data_size-2])

    # # ティルト角
    # tilt = []
    #
    # # ティルト角を計算
    # # ０度〜設定したティルト角の上限値に制限する
    # for i in range(np.size(tilt_switch)):
    #     if tilt_switch[i] == 1:
    #         tilt = np.append(tilt,tilt[i-1] + (90/4.0)*(pi/180)*time_diff[i])
    #         if tilt[i] >= GAMMA:
    #             tilt[i] = GAMMA
    #             continue
    #     elif tilt_switch[i] == -1:
    #         tilt = np.append(tilt,tilt[i-1] - (90/4.0)*(pi/180)*time_diff[i])
    #         if tilt[i] < 0.0:
    #             tilt[i] = 0
    #             continue
    #     elif tilt_switch[i] == 0:
    #         tilt = np.append(tilt,0)

    # 空力の計算
    F_x = const.MASS * (d_Va[:,0] + d_theta*Vg[:,2]) \
                        + const.MASS * const.GRA * np.sin(theta)
    F_z = const.MASS * (d_Va[:,2] - d_theta*Vg[:,0]) \
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

    # ロータ推力によるモーメント
    # ２行目は，ティルト軸が原点から微妙にずれているため追加．
    Mt = const.LEN_F*(Tf_up + Tf_down) \
        + const.LEN_TILT*(Tf_up + Tf_down)*np.sin(tilt) \
        - const.LEN_M*(Tm_up + Tm_down)*np.cos(tilt) \
        - const.LEN_R_X*(Tr_l + Tr_r)

    # 重力によるモーメント
    Lg = const.R_G_Z*const.MASS*const.GRA*np.cos(theta)*np.sin(phi)
    Mg = - const.R_G_Z*const.MASS*const.GRA*np.sin(theta) \
         - const.R_G_X*const.MASS*const.GRA*np.cos(theta)*np.cos(phi)
    Ng = const.R_G_X*const.MASS*const.GRA*np.cos(theta)*np.sin(phi)

    Ma = M - Mt - Mg

    #---------------------------
    # データのフィルタリング処理
    #---------------------------

    alpha = ffilt.fourier_filter(alpha,0.02,data_size,10)
    d_alpha = ffilt.fourier_filter(d_alpha,0.02,data_size,10)
    d_theta = ffilt.fourier_filter(d_theta,0.02,data_size,10)
    delta_e = ffilt.fourier_filter(delta_e,0.02,data_size,10)
    Va_mag = ffilt.fourier_filter(Va_mag,0.02,data_size,10)
    L = ffilt.fourier_filter(L,0.02,data_size,10)
    D = ffilt.fourier_filter(D,0.02,data_size,10)
    Ma = ffilt.fourier_filter(Ma,0.02,data_size,10)

    #---------------------------
    # ログデータから算出した空力係数
    #---------------------------

    CL_log = L / ((1/2)*RHO*(Va_mag)**2*const.S)
    CD_log = D / ((1/2)*RHO*(Va_mag)**2*const.S)
    Cm_log = Ma / ((1/2)*RHO*(Va_mag)**2*const.S*const.MAC)

    #---------------------------
    # データを一つにまとめる
    #---------------------------

    read_log_data = pd.DataFrame({
        'id' : id,
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
        # 'pitot_Va' : measurement_airspeed,
        'Tm_up' : Tm_up,
        'Tm_down' : Tm_down,
        'Tr_r' : Tr_r,
        'Tr_l' : Tr_l,
        'Tf_up' : Tf_up,
        'Tf_down' : Tf_down,
        'alpha' : alpha,
        'alpha_deg' : alpha_deg,
        'd_alpha' : d_alpha,
        'delta_e' : delta_e,
        'delta_a' : delta_a,
        'tilt' : tilt,
        'RHO' : RHO,
        'F_x' : F_x,
        'T_x' : T_x,
        'F_z' : F_z,
        'T_z' : T_z,
        'L' : L,
        'D' : D,
        # 'M' : M,
        'Mt' : Mt,
        'Mg' : Mg,
        'Ma' : Ma,
        'CL_log' : CL_log,
        'CD_log' : CD_log,
        'Cm_log' : Cm_log,
    })

    #---------------------------
    # データの整理と結合
    #---------------------------

    # 中央差分による両端を落とす
    read_log_data = read_log_data.drop(index=[0,data_size-1])

    format_log_data = pd.concat([input_log_data, read_log_data])

    return format_log_data,data_size
