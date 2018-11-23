# -*- coding: utf-8 -*-

#---------------------------
# Import library or package
#---------------------------

import numpy as np
from numpy import pi
import pandas as pd

import matplotlib.pyplot as plt
# import matplotlib.font_manager
from IPython import get_ipython

import math_extention as matex

#---------------------------
# Setting matplotlib
#---------------------------

# Plot out of Window
get_ipython().run_line_magic('matplotlib', 'qt')

# Set font
# print([f.name for f in matplotlib.font_manager.fontManager.ttflist])
plt.rc('font', **{'family':'YuGothic'})
plt.rcParams['font.size'] = 20
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15 # default: 12

# Set figure
plt.rcParams["figure.figsize"] = [20, 12]

#---------------------------
# Aircraft values
#---------------------------

# Moment of inertia [g/mm^2] -> [kg/m^2]
I = np.array(
    [[ 0.2484,-0.0037,-0.0078],
     [-0.0037, 0.1668, 0.0005],
     [-0.0078, 0.0005, 0.3804]]
)
I_XX = I[0,0] # X-axis
I_YY = I[1,1] # Y-axis
I_ZZ = I[2,2] # Z-axis

# Length [m]
LEN_M = 0.042 # Center of gravity <-> Main
LEN_F = 0.496 # Center of gravity <-> Sub front
LEN_S_X = 0.232 # Center of gravity <-> Sub x-axis
LEN_S_Y = 0.503 # Center of gravity <-> Sub y-axis
LEN_P = 0.353 # Center of gravity <-> Pixhawk

# Other values
MASS = 5.7376 # Airframe weight
GRA = 9.80665 # Gravity acceleration
RHO = 1.205 # Air density
S = 0.2087*2 + 0.1202 # Main wing + body
MAC = 0.43081 # MAC

# Position to stop tilt
# GAMMA = 90*(pi/180)

# Max thrust value of sub rotor
SUB_THRUST_MAX = 9.0

#---------------------------
# Read each log data
#---------------------------

# File number
FILE_NUM = 6
for file_number in range(FILE_NUM):

    #---------------------------
    # Read log data (CSV) -> Format log data
    #---------------------------

    if file_number == 0:
        # Read log data
        read_log_data = pd.read_csv(
            filepath_or_buffer='./log_data/Book3.csv',
            encoding='ASCII',
            sep=',',
            header=None
        )

        # Delete Time duplicate lines
        read_log_data = read_log_data.drop_duplicates(subset=390)

        # Convert "time"
        read_log_data['Time_ST'] = read_log_data.at[0,390]
        read_log_data['Time_Conv'] = (read_log_data[390] - read_log_data['Time_ST'])/1000000

        # Cut time band
        read_log_data = read_log_data.query(
            '17.52 <= Time_Conv <= 19.14'
        )

        # Append data (windspeed, thrust efficiency, gamma(tilt angle))
        V_W = -4.03
        THRUST_EF = 40/48
        GAMMA = 0

    elif file_number == 1:
        # Read log data
        read_log_data = pd.read_csv(
            filepath_or_buffer='./log_data/Book4.csv',
            encoding='ASCII',
            sep=',',
            header=None
        )

        # Delete Time duplicate lines
        read_log_data = read_log_data.drop_duplicates(subset=390)

        # Convert "time"
        read_log_data['Time_ST'] = read_log_data.at[0,390]
        read_log_data['Time_Conv'] = (read_log_data[390] - read_log_data['Time_ST'])/1000000

        # Cut time band
        read_log_data = read_log_data.query(
            '11.97 <= Time_Conv <= 13.30 \
            | 18.66 <= Time_Conv <= 21.08'
        )

        # Insert data(windspeed, thrust efficiency, gamma(tilt angle))
        V_W = -5.05
        THRUST_EF = 40/45
        GAMMA = 0

    elif file_number == 2:
        # Read log data
        read_log_data = pd.read_csv(
            filepath_or_buffer='./log_data/Book5.csv',
            encoding='ASCII',
            sep=',',
            header=None
        )

        # Delete Time duplicate lines
        read_log_data = read_log_data.drop_duplicates(subset=390)

        # Convert "time"
        read_log_data['Time_ST'] = read_log_data.at[0,390]
        read_log_data['Time_Conv'] = (read_log_data[390] - read_log_data['Time_ST'])/1000000

        # Cut time band
        read_log_data = read_log_data.query(
            '12.45 <= Time_Conv <= 13.66 \
            | 16.07 <= Time_Conv <= 17.03 \
            | 18.95 <= Time_Conv <= 22.88'
        )

        # Insert data(windspeed, thrust efficiency, gamma(tilt angle))
        V_W = -4.80
        THRUST_EF = 40/48
        GAMMA = 0

    elif file_number == 3:
        # Read log data
        read_log_data = pd.read_csv(
            filepath_or_buffer='./log_data/Book8.csv',
            encoding='ASCII',
            sep=',',
            header=None
        )

        # Delete Time duplicate lines
        read_log_data = read_log_data.drop_duplicates(subset=390)

        # Convert "time"
        read_log_data['Time_ST'] = read_log_data.at[0,390]
        read_log_data['Time_Conv'] = (read_log_data[390] - read_log_data['Time_ST'])/1000000

        # Cut time band
        read_log_data = read_log_data.query(
            '15.41 <= Time_Conv <= 20.10 \
            | 21.46 <= Time_Conv <= 23.07 \
            | 23.44 <= Time_Conv <= 24.64 \
            | 25.28 <= Time_Conv <= 27.38'
        )

        # Insert data(windspeed, thrust efficiency, gamma(tilt angle))
        V_W = -2.0
        THRUST_EF = 40/47
        GAMMA = 0

    elif file_number == 4:
        # Read log data
        read_log_data = pd.read_csv(
            filepath_or_buffer='./log_data/Book9.csv',
            encoding='ASCII',
            sep=',',
            header=None
        )

        # Delete Time duplicate lines
        read_log_data = read_log_data.drop_duplicates(subset=390)

        # Convert "time"
        read_log_data['Time_ST'] = read_log_data.at[0,390]
        read_log_data['Time_Conv'] = (read_log_data[390] - read_log_data['Time_ST'])/1000000

        # Cut time band
        read_log_data = read_log_data.query(
            '20.73 <= Time_Conv <= 30.28 \
            | 98.05 <= Time_Conv <= 104.1 \
            | 104.9 <= Time_Conv <= 107.1 \
            | 107.7 <= Time_Conv <= 109.7'
        )

        # Insert data(windspeed, thrust efficiency, gamma(tilt angle))
        V_W = -2.647
        THRUST_EF = 40/48
        GAMMA = 0

    elif file_number == 5:
        # Read log data
        read_log_data = pd.read_csv(
            filepath_or_buffer='./log_data/Book11.csv',
            encoding='ASCII',
            sep=',',
            header=None
        )

        # Delete Time duplicate lines
        read_log_data = read_log_data.drop_duplicates(subset=390)

        # Convert "time"
        read_log_data['Time_ST'] = read_log_data.at[0,390]
        read_log_data['Time_Conv'] = (read_log_data[390] - read_log_data['Time_ST'])/1000000

        # Cut time band
        read_log_data = read_log_data.query(
            '19.86 <= Time_Conv <= 25.27 \
            | 26.43 <= Time_Conv <= 29.83'
        )

        # Insert data(windspeed, thrust efficiency, gamma(tilt angle))
        V_W = -1.467
        THRUST_EF = 40/48
        GAMMA = 0

    #---------------------------
    # Assign log data
    #---------------------------

    # Angle
    phi = np.array(read_log_data.values[:,0])
    theta = np.array(read_log_data.values[:,1])
    psi = np.array(read_log_data.values[:,2])

    # Angular velocity
    d_phi = np.array(read_log_data.values[:,3])
    d_theta = np.array(read_log_data.values[:,4])
    d_psi = np.array(read_log_data.values[:,5])

    # Position
    x_position = np.array(read_log_data.values[:,53])
    y_position = np.array(read_log_data.values[:,54])
    z_position = np.array(read_log_data.values[:,55])

    # Velocity
    dx_position = np.array(read_log_data.values[:,58])
    dy_position = np.array(read_log_data.values[:,59])
    dz_position = np.array(read_log_data.values[:,60])

    # GPS altitude
    gps_altitude = np.array(read_log_data.values[:,79])

    # Airspeed by Pitot tube
    measurement_airspeed = np.array(read_log_data.values[:,133])

    # Pulese Width Modulation of rotors
    main_up_pwm = np.array(read_log_data.values[:,116]) # T1
    main_low_pwm = np.array(read_log_data.values[:,117]) # T2
    sub_right_pwm = np.array(read_log_data.values[:,118]) # T3
    sub_left_pwm = np.array(read_log_data.values[:,119]) # T4
    sub_front_up_pwm = np.array(read_log_data.values[:,120]) # T5
    sub_front_low_pwm = np.array(read_log_data.values[:,121]) # T6

    # Elevon steering angle (command 0 ~ 1)
    delta_e_right_command = np.array(read_log_data.values[:,124])
    delta_e_left_command = np.array(read_log_data.values[:,125])

    # Manual manipulation quantity
    manual_pitch = np.array(read_log_data.values[:,374])
    manual_thrust = np.array(read_log_data.values[:,377])
    manual_tilt = np.array(read_log_data.values[:,389])

    # Time
    time = np.array(read_log_data.values[:,392])

    # Get data size (rows)
    data_size = len(read_log_data)

    #---------------------------
    # Calculate datum
    #---------------------------

    # Rotor thrust
    Tm_up = THRUST_EF*0.5*GRA*(9.5636* 10**(-3)*main_up_pwm - 12.1379)
    Tm_down = THRUST_EF*0.5*GRA*(9.5636* 10**(-3)*main_low_pwm - 12.1379)
    Ts_r = GRA*(1.5701* 10**(-6) *(sub_right_pwm) *1.9386)
    Ts_l = GRA*(1.5701* 10**(-6) *(sub_left_pwm) *1.9386)
    Tf_up = GRA*(1.5701* 10**(-6) *(sub_front_up_pwm) *1.9386)
    Tf_down = GRA*(1.5701* 10**(-6) *(sub_front_low_pwm) *1.9386)

    # Thrust limmiter
    Tm_up[Tm_up < 0] = 0
    Tm_down[Tm_down < 0] = 0
    Ts_r[Ts_r > SUB_THRUST_MAX] = SUB_THRUST_MAX
    Ts_l[Ts_l > SUB_THRUST_MAX] = SUB_THRUST_MAX
    Tf_up[Tf_up > SUB_THRUST_MAX] = SUB_THRUST_MAX
    Tf_down[Tf_down > SUB_THRUST_MAX] = SUB_THRUST_MAX

    # Elevon sterring angle
    delta_e_right = (delta_e_right_command*400 + 1500)/8 - 1500/8)*pi/180
    delta_e_left = (delta_e_left_command*400 + 1500)/8 - 1500/8)*pi/180

    # Elevon -> elevator & aileron
    elevator = (delta_e_left - delta_e_right)/2
    aileron = (delta_e_left + delta_e_right)/2

    # Velocity
    pixhawk_groundspeed = []
    body_frame_velocity = []
    body_frame_wind_velocity = []
    body_frame_airspeed = []

    # Calculate velocity
    pixhawk_groundspeed = np.sqrt(
        dx_position**2
        + dy_position**2
        + dz_position**2
    )

    # Convert NED frame to body frame
    for i in range(data_size):
        body_frame_velocity.append(
            matex.ned2bc(phi[i],theta[i],psi[i],dx_position[i],dy_position[i],dz_position[i])
        )
        body_frame_wind_velocity.append(
            matex.ned2bc(phi[i],theta[i],0,V_W,0,0)
        )

    # List to ndarray
    body_frame_velocity = np.array(body_frame_velocity)
    body_frame_wind_velocity = np.array(body_frame_wind_velocity)

    # Convert pixhawk position to center
    body_frame_velocity[:,2] = body_frame_velocity[:,2] + d_theta*LEN_P

    # Calculate airspeed
    body_frame_airspeed = body_frame_velocity - body_frame_wind_velocity
    body_frame_airspeed_mag = np.sqrt(
        body_frame_airspeed[:,0]**2
        + body_frame_airspeed[:,1]**2
        + body_frame_airspeed[:,2]**2
    )

    # Plot
    # plt.plot(time,body_frame_airspeed_mag)
    # plt.plot(time,measurement_airspeed)
    # plt.show()

    # Calculate angle of attack, rad
    alpha = np.arctan2(body_frame_airspeed[:,2],body_frame_airspeed[:,0])

    # Calculate time difference
    time_diff = np.diff(time)
    time_diff = np.append(time_diff, time_diff[data_size-2]) # Append the last value

    # Acceleration
    body_frame_acceleration_list = []
    dd_phi = []
    dd_theta = []
    dd_psi = []

    # Calculate acceleration
    body_frame_acceleration_list = np.array(
    	matex.central_diff(body_frame_velocity[:,0],time)
    ) # x axis
    body_frame_acceleration_list = np.append(
        body_frame_acceleration_list,
        matex.central_diff(body_frame_velocity[:,1],time)
    ) # y axis
    body_frame_acceleration_list = np.append(
        body_frame_acceleration_list,
        matex.central_diff(body_frame_velocity[:,2],time)
    ) # z axis
    body_frame_acceleration =  np.reshape(body_frame_acceleration_list,(data_size,3),order='F') # Unit

    dd_phi = matex.central_diff(d_phi,time)
    dd_theta = matex.central_diff(d_theta,time)
    dd_psi = matex.central_diff(d_psi,time)

    # Tilt
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

    tilt_switch = np.append(tilt_switch,tilt_switch[data_size-2]) # Append the last value

    tilt = []

    for i in range(np.size(tilt_switch)):
        if tilt_switch[i] == 1:
            tilt = np.append(tilt,tilt[i-1] + (90/4.0)*time_diff[i])
            if tilt[i] >= GAMMA:
                tilt[i] = GAMMA
                continue
        elif tilt_switch[i] == -1:
            tilt = np.append(tilt,tilt[i-1] - (90/4.0)*time_diff[i])
            if tilt[i] < 0.0:
                tilt[i] = 0
                continue
        elif tilt_switch[i] == 0:
            tilt = np.append(tilt,0)

    # Caliculate L and D
    body_translation_x = MASS * (body_frame_acceleration[:,0] + d_theta*body_frame_velocity[:,2]) \
                        + MASS * GRA * np.sin(theta)
    body_translation_z = MASS * (body_frame_acceleration[:,2] - d_theta*body_frame_velocity[:,0]) \
                        + MASS * GRA * np.cos(theta)
    rotor_translation_x = (main_up_pwm + main_low_pwm) * np.sin(tilt)
    rotor_translation_z = - ((main_up_pwm + main_low_pwm) * np.cos(tilt) \
                          - (sub_right_pwm + sub_left_pwm + sub_front_up_pwm + sub_front_low_pwm))
    translation_x = body_translation_x - rotor_translation_x
    translation_z = body_translation_z - rotor_translation_z
    lift_force = translation_x * np.sin(alpha) - translation_z * np.cos(alpha)
    drag_force = - translation_x * np.cos(alpha) - translation_z * np.sin(alpha)

    # Caliculate moment
    # M = I_YY * dd_theta # Moment of all-axis
    # tau = LEN_F*()
