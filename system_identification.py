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
font = {'family':'YuGothic'}
plt.rc('font', **font)
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

V_W = -3.0000 # Wind speed
THRUST_EF = 40/48

# Position to stop tilt
GAMMA = 90*(pi/180)

# Max thrust value of sub rotor
SUB_THRUST_MAX = 9.0

#---------------------------
# Read log data (CSV)
#---------------------------

# Read log data
read_log_data = pd.read_csv(
    filepath_or_buffer="./log_data/Book6.csv",
    encoding="ASCII",
    sep=",",
    header=None
)

#---------------------------
# Delete Time duplicate lines
#---------------------------

read_log_data = read_log_data.drop_duplicates(subset=390)

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

# Time log
time_log = np.array(read_log_data.values[:,390] / 1000000)

# Set start time to 0 second
time = np.array(time_log - time_log[0])

# Get data size (rows)
data_size = len(read_log_data)

#---------------------------
# Calculate datum
#---------------------------

# Thrust by rotor
main_up_thrust = [] # T1
main_low_thrust = [] # T2
sub_right_thrust = [] # T3
sub_left_thrust = [] # T4
sub_front_up_thrust = [] # T5
sub_front_low_thrust = [] # T6

# Calculate thrust
# From 2017/06 AS Mr.Hirai
for i in range(data_size):

  # Linear approximation
  # List append is more fast
  main_up_thrust.append(THRUST_EF*0.5*9.8*(9.5636* 10**(-3)*main_up_pwm[i] - 12.1379))
  main_low_thrust.append(THRUST_EF*0.5*9.8*(9.5636* 10**(-3)*main_low_pwm[i] - 12.1379))
  sub_right_thrust.append(9.8*(1.5701* 10**(-6) *(sub_right_pwm[i]) *1.9386))
  sub_left_thrust.append(9.8*(1.5701* 10**(-6) *(sub_left_pwm[i]) *1.9386))
  sub_front_up_thrust.append(9.8*(1.5701* 10**(-6) *(sub_front_up_pwm[i]) *1.9386))
  sub_front_low_thrust.append(9.8*(1.5701* 10**(-6) *(sub_front_low_pwm[i]) *1.9386))

# List to ndarray(numpy)
main_up_thrust = np.array(main_up_thrust)
main_low_thrust = np.array(main_low_thrust)
sub_right_thrust = np.array(sub_right_thrust)
sub_left_thrust = np.array(sub_left_thrust)
sub_front_up_thrust = np.array(sub_front_up_thrust)
sub_front_low_thrust = np.array(sub_front_low_thrust)

#Thrust limmiter
main_up_thrust[main_up_thrust < 0] = 0
main_low_thrust[main_low_thrust < 0] = 0
sub_right_thrust[sub_right_thrust > SUB_THRUST_MAX] = SUB_THRUST_MAX
sub_left_thrust[sub_left_thrust > SUB_THRUST_MAX] = SUB_THRUST_MAX
sub_front_up_thrust[sub_front_up_thrust > SUB_THRUST_MAX] = SUB_THRUST_MAX
sub_front_low_thrust[sub_front_low_thrust > SUB_THRUST_MAX] = SUB_THRUST_MAX

# Elevon sterring angle
delta_e_right = []
delta_e_left = []

# Calculate elevon steering angle
for i in range(data_size):
  delta_e_right.append(((delta_e_right_command[i]*400 + 1500)/8 - 1500/8)*pi/180)
  delta_e_left.append(((delta_e_left_command[i]*400 + 1500)/8 - 1500/8)*pi/180)

# List to ndarray(numpy)
delta_e_right = np.array(delta_e_right)
delta_e_left = np.array(delta_e_left)

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

# Translation motion
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

plt.plot(time,lift_force, label='揚力')
plt.title('揚力')
plt.xlabel('Time[]')
plt.ylabel('L[N]')
plt.legend()
plt.show()
