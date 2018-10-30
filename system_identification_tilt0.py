# -*- coding: utf-8 -*-

#---------------------------
# Import library or package
#---------------------------

import numpy as np
from numpy import pi
import pymap3d as pm
import pandas as pd

#---------------------------
# Aircraft values
#---------------------------

# Moment of inertia [g/mm^2] -> [kg/m^2]
INERTIA_MOMENTS = np.matrix(
        [[ 0.2484,-0.0037,-0.0078],
         [-0.0037, 0.1668, 0.0005],
         [-0.0078, 0.0005, 0.3804]]
        )
INERTIA_MOMENT_XX = INERTIA_MOMENTS[0,0] # X-axis
INERTIA_MOMENT_YY = INERTIA_MOMENTS[1,1] # Y-axis
INERTIA_MOMENT_ZZ = INERTIA_MOMENTS[2,2] # Z-axis

# Length [m]
LENGTH_FROM_CENTER_TO_MAIN = 0.042 # Center of gravity <-> Main
LENGTH_FROM_CENTER_TO_SUB_FRONT = 0.496 # Center of gravity <-> Sub front
LENGTH_FROM_CENTER_TO_SUB_LEFT_RIGHT_X = 0.232 # Center of gravity <-> Sub x-axis
LENGTH_FROM_CENTER_TO_SUB_LEFT_RIGHT_Y = 0.503 # Center of gravity <-> Sub y-axis
LENGTH_FROM_CENTER_TO_PIXHAWK = 0.353 # Center of gravity <-> Pixhawk

# Other values
MASS = 5.7376 # Airframe weight
GRAVITY = 9.80665 # Gravity acceleration
RHO = 1.205 # Air density ρ
SURFACE_AREA = 0.2087*2 + 0.1202 # Main wing + body
MEAN_AERODYNAMIC_CHORD = 0.43081 # MAC

WIND_SPEED = -3.0000
THRUST_EFFICIENCY = 40/48

# Max thrust value of sub rotor
SUB_THRUST_MAX = 9.0

#---------------------------
# Read log data (CSV)
#---------------------------

# Read log data
read_log_data = pd.read_csv(filepath_or_buffer="./log_data/Book6.csv", encoding="ASCII", sep=",")

# Angle
phi = np.array(read_log_data.values[:,0])
theta = np.array(read_log_data.values[:,1])
psi = np.array(read_log_data.values[:,2])

# Angular velocity
dot_phi = np.array(read_log_data.values[:,3])
dot_theta = np.array(read_log_data.values[:,4])
dot_psi = np.array(read_log_data.values[:,5])

# Position
x_position = np.array(read_log_data.values[:,53])
y_position = np.array(read_log_data.values[:,54])
z_position = np.array(read_log_data.values[:,55])

# Velocity
dot_x_position = np.array(read_log_data.values[:,58])
dot_y_position = np.array(read_log_data.values[:,59])
dot_z_position = np.array(read_log_data.values[:,60])

# GPS altitude
gps_altitude = np.array(read_log_data.values[:,79])

# Airspeed by Pitot tube
airspeed = np.array(read_log_data.values[:,133])

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
# Caliculate datum
#---------------------------
# Thrust by rotor
main_up_thrust = [] # T1
main_low_thrust = [] # T2
sub_right_thrust = [] # T3
sub_left_thrust = [] # T4
sub_front_up_thrust = [] # T5
sub_front_low_thrust = [] # T6

# Caliculate thrust
# From 2017/06 AS Mr.Hirai
for i in range(data_size):

  # Linear approximation
  # List append is more fast
  main_up_thrust.append(THRUST_EFFICIENCY*0.5*9.8*(9.5636* 10**(-3)*main_up_pwm[i] - 12.1379))
  main_low_thrust.append(THRUST_EFFICIENCY*0.5*9.8*(9.5636* 10**(-3)*main_low_pwm[i] - 12.1379))
  sub_right_thrust.append(9.8*(1.5701* 10**(-6) *(sub_right_pwm) *1.9386))
  sub_left_thrust.append(9.8*(1.5701* 10**(-6) *(sub_left_pwm) *1.9386))
  sub_front_up_thrust.append(9.8*(1.5701* 10**(-6) *(sub_front_up_pwm) *1.9386))
  sub_front_low_thrust.append(9.8*(1.5701* 10**(-6) *(sub_front_low_pwm) *1.9386))

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

# Caliculate elevon steering angle
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
pixhawk_ground_velocity = []
pixhawk_body_frame_velocity = []
body_frame_velocity = [] # pixhawk -> center
body_frame_wind_velocity = []

# Caliculate velocity
pixhawk_ground_velocity = np.sqrt(dot_x_position**2 + dot_y_position**2 + dot_z_position**2)
pixhawk_body_frame_velocity =
