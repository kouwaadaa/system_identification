#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 16:36:52 2018

@author: cs22Mac
"""

#---------------------------
# Import library or package
#---------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import math_extention as matex

from numpy import pi

#---------------------------
# Aircraft values
#---------------------------

# Moment of inertia [g/mm^2] -> [kg/m^2]
INERTIA_MOMENTS = np.array(
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
RHO = 1.205 # Air density Ď
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
dphi = np.array(read_log_data.values[:,3])
dtheta = np.array(read_log_data.values[:,4])
dpsi = np.array(read_log_data.values[:,5])

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
#
#---------------------------

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
        matex.ned2bc(phi[i],theta[i],0,WIND_SPEED,0,0)
    )

# List to ndarray
body_frame_velocity = np.array(body_frame_velocity)
body_frame_wind_velocity = np.array(body_frame_wind_velocity)

# Convert pixhawk position to center
body_frame_velocity[:,2] = body_frame_velocity[:,2] + dtheta*LENGTH_FROM_CENTER_TO_PIXHAWK

# # 差分関数の確認
# size = np.size(dot_x_position)
# diff_x = np.insert(dot_x_position,[0,0],[0,0])
# diff_x = np.delete(diff_x,[size,size+1])
# diff = dot_x_position - diff_x
# diff = np.delete(diff,[0,1])

body_frame_acceleration = np.array(
	matex.central_diff(body_frame_velocity[:,0],time)
)

body_frame_acceleration = np.append(body_frame_acceleration,matex.central_diff(body_frame_velocity[:,1],time))
body_frame_acceleration = np.append(body_frame_acceleration,matex.central_diff(body_frame_velocity[:,2],time))
body_frame_acceleration =  body_frame_acceleration.reshape(data_size-2, 3)

# dot_xyz_position = np.concatenate([dot_x_position,dot_y_position,dot_z_position], axis=1)

# Velocity
# ned_velocity = []
#
# for i in range(data_size):
# 	ned_velocity.append(math_extention.bc2ned(phi[i],theta[i],psi[i],dot_x_position[i],dot_y_position[i],dot_z_position[i]))
#
# ned_velocity = np.array(ned_velocity)
#
# print(ned_velocity)
