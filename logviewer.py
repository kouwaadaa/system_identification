# -*- coding: utf-8 -*-

#---------------------------
# Import library or package
#---------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
