# -*- coding: utf-8 -*-
## import library or package
import numpy as np

#---------------------------
# VTOL Parameters
#---------------------------

# Moment of inertia [g/mm^2] -> [kg/m^2]
inertia_moments = np.matrix([[ 0.2484,-0.0037,-0.0078],
                             [-0.0037, 0.1668, 0.0005],
                             [-0.0078, 0.0005, 0.3804]])
inertia_moment_Ixx = inertia_moments[0,0] # X-axis
inertia_moment_Iyy = inertia_moments[1,1] # Y-axis
inertia_moment_Izz = inertia_moments[2,2] # Z-axis

# Length [m]
length_from_center_to_main = 0.042 # Center of gravity <-> Main
length_from_center_to_sub_front = 0.496 # Center of gravity <-> Sub front
length_from_center_to_sub_left_right_x = 0.232 # Center of gravity <-> Sub x-axis
length_from_center_to_sub_left_right_y = 0.503 # Center of gravity <-> Sub y-axis
length_from_center_to_pixhawk = 0.353 # Center of gravity <-> Pixhawk

# Other parameters
mass = 5.7376 # Airframe weight
gravity = 9.80665 # Gravity acceleration
rho = 1.205 # Air density ρ
surface_area = 0.2087*2 + 0.1202 # Main wing + body
mean_aerodynamic_chord = 0.43081 # MAC

#---------------------------
# 
#---------------------------
