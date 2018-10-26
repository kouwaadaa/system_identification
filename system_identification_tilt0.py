# -*- coding: utf-8 -*-

#---------------------------
# Import library or package
#---------------------------

import numpy as np
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
LENGTH_FROM_CENTER_TO_SUB_front = 0.496 # Center of gravity <-> Sub front
LENGTH_FROM_CENTER_TO_SUB_LEFT_RIGHT_X = 0.232 # Center of gravity <-> Sub x-axis
LENGTH_FROM_CENTER_TO_SUB_LEFT_RIGHT_Y = 0.503 # Center of gravity <-> Sub y-axis
LENGTH_FROM_CENTER_TO_PIXHAWK = 0.353 # Center of gravity <-> Pixhawk

# Other parameters
MASS = 5.7376 # Airframe weight
GRAVITY = 9.80665 # Gravity acceleration
RHO = 1.205 # Air density ρ
SURFACE_AREA = 0.2087*2 + 0.1202 # Main wing + body
MEAN_AERODYNAMIC_CHORD = 0.43081 # MAC

#---------------------------
# Read log data (CSV)
#---------------------------

# Read log data
read_log_data = pd.read_csv(filepath_or_buffer="./log_data/Book1.csv", encoding="ASCII", sep=",")

print(read_log_data.values[:,0])