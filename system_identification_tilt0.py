## import library or package
import numpy as np

## VTOL Parameter
# Moment of inertia [g/mm^2] -> [kg/m^2]
I = np.matrix('0.2484 -0.0037 -0.0078;-0.0037 0.1668 0.0005;-0.0078 0.0005 0.3804')
Ixx = I[0,0] # X-axis
Iyy = I[1,1] # Y-axis
Izz = I[2,2] # Z-axis

l1 = 0.042 # Center of gravity <-> Main
l2 = 0.496 # Center of gravity <-> Sub front
l3 = 0.232 # Center of gravity <-> Sub x-axis
l4 = 0.503 # Center of gravity <-> Sub y-axis
lp = 0.353 # Center of gravity <-> Pixhawk

