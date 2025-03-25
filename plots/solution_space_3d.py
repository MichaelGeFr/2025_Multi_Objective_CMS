import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#global font parameters for plots
plt.rcParams['font.family'] = 'Times New Roman'

#parameters of the oven
m_batch = 200 #mass of batch in kg

#time relevant parameters of the heat treatment process
t_ref_h = 0.25 #reference time for heating up at full load in h
t_n = 2 #reference time for the nitriding process in h
t_ref_c = 1 #reference time for cooling at full load in h
t_shift = 8 #shift hours in h

#power related parameters
P_h = 100 #gas heater power in kJ/s
P_a = 10 #stationary gas heater power during annealing in kJ/s
P_c = 7.5 #cooling fan power in kJ/s

# Define the functions f1 and f2
def f1(x1, x2):
    return (m_batch * x1) / (t_ref_h * x1 + t_n + t_ref_c * (x1/x2**2))

def f2(x1, x2):
    return P_h * t_ref_h * x1 + P_a * t_n + P_c * t_ref_c * x1 * x2**3

# Create a grid of x1 and x2 values
x1_values = np.linspace(0.1, 1, 100)  # Avoid zero to prevent division by zero
x2_values = np.linspace(0.1, 1, 100)
X1, X2 = np.meshgrid(x1_values, x2_values)

# Calculate the values for both functions
Z1 = f1(X1, X2)
Z2 = f2(X1, X2)

# Calculate the normative values for both functions
N1 = Z1/(Z1+Z2)
N2 = Z2/(Z1+Z2)

# Plotting
fig = plt.figure(figsize=(3.36, 3.36))
ax = fig.add_subplot(111, projection='3d')

# Plot f1
ax.plot_surface(X1, X2, N1, alpha=0.7, cmap='viridis')

# Plot f2
ax.plot_surface(X1, X2, N2, alpha=0.5, cmap='plasma')

# Labels and title
ax.set_xlabel('Batch mass normalized', fontsize=10)
ax.set_ylabel('Fan speed normalized', fontsize=10)
ax.set_zlabel('energy consumption - output normalized', fontsize=10)
ax.set_title('3D plot objective space normalized', fontsize=10)
plt.savefig('3d_objective_space_normalized.pdf')
plt.show()
