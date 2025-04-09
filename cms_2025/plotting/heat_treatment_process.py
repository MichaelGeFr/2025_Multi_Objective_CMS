import numpy as np
import matplotlib.pyplot as plt

# Provided data for the process time steps and required temperatures
t_process = np.array([0, 1, 2, 3])  # Process time steps in h
T_retort_req = np.array([20, 585, 585, 90])  # Required temperature in °C

#global font parameters for plots
plt.rcParams['font.family'] = 'Times New Roman'

# Creating plot of temperature-time-curve without interpolation
plt.figure(figsize=(3.36,3.36))
plt.step(t_process, T_retort_req, where='pre', marker='o', linestyle='-', color='b')
plt.title("Time-temperature curve", fontsize=10)
plt.xlabel("Time in hours", fontsize=10)
plt.ylabel("Temperature in °C", fontsize=10)
plt.grid(True)
plt.xlim(min(t_process), max(t_process))  # Set x-axis limits
plt.ylim(min(T_retort_req) - 20, max(T_retort_req) + 20)  # Set y-axis limits with some padding
plt.savefig('objective_space.pdf', bbox_inches='tight')
plt.show()
