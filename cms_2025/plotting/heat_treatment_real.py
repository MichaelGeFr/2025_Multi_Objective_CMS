import numpy as np
import matplotlib.pyplot as plt

# Define the extended process time steps and required temperatures for heating, holding, and cooling
t_process = np.array([0, 1, 5, 9, 14])  # Time steps in hours: 1h heating, 4h holding, 5h cooling
T_retort_req = np.array([20, 585, 585, 20, 20])  # Required temperature: rise, hold, cool down

# Simulation parameters for PT1 behavior
time_steps = 1500  # Increase the number of time steps for a smoother curve
t_sim = np.linspace(0, 14, time_steps)  # Extended simulated time in hours
T_sim = np.zeros_like(t_sim)  # Initialize simulated temperature array

# PT1 system parameters
time_constant = 0.75  # Time constant of the PT1 element (adjusted for fast response)
T_sim[0] = T_retort_req[0]  # Initial temperature matches the required temperature

# Simulating the PT1 response using a differential equation
for i in range(1, len(t_sim)):
    # Determine the required temperature at this time step
    req_index = np.searchsorted(t_process, t_sim[i]) - 1
    if req_index >= len(T_retort_req):
        req_index = len(T_retort_req) - 1
    T_req = T_retort_req[req_index]
    
    # Calculate the rate of temperature change (PT1 system equation)
    dT = (T_req - T_sim[i-1]) / time_constant * (t_sim[i] - t_sim[i-1])
    T_sim[i] = T_sim[i-1] + dT

# Creating enhanced plot with PT1 heating, holding, and cooling behavior
plt.figure(figsize=(12, 6))

# Step plot of the required temperature
plt.step(t_process, T_retort_req, where='mid', marker='o', linestyle='-', color='b', label='Required Temperature')

# Simulated temperature response with PT1 behavior
plt.plot(t_sim, T_sim, linestyle='--', color='r', linewidth=2, label='Simulated PT1 Response')

# Additional plot customizations
plt.title("Time-Temperature Curve with Synchronized PT1 Heating, Holding, and Cooling Behavior", fontsize=14)
plt.xlabel("Time in hours", fontsize=12)
plt.ylabel("Temperature in °C", fontsize=12)
plt.text(6.5, 520, "Gas-Nitriding Process", fontsize=10, horizontalalignment='center')
plt.legend()
plt.grid(True)
plt.show()
