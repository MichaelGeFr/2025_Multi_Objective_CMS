import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create grid and compute F = X1^2 + X2^2
x1 = np.linspace(-5, 5, 100) #creates a grid of values for x1 between -5 and 5
x2 = np.linspace(-5, 5, 100)
X1, X2 = np.meshgrid(x1, x2) #creates a 2D grid for both variables
F = X1**2 + X2**2 #computes the value of F at each point on the grid

# Create a figure and a 3D Axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X1, X2, F, cmap='viridis') #plots the 3D surface with a colormap (viridis in this case).

# Labels and title
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('F(X1, X2)')
ax.set_title('3D plot of F = X1^2 + X2^2')

# Show the plot
plt.show()
