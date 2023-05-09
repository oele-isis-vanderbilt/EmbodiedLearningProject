
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define parameters
n = 1000  # number of points
amp = 1.0  # amplitude
freq = 10.0  # frequency
z_scale = 10.0  # z-axis scaling factor

# Generate equally spaced points along a sine wave
x = np.linspace(0, 2*np.pi, n)
y = amp * np.sin(freq * x)
z = z_scale * x  # scale z-axis

# Stack points along z-axis to create a 3D spiral
xyz = np.column_stack((x, y, z))

# Plot 3D spiral
fig = plt.figure()
# ax = fig.gca(projection='3d')
ax = fig.add_subplot(111, projection='3d')
ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color='blue')
plt.show()
