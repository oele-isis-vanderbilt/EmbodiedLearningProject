import numpy as np
from filterpy.kalman import KalmanFilter

# Generate a simple 3D trajectory with some noise
n = 100  # Number of data points
x_true = np.linspace(0, 1, n)
y_true = np.sin(2 * np.pi * x_true)
z_true = np.zeros_like(x_true)
position_true = np.vstack((x_true, y_true, z_true))
position_noisy = position_true + np.random.normal(size=(3, n), scale=0.1)


# Define the Kalman filter
kf = KalmanFilter(dim_x=6, dim_z=3)

# Define the state transition matrix (constant velocity model)
dt = 1  # Time step (assume constant)
kf.F = np.array([[1, 0, 0, dt, 0, 0],
                 [0, 1, 0, 0, dt, 0],
                 [0, 0, 1, 0, 0, dt],
                 [0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 1]])

# Define the measurement matrix (observe position only)
kf.H = np.array([[1, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0]])

# Set the initial state and covariance
kf.x = np.zeros((6, 1))
kf.P = np.diag([0.1, 0.1, 0.1, 1, 1, 1])


# Apply the Kalman filter to the noisy position data
position_filtered = np.zeros_like(position_noisy)
for i in range(n):
    kf.predict()
    kf.update(position_noisy[:, i].reshape((3, 1)))
    position_filtered[:, i] = kf.x[:3, 0]

import matplotlib.pyplot as plt

# Plot the results
fig, ax = plt.subplots()
ax.plot(x_true, y_true, label='True position')
ax.plot(x_true, position_noisy[1, :], '.', label='Noisy position')
ax.plot(x_true, position_filtered[1, :], label='Filtered position')
ax.legend()
plt.show()
