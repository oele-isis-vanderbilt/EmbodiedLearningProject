
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

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
data = np.column_stack((x, y, z))

# Generate noisy 3D vector data
np.random.seed(42)
# data = np.random.rand(100, 3) * 10
noise = np.random.randn(1000, 3) * 0.5
noisy_data = data + noise

# Apply Kalman filter to smooth the data
from pykalman import KalmanFilter
kf = KalmanFilter(n_dim_obs=3, n_dim_state=3)
smoothed_data = kf.em(noisy_data).smooth(noisy_data)[0]

# Apply Savitzky-Golay filter to smooth the data
sg_data = savgol_filter(noisy_data, window_length=3, polyorder=2)

# Plot the noisy, Kalman-filtered, and Savitzky-Golay-filtered data
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
# ax.plot(noisy_data[:, 0], noisy_data[:, 1], noisy_data[:, 2], '.', color='red', label='Noisy data')
ax.plot(smoothed_data[:, 0], smoothed_data[:, 1], smoothed_data[:, 2], color='green', label='Kalman smoothed')
# ax.plot(sg_data[:, 0], sg_data[:, 1], sg_data[:, 2], color='blue', label='Savitzky-Golay smoothed')
ax.legend(loc='upper left')
plt.show()
