import numpy as np
import matplotlib.pyplot as plt

# Data for speed of sound
x = range(0, 50)
y = [331.3 + 0.6 * (i) for i in x]

# Set up the figure with subplots (1 row, 3 columns)
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Plot speed of sound
axs[0].plot(x, y)
axs[0].set_title("Speed of sound proportional to temperature (K)")
axs[0].set_xlabel("Celsius")
axs[0].set_ylabel("m/s")

# Calculate error
v_20_celsius = 331.3 + 0.6 * 20
t_min = 0.02 / v_20_celsius
t_max = 4.0 / v_20_celsius
y_error_min = [(v - v_20_celsius) * t_min * 1000 for v in y]  # in mm
y_error_max = [(v - v_20_celsius) * t_max * 1000 for v in y]  # in mm

# Plot error over 0.02 meter
axs[1].plot(x, y_error_min)
axs[1].set_title("Error over 0.02 meter")
axs[1].set_xlabel("Celsius")
axs[1].set_ylabel("Error in (mm)")

# Plot error over 4 meters
axs[2].plot(x, y_error_max)
axs[2].set_title("Error over 4 meters")
axs[2].set_xlabel("Celsius")
axs[2].set_ylabel("Error in (mm)")

# Adjust layout for better spacing
plt.tight_layout()

# Show the plots
plt.show()
