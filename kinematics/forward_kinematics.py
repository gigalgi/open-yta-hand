
import numpy as np

# Define the segment lengths (L1 to L6)
L = np.array([8.6, 8.6, 8.6, 8.6, 8.6, 8.6]) # Lengths of each segment in mm

# Desired end-effector orientation
theta_d = np.radians(45)  # Desired orientation (radians)

# Calculate the joint angles assuming equal distribution of total bend angle
n = len(L)  # Number of joints
theta_i = theta_d / n  # Angle at each joint

# Calculate the final end-effector position (x_d, y_d)
x_d = np.sum([L[i] * np.cos((i + 1) * theta_i) for i in range(n)])
y_d = np.sum([L[i] * np.sin((i + 1) * theta_i) for i in range(n)])

# Output the results
print("Joint angles (radians):", np.array(np.degrees([theta_i] * n)))
print("Estimated end-effector position (x, y):", (x_d, y_d))
print("Desired end-effector orientation (theta):", np.degrees(theta_d))