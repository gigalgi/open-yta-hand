import numpy as np
from scipy.optimize import minimize

# Define the segment lengths (L1 to L6)
L = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])   # Lengths of each segment

# Desired end-effector position (x_d, y_d)
x_d = 39.25
y_d = 39.45

# Function to calculate the forward kinematics (end-effector position)
def piecewise_forward_kinematics(joint_angles, segment_lengths):
    x = 0.0
    y = 0.0
    theta = 0.0
    
    for i in range(len(joint_angles)):
        theta += joint_angles[i]
        x += segment_lengths[i] * np.cos(theta)
        y += segment_lengths[i] * np.sin(theta)
    
    return x, y

# Cost function to minimize the difference between desired and calculated positions
def cost_function(joint_angles, segment_lengths, x_d, y_d):
    x, y = piecewise_forward_kinematics(joint_angles, segment_lengths)
    return (x - x_d)**2 + (y - y_d)**2  # Squared error between desired and actual positions

# Initial guess for joint angles (you can start with zeros or small angles)
initial_angles = np.zeros(len(L))

# Perform optimization to find the joint angles that minimize the cost function
result = minimize(cost_function, initial_angles, args=(L, x_d, y_d), method='BFGS')

# Extract the optimized joint angles
optimized_joint_angles = result.x

# Calculate the final position using the optimized joint angles
final_position = piecewise_forward_kinematics(optimized_joint_angles, L)

# Output the results
print("Desired end-effector position (x_d, y_d):", (x_d, y_d))
print("Calculated end-effector position (x, y):", final_position)
print("Optimized joint angles (radians):", optimized_joint_angles)
