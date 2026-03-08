import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define the parameters
r = 8       # Radius of the pulley
D = -20     # Distance from the center of the joint to the pulley along the x-axis
L0 = 10     # Initial length of the cable
n_joints = 7  # Number of joints
max_angle_deg = 40  # Maximum angle for each joint in degrees

# Radius of the circles at each joint (assume equal for simplicity)
Ra = [6.0 for _ in range(n_joints)]  # Radius of the circles at the end of each link
L_links = [8.6 for _ in range(n_joints)]  # Lengths of each link

# Initial state
current_angle = 0

# Function to calculate joint angle based on total cable length
def calculate_theta(current_angle, r):
    return np.radians(current_angle)  # Convert degrees to radians

# Function to calculate the rolling angle for each joint
def calculate_phi(theta, Ra_i, r, n_joints):
    return (theta * (r / Ra_i)) / (n_joints - 1)

# Function to calculate the homogeneous transformation matrix for each joint
def calculate_transformation_matrix(phi, x, y):
    return np.array([
        [np.cos(phi), -np.sin(phi), 0, x],
        [np.sin(phi),  np.cos(phi), 0, y],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

# Function to update the plot
def update_plot():
    global current_angle
    
    theta = calculate_theta(current_angle, r)
    phis = [calculate_phi(theta, Ra_i, r, n_joints) for Ra_i in Ra]
    accumulated_phis = np.cumsum(phis)
    
    # Apply maximum angle constraint
    max_angle_rad = np.radians(max_angle_deg)
    accumulated_phis = np.clip(accumulated_phis, -max_angle_rad, max_angle_rad)
    
    # Calculate the added cable length
    added_cable_length = theta * r 
    
    # Initialize the starting position
    x, y = 0, 0
    transformations = []
    
    # Create the transformation matrices for each joint
    for i, phi in enumerate(accumulated_phis):
        T = calculate_transformation_matrix(phi, x, y)
        transformations.append(T)
        
        # Calculate the new position based on the transformation
        x += (Ra[i] + (Ra[i] if i < n_joints - 1 else 0)) * np.cos(np.sum(accumulated_phis[:i+1]))
        y += (Ra[i] + (Ra[i] if i < n_joints - 1 else 0)) * np.sin(np.sum(accumulated_phis[:i+1]))
        
    # Remove previous lines and circles
    while ax.lines:
        ax.lines[0].remove()
    while ax.patches:
        ax.patches[0].remove()
    
    # Plot each joint
    prev_x, prev_y = 0, 0
    for i, T in enumerate(transformations):
        joint_x = T[0, 3]
        joint_y = T[1, 3]
        
        # Plot the circle for the joint
        circle = plt.Circle((joint_x, joint_y), Ra[i], color='g', fill=False)
        ax.add_artist(circle)
        
        # Plot the link
        ax.plot([prev_x, joint_x], [prev_y, joint_y], 'r-')
        
        prev_x, prev_y = joint_x, joint_y
    
    # Update text with angle and cable length
    joint_angles_text = 'Joint Angles: ' + ', '.join([f"{np.degrees(phi):.2f}" for phi in accumulated_phis])
    text.set_text(joint_angles_text)
    cable_text.set_text(f'Cable Added: {added_cable_length:.2f}')
    pulley_text.set_text(f'Pulley Angle: {current_angle:.2f}°')
    
    fig.canvas.draw_idle()

# Event handler for key press
def on_key(event):
    global current_angle
    if event.key == 'left':
        current_angle += 10
    elif event.key == 'right':
        current_angle -= 10
    update_plot()

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 6))  # Increase figure size
ax.set_xlim(-100, 100)
ax.set_ylim(-25, 100)
ax.set_aspect('equal')

# Initialize text for angle
text = ax.text(-70, 80, '', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
# Initialize text for cable length
cable_text = ax.text(-70, 70, '', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
# Initialize text for pulley angle
pulley_text = ax.text(-70, 60, '', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

# Add legend and show the plot
plt.legend()
plt.title('Tendon Driven Finger Simulation')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid()

# Connect the key press event to the handler
fig.canvas.mpl_connect('key_press_event', on_key)

# Initial plot update
update_plot()

plt.show()





