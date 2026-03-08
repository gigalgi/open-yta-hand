import numpy as np

def calculate_theta_and_s(p0, p1):
    # Calculate the chord vector H
    H = np.array(p1) - np.array(p0)
    
    # Calculate the magnitude of H
    H_magnitude = np.linalg.norm(H)
    
    # Calculate the dot product with the Z-axis unit vector (0, 0, 1)
    Z0 = np.array([0, 0, 1])
    dot_product = np.dot(Z0, H)
    
    # Calculate the bending angle theta
    theta = 2 * np.arccos(dot_product / H_magnitude)
    
    # Calculate the arc length S
    # Note: sin(theta/2) can be zero if theta is very small; handle this case
    if np.sin(theta / 2) != 0:
        S = (H_magnitude * theta) / (2 * np.sin(theta / 2))
    else:
        S = H_magnitude  # If theta is very small, arc length approximates the chord length

    return theta, S

# Example usage with p0 = (0, 0, 0) and p1 = (24, 0, 80)
p0 = (0, 0, 0)
p1 = (24, 0, 80)

theta, S = calculate_theta_and_s(p0, p1)
print(f"Bending Angle (theta): {theta} radians, or {np.degrees(theta)} degrees")
print(f"Arc Length (S): {S} units")
