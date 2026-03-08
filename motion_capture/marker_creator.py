import cv2
import numpy as np

# Load the ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()

# Function to generate and save an ArUco marker
def create_aruco_marker(marker_id, marker_size=200, output_filename="aruco_marker.png"):
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
    cv2.imwrite(output_filename, marker_image)
    print(f"Aruco marker with ID {marker_id} saved as {output_filename}")

# Example: Create an ArUco marker with ID 1
create_aruco_marker(marker_id=2)