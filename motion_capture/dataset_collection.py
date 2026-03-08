import cv2
import numpy as np
import pandas as pd
import serial
import time
import threading

# Initialize video capture (use 0 for webcam or provide video file path)
cap = cv2.VideoCapture(1)

# Initialize serial communication (adjust the port name and baud rate as needed)
ser = serial.Serial('COM5', 115200, timeout=1)
time.sleep(2)  # Allow some time for the serial connection to initialize

# Define a DataFrame to store the collected data
columns = ['Marker ID', 'X', 'Y', 'Phi (degrees)', 'Theta (pulley angle)']
data = pd.DataFrame(columns=columns)

# Global variable to store the latest theta value
theta = None

def read_serial():
    global theta
    while True:
        if ser.in_waiting > 0:
            try:
                theta = int(float(ser.readline().decode('utf-8').strip()))
            except ValueError:
                theta = None

# Start a separate thread for reading the serial port
serial_thread = threading.Thread(target=read_serial, daemon=True)
serial_thread.start()

def detect_aruco_markers_2d(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define the ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

    # Define the ArUco parameters
    parameters = cv2.aruco.DetectorParameters()

    # Create ArUco detector
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # Detect ArUco markers
    corners, ids, rejected = detector.detectMarkers(gray)

    markers = []
    if ids is not None:
        for i, corner in enumerate(corners):
            # Get the center of the marker
            center = np.mean(corner[0], axis=0).astype(int)

            # Calculate orientation
            top_right = corner[0][0]
            top_left = corner[0][3]
            orientation = np.arctan2(top_left[1] - top_right[1], top_left[0] - top_right[0])
            orientation_deg = int(np.degrees(orientation))

            # Draw the center point
            cv2.circle(image, tuple(center), 5, (0, 255, 0), -1)

            # Draw the orientation line
            end_point = (int(center[0] + 50 * np.cos(orientation)), int(center[1] + 50 * np.sin(orientation)))
            cv2.line(image, tuple(center), end_point, (255, 0, 0), 2)

            if theta is not None:
                # Print marker ID, 2D position, orientation, and pulley angle
                print(f"Marker ID: {ids[i][0]}")
                print(f"2D Position: X: {center[0]}, Y: {center[1]}")
                print(f"Orientation: {orientation_deg:.2f} degrees")
                print(f"Theta (pulley angle): {theta:.2f}")
                print("--------------------")

                markers.append((ids[i][0], center[0], center[1], orientation_deg, theta))

    return image, markers

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect and draw markers
    output, markers = detect_aruco_markers_2d(frame)

    # Add detected markers to the DataFrame
    for marker in markers:
        marker_id, x, y, phi, theta_value = marker
        new_row = {'Marker ID': marker_id, 'X': x, 'Y': y, 'Phi (degrees)': phi, 'Theta (pulley angle)': theta_value}
        data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)

    # Display the result
    cv2.imshow('ArUco Marker 2D Detection', output)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the collected data to a CSV file
data.to_csv('aruco_motion_capture_data.csv', index=False)

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
ser.close()
