import cv2
import numpy as np

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

    if ids is not None:
        for i, corner in enumerate(corners):
            # Get the center of the marker
            center = np.mean(corner[0], axis=0).astype(int)

            # Calculate orientation
            top_right = corner[0][0]
            top_left = corner[0][3]
            orientation = np.arctan2(top_left[1] - top_right[1], top_left[0] - top_right[0])
            orientation_deg = np.degrees(orientation)

            # Draw the center point
            cv2.circle(image, tuple(center), 5, (0, 255, 0), -1)

            # Draw the orientation line
            end_point = (int(center[0] + 50 * np.cos(orientation)), int(center[1] + 50 * np.sin(orientation)))
            cv2.line(image, tuple(center), end_point, (255, 0, 0), 2)

            # Print marker ID, 2D position, and orientation
            print(f"Marker ID: {ids[i][0]}")
            print(f"2D Position: X: {center[0]}, Y: {center[1]}")
            print(f"Orientation: {orientation_deg:.2f} degrees")
            print("--------------------")

    return image

# Initialize video capture (use 0 for webcam or provide video file path)
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect and draw markers
    output = detect_aruco_markers_2d(frame)

    # Display the result
    cv2.imshow('ArUco Marker 2D Detection', output)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()