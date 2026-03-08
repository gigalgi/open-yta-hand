import serial
import csv
import time
import os

# Set up the serial connection (adjust 'COM3' and baud rate as needed)
arduino_port = 'COM3'  # Replace with your Arduino port, e.g., '/dev/ttyUSB0' on Linux
baud_rate = 115200
csv_file = 'arduino_data.csv'

# Open the serial port
ser = serial.Serial(arduino_port, baud_rate)
time.sleep(2)  # Give some time for the connection to establish

# Check if the file already exists
file_exists = os.path.isfile(csv_file)

# Open the CSV file in append mode if it exists, else in write mode
with open(csv_file, mode='a' if file_exists else 'w', newline='') as file:
    writer = csv.writer(file)
    
    # Write header only if file is new
    if not file_exists:
        writer.writerow(['setpoint_1', 'theta', 'setpoint_2', 'force', 'output'])

    print("Starting data logging... Press Ctrl+C to stop.")
    
    try:
        while True:
            if ser.in_waiting > 0:
                # Read a line from the serial data
                line = ser.readline().decode('ascii').strip()
                
                # Split the line by commas
                data = line.split(',')
                
                # Ensure the data is in the correct format (5 elements)
                if len(data) == 5:
                    try:
                        # Convert each value to a float to handle negative values
                        data = [float(value) for value in data]
                        
                        # Write the data to the CSV file
                        writer.writerow(data)
                        file.flush()  # Force write the data to disk
                        print(f"Logged data: {data}")
                    except ValueError:
                        print(f"Error converting data to float: {data}")
                else:
                    print(f"Unexpected data format: {data}")

    except KeyboardInterrupt:
        print("\nData logging stopped.")
    finally:
        # Close the serial connection
        ser.close()
