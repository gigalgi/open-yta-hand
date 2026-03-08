import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset from CSV file
df = pd.read_csv('dataset/finger_trajectory_dataset.csv')  # Load your dataset

# Assuming CSV file has columns: x, y, phi, theta
data = df.values  # Convert DataFrame to NumPy array

# Split the dataset into training, validation, and testing sets
train_data, temp_data = train_test_split(data, test_size=0.15, random_state=42)
test_data = temp_data

# Output the shapes of the datasets
print(f"Training Data: {train_data.shape}")
print(f"Testing Data: {test_data.shape}")

# Save the split datasets in CSV format
train_df = pd.DataFrame(train_data, columns=df.columns)
test_df = pd.DataFrame(test_data, columns=df.columns)

# Sorting by 'column_name' in ascending order
train_df = train_df.sort_values(by='Theta', ascending=True)
test_df = test_df.sort_values(by='Theta', ascending=True)

train_df.to_csv('dataset/train_data.csv', index=False, header=False)
test_df.to_csv('dataset/test_data.csv', index=False, header=False)
