# %%
import pandas as pd
import numpy as np
# Read CSV file into a pandas DataFrame
df = pd.read_csv('aruco_motion_capture_data.csv')

# Print the first 10 rows of the DataFrame
print(df.head(10))

# %%
# Compare column 3 with 4, update column 3 if conditions are met
for index, row in df.iterrows():
    if row[3] < 0 and row[4] > 200:
        df.at[index, df.columns[3]] = row[3] + 360

# Print the first 10 rows of the DataFrame
print(df.head(10))


# %%
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df)

# %%
n = 480
df['Y']=(n + 1) - df['Y']

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df)

# %%
# Define offsets for columns x, y, phi (index 1, 2, 3)
offset_x = 272
#offset_y = 234
offset_theta = 90

# Subtract offsets from respective columns
df[df.columns[1]] -= offset_x
#df[df.columns[2]] += offset_y
df[df.columns[4]] -= offset_theta

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df)

# %%
a, b = 391,0
c, d = 86, -35  # Target range [50, 100]
x, y = -34, 94
v, w = -34,62
# Apply the mapping formula: y = ((x - a) * (d - c) / (b - a)) + c
df['Y'] = ((df['Y'] - a) * (d - c) / (b - a)) + c

df['X'] = ((df['X'] - x) * (w - v) / (y - x)) + v

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df)

# %%
# Define offsets for columns x, y, phi (index 1, 2, 3)
offset_x = 8.5
# Subtract offsets from respective columns
df[df.columns[1]] += offset_x

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df)

# %%
df['X'] = df['X'].astype(int)
df['Y'] = df['Y'].astype(int)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df)
    
df.to_csv('finger_motion_data.csv',  index=False, columns=["X","Y","Phi (degrees)","Theta (pulley angle)"],  header=False,) 


