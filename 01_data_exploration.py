import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2.1: Load the data
# The data has no headers, so we'll add them
columns = ['unit_id', 'time_cycles', 'setting_1', 'setting_2', 'setting_3']
columns += [f'sensor_{i}' for i in range(1, 22)]  # 21 sensors

# Load training data
train_df = pd.read_csv('data/train_FD001.txt', sep=' ', header=None, 
                        names=columns, index_col=False)

# Remove extra columns (the file has trailing spaces that create NaN columns)
train_df = train_df.dropna(axis=1)

print("Dataset Shape:", train_df.shape)
print("\nFirst 5 rows:")
print(train_df.head())

print("\nDataset Info:")
print(train_df.info())

# Step 2.2: Understand the data structure
print("\n=== Data Structure ===")
print(f"Number of unique engines: {train_df['unit_id'].nunique()}")
print(f"Total number of measurements: {len(train_df)}")

# Check one engine's lifecycle
engine_1 = train_df[train_df['unit_id'] == 1]
print(f"\nEngine 1 ran for {engine_1['time_cycles'].max()} cycles before failure")

# Step 2.3: Visualize engine degradation
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot first 4 engines' lifecycles
for i, engine_id in enumerate([1, 2, 3, 4]):
    ax = axes[i//2, i%2]
    engine_data = train_df[train_df['unit_id'] == engine_id]
    
    # Plot a few sensors
    ax.plot(engine_data['time_cycles'], engine_data['sensor_2'], label='Sensor 2')
    ax.plot(engine_data['time_cycles'], engine_data['sensor_3'], label='Sensor 3')
    ax.plot(engine_data['time_cycles'], engine_data['sensor_4'], label='Sensor 4')
    ax.set_title(f'Engine {engine_id} - Sensor Readings Over Time')
    ax.set_xlabel('Time Cycles')
    ax.set_ylabel('Sensor Value')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig('results/engine_degradation.png')
plt.show()

print("\n✅ Exploration complete! Check 'results/engine_degradation.png'")