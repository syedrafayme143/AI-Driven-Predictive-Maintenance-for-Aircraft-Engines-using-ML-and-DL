import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# STEP 1: Load the data
# ============================================
print("="*50)
print("PHASE 2: DATA PREPROCESSING")
print("="*50)

columns = ['unit_id', 'time_cycles', 'setting_1', 'setting_2', 'setting_3']
columns += [f'sensor_{i}' for i in range(1, 22)]

# Load from your CMaps folder
train_df = pd.read_csv('data/train_FD001.txt', sep=' ', header=None, 
                        names=columns, index_col=False)
train_df = train_df.dropna(axis=1)

print(f"\n✅ Data loaded successfully!")
print(f"Shape: {train_df.shape}")
print(f"Engines: {train_df['unit_id'].nunique()}")

# ============================================
# STEP 2: Calculate RUL (Target Variable)
# ============================================
print("\n" + "-"*50)
print("STEP 1: Calculating RUL")
print("-"*50)

# Find when each engine failed
max_cycles = train_df.groupby('unit_id')['time_cycles'].max().reset_index()
max_cycles.columns = ['unit_id', 'max_cycle']

# Add max_cycle to each row
train_df = train_df.merge(max_cycles, on='unit_id', how='left')

# Calculate RUL = cycles left until failure
train_df['RUL'] = train_df['max_cycle'] - train_df['time_cycles']

print(f"\n✅ RUL calculated!")
print(f"RUL range: {train_df['RUL'].min()} to {train_df['RUL'].max()} cycles")
print("\nExample - Engine 1:")
print(train_df[train_df['unit_id']==1][['unit_id', 'time_cycles', 'max_cycle', 'RUL']].head(10))

# ============================================
# STEP 3: Remove Useless Sensors
# ============================================
print("\n" + "-"*50)
print("STEP 2: Finding useless sensors")
print("-"*50)

sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
sensor_variance = train_df[sensor_cols].var()

print("\nSensor variances:")
for sensor, var in sensor_variance.sort_values().items():
    print(f"{sensor}: {var:.6f}")

# Remove sensors that don't change (variance < 0.001)
threshold = 0.001
useless_sensors = sensor_variance[sensor_variance < threshold].index.tolist()

print(f"\n❌ Removing useless sensors: {useless_sensors}")
train_df = train_df.drop(columns=useless_sensors)

# Update sensor list
sensor_cols = [col for col in sensor_cols if col not in useless_sensors]
print(f"✅ Keeping {len(sensor_cols)} useful sensors")

# ============================================
# STEP 4: Correlation with RUL
# ============================================
print("\n" + "-"*50)
print("STEP 3: Analyzing correlations")
print("-"*50)

correlations = train_df[sensor_cols + ['RUL']].corr()['RUL'].drop('RUL').sort_values()

print("\nCorrelation with RUL:")
for sensor, corr in correlations.items():
    print(f"{sensor}: {corr:.4f}")

# Plot correlations
plt.figure(figsize=(10, 6))
correlations.plot(kind='barh', color='steelblue')
plt.title('Sensor Correlation with RUL', fontsize=14, fontweight='bold')
plt.xlabel('Correlation Coefficient')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.savefig('results/sensor_correlations.png', dpi=300)
print("\n✅ Correlation chart saved to 'results/sensor_correlations.png'")
plt.close()

# ============================================
# STEP 5: Normalize the data
# ============================================
print("\n" + "-"*50)
print("STEP 4: Normalizing data")
print("-"*50)

feature_cols = ['setting_1', 'setting_2', 'setting_3'] + sensor_cols

print(f"\nNormalizing {len(feature_cols)} features...")

# Create scaler
scaler = MinMaxScaler()

# Normalize to [0, 1] range
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])

print("✅ Normalization complete!")
print("\nFeature ranges after normalization:")
print(train_df[feature_cols].describe().loc[['min', 'max']])

# ============================================
# STEP 6: Save everything
# ============================================
print("\n" + "-"*50)
print("STEP 5: Saving processed data")
print("-"*50)

# Save processed data
train_df.to_csv('data/train_processed.csv', index=False)
print("✅ Saved: 'data/train_processed.csv'")

# Save scaler for later use
import joblib
joblib.dump(scaler, 'models/scaler.pkl')
print("✅ Saved: 'models/scaler.pkl'")

# Save feature names for later
joblib.dump(feature_cols, 'models/feature_columns.pkl')
print("✅ Saved: 'models/feature_columns.pkl'")

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*50)
print("PHASE 2 COMPLETE! ✅")
print("="*50)
print(f"\nDataset Summary:")
print(f"- Total samples: {len(train_df):,}")
print(f"- Engines: {train_df['unit_id'].nunique()}")
print(f"- Features: {len(feature_cols)}")
print(f"- Sensors removed: {len(useless_sensors)}")
print(f"- Sensors kept: {len(sensor_cols)}")
print(f"\nFiles created:")
print(f"1. data/train_processed.csv")
print(f"2. models/scaler.pkl")
print(f"3. models/feature_columns.pkl")
print(f"4. results/sensor_correlations.png")
print("\n🎯 Next: Run Phase 3 (Machine Learning Baseline)")
print("="*50)