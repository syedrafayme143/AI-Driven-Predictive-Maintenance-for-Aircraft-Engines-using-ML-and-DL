import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib

print("="*50)
print("PHASE 6.1: 1D CNN MODEL")
print("="*50)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================
# STEP 1: Load processed data
# ============================================
print("\nLoading processed data...")
train_df = pd.read_csv('data/train_processed.csv')
feature_cols = joblib.load('models/feature_columns.pkl')

print(f"✅ Data loaded: {train_df.shape}")
print(f"✅ Features: {len(feature_cols)}")

# ============================================
# STEP 2: Create Time-Series Sequences
# ============================================
print("\n" + "="*50)
print("CREATING TIME-SERIES SEQUENCES")
print("="*50)

def create_sequences(df, sequence_length=50):
    """
    Convert tabular data into sequences for CNN.
    Same as LSTM - CNN also works on sequences!
    """
    sequences = []
    targets = []
    
    # Group by engine
    for engine_id in df['unit_id'].unique():
        engine_data = df[df['unit_id'] == engine_id].sort_values('time_cycles')
        
        # Extract features and RUL
        features = engine_data[feature_cols].values
        rul = engine_data['RUL'].values
        
        # Create sequences
        for i in range(sequence_length, len(features)):
            # Take last 'sequence_length' timesteps
            seq = features[i-sequence_length:i]
            target = rul[i]
            
            sequences.append(seq)
            targets.append(target)
    
    return np.array(sequences), np.array(targets)

# Create sequences
sequence_length = 50
print(f"\nCreating sequences with window size: {sequence_length} cycles")
print("This may take a minute...")

X, y = create_sequences(train_df, sequence_length)

print(f"\n✅ Sequences created!")
print(f"Sequence shape: {X.shape}")  # (samples, timesteps, features)
print(f"Target shape: {y.shape}")
print(f"\nExplanation:")
print(f"  - {X.shape[0]:,} sequences")
print(f"  - {X.shape[1]} timesteps per sequence")
print(f"  - {X.shape[2]} features per timestep")

# ============================================
# STEP 3: Split data
# ============================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n📊 Data split:")
print(f"- Training sequences: {X_train.shape[0]:,}")
print(f"- Validation sequences: {X_val.shape[0]:,}")

# ============================================
# STEP 4: Build 1D CNN Model
# ============================================
print("\n" + "="*50)
print("BUILDING 1D CNN MODEL")
print("="*50)

print("\n💡 Why 1D CNN?")
print("- CNNs detect local patterns in sensor readings")
print("- Faster training than LSTM")
print("- Good at finding anomalies in time-series")
print("- Uses convolution filters to extract features\n")

model = keras.Sequential([
    # First Convolutional Block
    layers.Conv1D(filters=64, kernel_size=3, activation='relu', 
                  input_shape=(sequence_length, X.shape[2])),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.2),
    
    # Second Convolutional Block
    layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.2),
    
    # Third Convolutional Block
    layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    layers.BatchNormalization(),
    layers.GlobalMaxPooling1D(),
    layers.Dropout(0.3),
    
    # Dense layers
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    
    # Output layer (single value = RUL prediction)
    layers.Dense(1)
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("\n📐 Model Architecture:")
print(model.summary())

# ============================================
# STEP 5: Train the Model
# ============================================
print("\n" + "="*50)
print("TRAINING 1D CNN")
print("="*50)

print("\n🚀 Starting training...")
print("This will be FASTER than LSTM (5-10 minutes).")
print("Watch the val_loss - we want it to decrease!\n")

# Callbacks
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ============================================
# STEP 6: Evaluate Model
# ============================================
print("\n" + "="*50)
print("EVALUATING 1D CNN")
print("="*50)

# Predictions
y_train_pred = model.predict(X_train, verbose=0).flatten()
y_val_pred = model.predict(X_val, verbose=0).flatten()

# Metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
val_mae = mean_absolute_error(y_val, y_val_pred)
val_r2 = r2_score(y_val, y_val_pred)

print(f"\n📊 1D CNN Results:")
print(f"\nTraining Set:")
print(f"  RMSE: {train_rmse:.2f} cycles")
print(f"  MAE:  {train_mae:.2f} cycles")
print(f"  R²:   {train_r2:.4f}")
print(f"\nValidation Set:")
print(f"  RMSE: {val_rmse:.2f} cycles")
print(f"  MAE:  {val_mae:.2f} cycles")
print(f"  R²:   {val_r2:.4f}")

# Compare with LSTM
print("\n" + "="*50)
print("COMPARISON WITH LSTM")
print("="*50)

lstm_results = pd.read_csv('results/lstm_results.csv')
lstm_rmse = lstm_results.loc[0, 'Val RMSE']

print(f"\n🚀 LSTM Model: {lstm_rmse:.2f} cycles")
print(f"🔷 1D CNN Model: {val_rmse:.2f} cycles")

if val_rmse < lstm_rmse:
    improvement = ((lstm_rmse - val_rmse) / lstm_rmse) * 100
    print(f"\n✅ 1D CNN is BETTER by {improvement:.1f}%!")
else:
    difference = ((val_rmse - lstm_rmse) / lstm_rmse) * 100
    print(f"\n📊 LSTM is {difference:.1f}% better (both are good!)")

# ============================================
# STEP 7: Visualizations
# ============================================
print("\n" + "="*50)
print("CREATING VISUALIZATIONS")
print("="*50)

# 1. Training History
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Loss
axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=11)
axes[0].set_ylabel('Loss (MSE)', fontsize=11)
axes[0].set_title('1D CNN Training History - Loss', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# MAE
axes[1].plot(history.history['mae'], label='Training MAE', linewidth=2)
axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=11)
axes[1].set_ylabel('MAE', fontsize=11)
axes[1].set_title('1D CNN Training History - MAE', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/cnn_training_history.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: 'results/cnn_training_history.png'")
plt.close()

# 2. Predictions vs Actual
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Training
axes[0].scatter(y_train, y_train_pred, alpha=0.5, s=10, color='blue')
axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('True RUL (cycles)', fontsize=11)
axes[0].set_ylabel('Predicted RUL (cycles)', fontsize=11)
axes[0].set_title(f'Training Set - 1D CNN\nRMSE: {train_rmse:.2f} | R²: {train_r2:.3f}', 
                  fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Validation
axes[1].scatter(y_val, y_val_pred, alpha=0.5, s=10, color='green')
axes[1].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[1].set_xlabel('True RUL (cycles)', fontsize=11)
axes[1].set_ylabel('Predicted RUL (cycles)', fontsize=11)
axes[1].set_title(f'Validation Set - 1D CNN\nRMSE: {val_rmse:.2f} | R²: {val_r2:.3f}', 
                  fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/cnn_predictions.png', dpi=300, bbox_inches='tight')
print("✅ Saved: 'results/cnn_predictions.png'")
plt.close()

# ============================================
# STEP 8: Save Model
# ============================================
model.save('models/cnn_model.h5')
print("\n✅ Model saved: 'models/cnn_model.h5'")

# Save results
cnn_results = pd.DataFrame({
    'Model': ['1D CNN'],
    'Train RMSE': [train_rmse],
    'Val RMSE': [val_rmse],
    'Train MAE': [train_mae],
    'Val MAE': [val_mae],
    'Val R²': [val_r2]
})

cnn_results.to_csv('results/cnn_results.csv', index=False)
print("✅ Saved: 'results/cnn_results.csv'")

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*50)
print("PHASE 6.1 COMPLETE! ✅")
print("="*50)
print(f"\n1D CNN Performance:")
print(f"  Validation RMSE: {val_rmse:.2f} cycles")
print(f"  Validation MAE:  {val_mae:.2f} cycles")
print(f"  Validation R²:   {val_r2:.4f}")
print(f"\nFiles created:")
print(f"1. models/cnn_model.h5")
print(f"2. results/cnn_training_history.png")
print(f"3. results/cnn_predictions.png")
print(f"4. results/cnn_results.csv")
print(f"\n🎯 Next: Run Phase 6.2 (Bi-LSTM Model)")
print("="*50)