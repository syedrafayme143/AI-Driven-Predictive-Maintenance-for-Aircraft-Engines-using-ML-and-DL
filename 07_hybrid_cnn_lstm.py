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
print("PHASE 6.3: HYBRID CNN-LSTM MODEL")
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
    Convert tabular data into sequences for Hybrid CNN-LSTM.
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
print(f"Sequence shape: {X.shape}")
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
# STEP 4: Build Hybrid CNN-LSTM Model
# ============================================
print("\n" + "="*50)
print("BUILDING HYBRID CNN-LSTM MODEL")
print("="*50)

print("\n💡 Why Hybrid CNN-LSTM?")
print("- CNN extracts spatial features from sensors")
print("- LSTM learns temporal patterns over time")
print("- Combines best of both architectures")
print("- Often achieves best performance!\n")

model = keras.Sequential([
    # CNN Feature Extraction Block
    layers.Conv1D(filters=64, kernel_size=3, activation='relu', 
                  input_shape=(sequence_length, X.shape[2])),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.2),
    
    layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    # LSTM Temporal Learning Block
    layers.LSTM(64, return_sequences=True),
    layers.Dropout(0.2),
    
    layers.LSTM(32, return_sequences=False),
    layers.Dropout(0.2),
    
    # Dense Output Block
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    
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
print("TRAINING HYBRID CNN-LSTM")
print("="*50)

print("\n🚀 Starting training...")
print("This will take 10-20 minutes (more complex model).")
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
print("EVALUATING HYBRID CNN-LSTM")
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

print(f"\n📊 Hybrid CNN-LSTM Results:")
print(f"\nTraining Set:")
print(f"  RMSE: {train_rmse:.2f} cycles")
print(f"  MAE:  {train_mae:.2f} cycles")
print(f"  R²:   {train_r2:.4f}")
print(f"\nValidation Set:")
print(f"  RMSE: {val_rmse:.2f} cycles")
print(f"  MAE:  {val_mae:.2f} cycles")
print(f"  R²:   {val_r2:.4f}")

# Compare with all previous models
print("\n" + "="*50)
print("COMPARISON WITH ALL MODELS")
print("="*50)

lstm_results = pd.read_csv('results/lstm_results.csv')
cnn_results = pd.read_csv('results/cnn_results.csv')
bilstm_results = pd.read_csv('results/bilstm_results.csv')

all_dl_results = pd.concat([lstm_results, cnn_results, bilstm_results], ignore_index=True)

print("\n" + all_dl_results[['Model', 'Val RMSE', 'Val MAE', 'Val R²']].to_string(index=False))
print(f"\n🔥 Hybrid CNN-LSTM: {val_rmse:.2f} cycles")

best_so_far = all_dl_results['Val RMSE'].min()
if val_rmse < best_so_far:
    improvement = ((best_so_far - val_rmse) / best_so_far) * 100
    print(f"\n🏆 Hybrid is the BEST! ({improvement:.1f}% improvement)")
else:
    print(f"\n📊 All models performing well!")

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
axes[0].set_title('Hybrid CNN-LSTM Training History - Loss', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# MAE
axes[1].plot(history.history['mae'], label='Training MAE', linewidth=2)
axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=11)
axes[1].set_ylabel('MAE', fontsize=11)
axes[1].set_title('Hybrid CNN-LSTM Training History - MAE', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/hybrid_training_history.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: 'results/hybrid_training_history.png'")
plt.close()

# 2. Predictions vs Actual
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Training
axes[0].scatter(y_train, y_train_pred, alpha=0.5, s=10, color='red')
axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
             'k--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('True RUL (cycles)', fontsize=11)
axes[0].set_ylabel('Predicted RUL (cycles)', fontsize=11)
axes[0].set_title(f'Training Set - Hybrid CNN-LSTM\nRMSE: {train_rmse:.2f} | R²: {train_r2:.3f}', 
                  fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Validation
axes[1].scatter(y_val, y_val_pred, alpha=0.5, s=10, color='darkgreen')
axes[1].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 
             'k--', lw=2, label='Perfect Prediction')
axes[1].set_xlabel('True RUL (cycles)', fontsize=11)
axes[1].set_ylabel('Predicted RUL (cycles)', fontsize=11)
axes[1].set_title(f'Validation Set - Hybrid CNN-LSTM\nRMSE: {val_rmse:.2f} | R²: {val_r2:.3f}', 
                  fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/hybrid_predictions.png', dpi=300, bbox_inches='tight')
print("✅ Saved: 'results/hybrid_predictions.png'")
plt.close()

# ============================================
# STEP 8: Save Model
# ============================================
model.save('models/hybrid_cnn_lstm.h5')
print("\n✅ Model saved: 'models/hybrid_cnn_lstm.h5'")

# Save results
hybrid_results = pd.DataFrame({
    'Model': ['Hybrid CNN-LSTM'],
    'Train RMSE': [train_rmse],
    'Val RMSE': [val_rmse],
    'Train MAE': [train_mae],
    'Val MAE': [val_mae],
    'Val R²': [val_r2]
})

hybrid_results.to_csv('results/hybrid_results.csv', index=False)
print("✅ Saved: 'results/hybrid_results.csv'")

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*50)
print("PHASE 6.3 COMPLETE! ✅")
print("="*50)
print(f"\nHybrid CNN-LSTM Performance:")
print(f"  Validation RMSE: {val_rmse:.2f} cycles")
print(f"  Validation MAE:  {val_mae:.2f} cycles")
print(f"  Validation R²:   {val_r2:.4f}")
print(f"\nFiles created:")
print(f"1. models/hybrid_cnn_lstm.h5")
print(f"2. results/hybrid_training_history.png")
print(f"3. results/hybrid_predictions.png")
print(f"4. results/hybrid_results.csv")
print(f"\n🎯 Next: Run Phase 7 (Hyperparameter Tuning)")
print("="*50)