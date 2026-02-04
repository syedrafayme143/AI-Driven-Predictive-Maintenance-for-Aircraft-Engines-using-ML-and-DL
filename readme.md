# 🚀 Turbofan Engine Predictive Maintenance
This project implements an **end-to-end predictive maintenance pipeline** for aircraft turbofan engines using **NASA’s C-MAPSS dataset**.

The objective is to predict **Remaining Useful Life (RUL)** using machine learning and deep learning techniques.

---
## 📁 Project Structure

```text
data/
models/
results/
turbofan/
01_data_exploration.py
02_data_preprocessing.py
03_ml_baseline.py
04_deep_learning_lstm.py
05_cnn_model.py
06_bilstm_model.py
07_hybrid_cnn_lstm.py
08_hyperparameter_tuning.py
09_final_comparison.py
dataimport.py
```

---
## 🧠 Pipeline Overview

### `01_data_exploration.py`
- Exploratory Data Analysis (EDA) and dataset understanding

### `02_data_preprocessing.py`
- Data cleaning, feature engineering, normalization, and windowing

### `03_ml_baseline.py`
- Classical machine learning baseline models (RF, XGBoost, LightGBM)

### `04_deep_learning_lstm.py`
- LSTM-based deep learning model for time-series RUL prediction

### `05_cnn_model.py`
- 1D CNN model for extracting spatial sensor patterns

### `06_bilstm_model.py`
- Bidirectional LSTM for enhanced temporal dependency learning

### `07_hybrid_cnn_lstm.py`
- Hybrid CNN-LSTM architecture combining spatial + temporal learning

### `08_hyperparameter_tuning.py`
- Hyperparameter tuning for performance optimization

### `09_final_comparison.py`
- Final evaluation, visualization, and executive summary generation

### `dataimport.py`
- Utility module for loading and managing datasets

---
## 🛠️ Technologies Used

- Python
- NumPy, Pandas
- Scikit-learn
- TensorFlow / Keras
- XGBoost, LightGBM
- Matplotlib, Seaborn

---
## ▶️ How to Run

Run scripts **in order**:

```bash
python 01_data_exploration.py
python 02_data_preprocessing.py
python 03_ml_baseline.py
python 04_deep_learning_lstm.py
python 05_cnn_model.py
python 06_bilstm_model.py
python 07_hybrid_cnn_lstm.py
python 08_hyperparameter_tuning.py
python 09_final_comparison.py
```

---
## 📅 Project Info

- Generated on: 2026-02-04
- Author: Your Name
