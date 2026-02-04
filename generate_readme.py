import os
from datetime import date

PROJECT_NAME = "Turbofan Engine Predictive Maintenance"
AUTHOR = "Your Name"
TODAY = date.today().isoformat()


def describe_script(filename):
    name = filename.lower()

    if "exploration" in name:
        return "Exploratory Data Analysis (EDA) and dataset understanding"
    if "preprocess" in name:
        return "Data cleaning, feature engineering, normalization, and windowing"
    if "ml_baseline" in name:
        return "Classical machine learning baseline models (RF, XGBoost, LightGBM)"
    if "bilstm" in name:
        return "Bidirectional LSTM for enhanced temporal dependency learning"
    if "hybrid" in name:
        return "Hybrid CNN-LSTM architecture combining spatial + temporal learning"
    if "cnn" in name:
        return "1D CNN model for extracting spatial sensor patterns"
    if "lstm" in name:
        return "LSTM-based deep learning model for time-series RUL prediction"
    if "hyperparameter" in name:
        return "Hyperparameter tuning for performance optimization"
    if "final" in name or "comparison" in name:
        return "Final evaluation, visualization, and executive summary generation"
    if "dataimport" in name:
        return "Utility module for loading and managing datasets"

    return "Project utility or helper script"


# --------------------------------------------
# Scan project root
# --------------------------------------------
items = sorted(os.listdir("."))

folders = [f for f in items if os.path.isdir(f) and not f.startswith(".")]
scripts = sorted(
    f for f in items
    if f.endswith(".py") and f != "generate_readme.py"
)

# --------------------------------------------
# Build README line-by-line (SAFE)
# --------------------------------------------
lines = []

lines.append(f"# 🚀 {PROJECT_NAME}\n")
lines.append(
    "This project implements an **end-to-end predictive maintenance pipeline** "
    "for aircraft turbofan engines using **NASA’s C-MAPSS dataset**.\n\n"
)
lines.append(
    "The objective is to predict **Remaining Useful Life (RUL)** using "
    "machine learning and deep learning techniques.\n\n"
)

lines.append("---\n")
lines.append("## 📁 Project Structure\n\n")
lines.append("```text\n")

for folder in folders:
    lines.append(f"{folder}/\n")

for script in scripts:
    lines.append(f"{script}\n")

lines.append("```\n\n")
lines.append("---\n")
lines.append("## 🧠 Pipeline Overview\n\n")

for script in scripts:
    lines.append(f"### `{script}`\n")
    lines.append(f"- {describe_script(script)}\n\n")

lines.append("---\n")
lines.append("## 🛠️ Technologies Used\n\n")
lines.append("- Python\n")
lines.append("- NumPy, Pandas\n")
lines.append("- Scikit-learn\n")
lines.append("- TensorFlow / Keras\n")
lines.append("- XGBoost, LightGBM\n")
lines.append("- Matplotlib, Seaborn\n\n")

lines.append("---\n")
lines.append("## ▶️ How to Run\n\n")
lines.append("Run scripts **in order**:\n\n")
lines.append("```bash\n")
lines.append("python 01_data_exploration.py\n")
lines.append("python 02_data_preprocessing.py\n")
lines.append("python 03_ml_baseline.py\n")
lines.append("python 04_deep_learning_lstm.py\n")
lines.append("python 05_cnn_model.py\n")
lines.append("python 06_bilstm_model.py\n")
lines.append("python 07_hybrid_cnn_lstm.py\n")
lines.append("python 08_hyperparameter_tuning.py\n")
lines.append("python 09_final_comparison.py\n")
lines.append("```\n\n")

lines.append("---\n")
lines.append("## 📅 Project Info\n\n")
lines.append(f"- Generated on: {TODAY}\n")
lines.append(f"- Author: {AUTHOR}\n")

# --------------------------------------------
# Write README.md
# --------------------------------------------
with open("README.md", "w", encoding="utf-8") as f:
    f.writelines(lines)

print("✅ README.md successfully generated!")
