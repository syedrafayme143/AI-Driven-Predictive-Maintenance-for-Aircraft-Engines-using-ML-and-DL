import kagglehub
import shutil
import os

# 1. Download dataset to kagglehub cache
cache_path = kagglehub.dataset_download("behrad3d/nasa-cmaps")
print("Downloaded to cache:", cache_path)

# 2. Target directory: turbofan-predictive-maintenance/data
project_data_dir = os.path.join(os.getcwd(), "data")

# Create data folder if it doesn't exist
os.makedirs(project_data_dir, exist_ok=True)

# 3. Copy dataset from cache to project/data
for item in os.listdir(cache_path):
    src = os.path.join(cache_path, item)
    dst = os.path.join(project_data_dir, item)

    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)

print("Dataset copied to:", project_data_dir)
