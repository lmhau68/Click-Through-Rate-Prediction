import os
from pathlib import Path

# Các thư mục cốt lõi của dự án
list_of_directories = [
    "data/raw",              # Chứa dataset CSV gốc tải về
    "data/processed",        # Chứa data đã làm sạch
    "notebooks/Exploratory", # Nơi phân tích và trực quan hóa dữ liệu
    "notebooks/Modeling",    # Nơi thử nghiệm thuật toán
    "src/models",            # Chứa code huấn luyện chính thức
    "api",                   # Chứa code backend kết nối web
    "models"                 # Nơi xuất file mô hình (.pkl) sau khi train
]

for dir_path in list_of_directories:
    os.makedirs(dir_path, exist_ok=True)

# Các file cấu hình và logic trống
list_of_files = [
    ".gitignore",
    "requirements.txt",
    "README.md",
    "src/__init__.py",
    "src/models/__init__.py",
    "src/models/train_model.py",
    "src/models/predict_model.py"
]

for filepath in list_of_files:
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            pass