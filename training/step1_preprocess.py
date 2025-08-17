import os
import cv2
import numpy as np
import pickle

# Kích thước ảnh đầu vào
img_size = 24

# Đường dẫn thư mục gốc của project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Đường dẫn tương đối
closed_path = os.path.join(BASE_DIR, 'train', 'Closed_Eyes')
open_path   = os.path.join(BASE_DIR, 'train', 'Open_Eyes')
output_path = os.path.join(BASE_DIR, '..', 'data', 'eye_data.pkl')

X = []
y = []

def load_images_from_folder(folder, label):
    for filename in os.listdir(folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (img_size, img_size))
                X.append(img)
                y.append(label)

# Load ảnh
load_images_from_folder(closed_path, 0)  # Mắt nhắm
load_images_from_folder(open_path, 1)    # Mắt mở

# Chuyển thành array
X = np.array(X).reshape(-1, img_size, img_size, 1) / 255.0
y = np.array(y)

# In thông tin kiểm tra
print("Tổng số ảnh:", len(X))
print("Shape của 1 ảnh:", X[0].shape)
print("Số ảnh nhãn 0 (mắt nhắm):", sum(y == 0))
print("Số ảnh nhãn 1 (mắt mở):", sum(y == 1))

# Lưu dữ liệu
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "wb") as f:
    pickle.dump((X, y), f)
print(f"✅ Dữ liệu đã được lưu vào {output_path}")
