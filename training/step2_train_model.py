import os
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf  # ← chỉ 1 dòng import TensorFlow
import matplotlib.pyplot as plt

# ======= ĐƯỜNG DẪN =======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(BASE_DIR, '..', 'data')
eye_data_path = os.path.join(data_dir, 'eye_data.pkl')
model_save_path = os.path.join(data_dir, 'eye_state_model.h5')

# 1) Nạp dữ liệu từ bước 1
with open(eye_data_path, "rb") as f:
    X, y = pickle.load(f)

# 2) Chia train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3) Xây mô hình CNN (tf.keras)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                           input_shape=(24, 24, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')       # nhị phân
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 4) Huấn luyện
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# 5) Lưu mô hình
os.makedirs(data_dir, exist_ok=True)
model.save(model_save_path)
print(f"Đã lưu {model_save_path}")

# 6) Vẽ biểu đồ
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss')
plt.legend()

plt.show()

# 7) Vẽ confusion matrix trên tập test

y_pred = model.predict(X_test)
y_pred_label = (y_pred > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred_label)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[
            'Closed', 'Open'], yticklabels=['Closed', 'Open'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
