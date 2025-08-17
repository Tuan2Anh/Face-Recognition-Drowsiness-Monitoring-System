# Face-Recognition-Drowsiness-Monitoring-System  

A real-time system for **facial recognition** and **drowsiness detection**, designed to improve driver safety and prevent accidents.  

---

## Features  
- Real-time **eye-state classification** (open vs closed eyes) using TensorFlow CNN.  
- **Face authentication** integrated with MTCNN + InceptionResnetV1.  
- Custom preprocessing pipeline for **loading, labeling, and normalizing** eye images (24×24 grayscale).  
- Model training with **80/20 split**, performance evaluation via accuracy/loss plots and confusion matrix.  
- **Arduino alerts** (LED & buzzer) triggered when drowsiness or unauthorized access is detected.  
- Fallback logic and fail-safes to **reduce false positives**.  

---

## Tech Stack  
- **Python**, **TensorFlow/Keras**, **NumPy**, **OpenCV**  
- **MTCNN**, **InceptionResnetV1**  
- **Arduino (LED/Buzzer alerts)**  

## Model Overview  
- Input: Grayscale eye images (24×24)  
- Architecture: Convolutional Neural Network (CNN)  
- Output: Binary classifier (Open / Closed)  
- Evaluation: Accuracy, loss curves, confusion matrix  

## How It Works  
1. Capture live video stream.  
2. Detect face using **MTCNN**.  
3. Authenticate face with **InceptionResnetV1**.  
4. Classify eye state (open/closed) with CNN model.  
5. Trigger **LED/Buzzer alert via Arduino** if drowsiness detected or unauthorized user appears.  

## Installation & Usage  
```bash
# Clone repository
git clone https://github.com/Tuan2Anh/Face-Recognition-Drowsiness-Monitoring-System.git
cd Face-Recognition-Drowsiness-Monitoring-System

# Install dependencies
pip install -r requirements.txt

# Run detection
python main.py
