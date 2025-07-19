# PRODIGY_ML_04

# ✋ Hand Gesture Recognition using CNN

This project implements a **Convolutional Neural Network (CNN)** to recognize hand gestures using webcam input. The model is trained on the **[LeapGestRecog Dataset](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)** and supports real-time gesture prediction using OpenCV and TensorFlow.

---

## 🎯 Objective

To classify static hand gestures in real time from a webcam feed using a CNN trained on grayscale gesture images.

---

## 🧠 Model Overview

- **Algorithm**: Convolutional Neural Network (CNN)
- **Framework**: TensorFlow / Keras
- **Evaluation Metric**: Accuracy
- **Tools Used**: Python, OpenCV, NumPy, TensorFlow, Scikit-learn

---

## 📦 Features

- 📷 **Webcam Support**: Real-time prediction
- 🔍 **Preprocessing**: Resize to 128x128, grayscale, normalize
- 🧠 **Model**: 3 convolutional layers + dense classifier
- 🗂️ **10 Gesture Classes**:
  - Palm
  - Fist
  - Thumb
  - Index
  - L
  - OK
  - Palm Moved
  - Fist Moved
  - C
  - Down

---

## 🧪 Training Details

- Dataset: [LeapGestRecog](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)
- Preprocessing:
  - Resize to **128x128**
  - Grayscale + Normalize
  - One-hot encode labels
- CNN Architecture:
  - Conv2D → MaxPool → Dropout
  - Flatten → Dense (ReLU) → Dense (Softmax)
- Trained for: `50 epochs` (adjustable)

---

## 🛠️ Dependencies
Install all required packages:
- tensorflow>=2.8.0

- opencv-python

- numpy

- scikit-learn

- joblib

- matplotlib

---

## ▶️ Running the App

```bash
cd app/
python webcam_app.py
```
---

## 📊 Results
✅ Accuracy: ~97% on test set

🎯 Real-time predictions with consistent outputs in good lighting

🔍 Debug output shows prediction vector, decoded label

---

## 📌 Notes
- Lighting and hand positioning significantly affect webcam accuracy.

- Consider using background removal or segmentation for better performance.

- Extendable to gesture-based control systems or sign language recognition.

