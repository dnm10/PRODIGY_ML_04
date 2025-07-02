import os
import cv2
import numpy as np

def load_images(data_dir, target_size=(128, 128)):
    X, y = [], []
    folders = os.listdir(data_dir)
    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        gestures = os.listdir(folder_path)
        for gesture in gestures:
            gesture_path = os.path.join(folder_path, gesture)
            if not os.path.isdir(gesture_path):
                continue
            images = os.listdir(gesture_path)
            for image_name in images:
                image_path = os.path.join(gesture_path, image_name)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, target_size)
                X.append(img)
                y.append(gesture)  # Label by gesture folder name
    return np.array(X), np.array(y)
