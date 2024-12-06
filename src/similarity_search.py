# similarity_search.py

import numpy as np
import os
from scipy.spatial import distance
from feature_extraction import extract_features
import matplotlib.pyplot as plt
import cv2

# Thư mục chứa ảnh trong cơ sở dữ liệu và các file lưu trữ đặc trưng
DATABASE_DIR = 'database_images'
FEATURES_DIR = 'features'

# Hàm tính toán độ tương đồng giữa ảnh đầu vào và các ảnh trong cơ sở dữ liệu
def find_similar_images(input_img_path, top_n=5):
    input_features = extract_features(input_img_path)
    similarities = []
    
    for feature_file in os.listdir(FEATURES_DIR):
        feature_path = os.path.join(FEATURES_DIR, feature_file)
        if feature_path.endswith('.npy'):
            stored_features = np.load(feature_path)
            dist = distance.euclidean(input_features, stored_features)
            similarities.append((feature_file, dist))
    
    # Sắp xếp theo độ tương đồng (thấp nhất trước)
    similarities.sort(key=lambda x: x[1])
    
    return similarities[:top_n]

# Hàm hiển thị các ảnh tương đồng
def display_top_similar_images(similar_images):
    plt.figure(figsize=(15, 15))
    
    for i, (img_file, dist) in enumerate(similar_images):
        img_path = os.path.join(DATABASE_DIR, img_file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.subplot(1, len(similar_images), i + 1)
        plt.imshow(img)
        plt.title(f"Dist: {dist:.2f}")
        plt.axis('off')
    
    plt.show()
