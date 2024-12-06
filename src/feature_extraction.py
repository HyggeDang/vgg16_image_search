import numpy as np
import os
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

# Hàm tải mô hình đã huấn luyện
def load_trained_model(model_file='best_model_vgg16.h5'):
    model = load_model(model_file)
    return model

# Hàm trích xuất đặc trưng từ ảnh
def extract_features(model, img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Trích xuất đặc trưng từ mô hình đã huấn luyện (sử dụng lớp trước đầu ra)
    features = model.predict(img_array)
    features_flat = features.flatten()
    
    return features_flat

# Hàm lưu trữ đặc trưng và đường dẫn ảnh
def save_features_and_paths(features, path, label, features_dir='features', paths_dir='paths'):
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(paths_dir, exist_ok=True)
    
    # Lưu đặc trưng ảnh vào file .pkl
    feature_file = os.path.join(features_dir, f"{label}_feature.pkl")
    path_file = os.path.join(paths_dir, f"{label}_path.pkl")
    
    # Lưu đặc trưng ảnh dưới dạng pickle
    with open(feature_file, 'wb') as f:
        pickle.dump(features, f)
    
    # Lưu đường dẫn ảnh dưới dạng pickle
    with open(path_file, 'wb') as f:
        pickle.dump(path, f)

# Hàm xử lý và lưu trữ tất cả các ảnh trong cơ sở dữ liệu
def process_database(model, database_dir='database_images'):
    labels = os.listdir(database_dir)
    for label in labels:
        label_path = os.path.join(database_dir, label)
        if os.path.isdir(label_path):
            for img_file in os.listdir(label_path):
                if img_file.endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(label_path, img_file)
                    features = extract_features(model, img_path)
                    save_features_and_paths(features, img_path, label)

    print("Đặc trưng ảnh và đường dẫn đã được lưu trữ.")

if __name__ == '__main__':
    # Tải mô hình đã huấn luyện từ file
    model = load_trained_model('best_model_vgg16.h5')
    
    # Chạy quá trình trích xuất đặc trưng và lưu trữ
    process_database(model)
