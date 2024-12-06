# main.py

from feature_extraction import extract_features, save_features
from similarity_search import find_similar_images, display_top_similar_images

# Trích xuất và lưu trữ đặc trưng cho tất cả các ảnh trong cơ sở dữ liệu
def process_database():
    for img_file in os.listdir('database_images'):
        if img_file.endswith(('.jpg', '.png')):
            img_path = os.path.join('database_images', img_file)
            features = extract_features(img_path)
            output_file = os.path.join('features', img_file.replace('.jpg', '.npy').replace('.png', '.npy'))
            save_features(features, output_file)
    print("Đặc trưng của ảnh đã được lưu trữ.")

# Nhận ảnh đầu vào từ người dùng
input_img_path = 'input_image.jpg'  # Thay thế bằng đường dẫn ảnh đầu vào của bạn

# Tìm kiếm và hiển thị 5 ảnh tương đồng nhất với ảnh đầu vào
similar_images = find_similar_images(input_img_path, top_n=5)
display_top_similar_images(similar_images)
