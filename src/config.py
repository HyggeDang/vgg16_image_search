# config.py

# Các tham số có thể tùy chỉnh
IMG_SIZE = (224, 224)  # Kích thước ảnh đầu vào
BATCH_SIZE = 32  # Số lượng ảnh trong mỗi batch
DATA_DIR = 'data'  # Thư mục chứa dữ liệu
VALIDATION_SPLIT = 0.2  # Tỷ lệ chia dữ liệu cho tập kiểm tra
TEST_SPLIT = 0.1  # Tỷ lệ chia dữ liệu cho tập đánh giá
LEARNING_RATE = 0.001  # Tốc độ học
EPOCHS = 50  # Số lượng epochs

# Đường dẫn lưu mô hình
MODEL_PATH = 'best_model_vgg16.h5'