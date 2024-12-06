import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Các tham số có thể tùy chỉnh
img_size = (224, 224)  # Kích thước ảnh đầu vào
batch_size = 32  # Số lượng ảnh trong mỗi batch
data_dir = 'data'  # Thư mục chứa dữ liệu
validation_split = 0.2  # Tỷ lệ chia dữ liệu cho tập kiểm tra
test_split = 0.1  # Tỷ lệ chia dữ liệu cho tập đánh giá
learning_rate = 0.001  # Tốc độ học
epochs = 50  # Số lượng epochs

# Hàm tiền xử lý tùy chỉnh (nếu cần)
def custom_preprocessing_function(img):
    # Ví dụ: thực hiện chuẩn hóa ảnh
    img = img / 255.0  # Chuẩn hóa giá trị pixel
    return img

# Tạo ImageDataGenerator để tăng cường dữ liệu cho tập huấn luyện
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Chuẩn hóa ảnh
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    elastic_transform=True,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],  # Thay đổi độ sáng ảnh
    fill_mode='nearest',
    validation_split=validation_split,  # Chia tập huấn luyện và kiểm tra
    preprocessing_function=custom_preprocessing_function  # Hàm tiền xử lý tùy chỉnh
)

# Tạo ImageDataGenerator cho tập kiểm tra
val_datagen = ImageDataGenerator(
    rescale=1./255,  # Chuẩn hóa ảnh
    validation_split=validation_split,  # Chia tập huấn luyện và kiểm tra
    preprocessing_function=custom_preprocessing_function  # Hàm tiền xử lý tùy chỉnh
)

# Tạo ImageDataGenerator cho tập đánh giá
test_datagen = ImageDataGenerator(
    rescale=1./255,  # Chuẩn hóa ảnh
    validation_split=test_split,  # Chia tập kiểm tra và tập đánh giá
    preprocessing_function=custom_preprocessing_function  # Hàm tiền xử lý tùy chỉnh
)

# Tạo các generator để nạp dữ liệu từ thư mục
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',  # Chỉ định rằng đây là tập huấn luyện
    shuffle=True  # Xáo trộn dữ liệu
)

val_generator = val_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',  # Chỉ định rằng đây là tập kiểm tra
    shuffle=True  # Xáo trộn dữ liệu
)

test_generator = test_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',  # Chỉ định rằng đây là tập đánh giá
    shuffle=False  # Không xáo trộn để so sánh kết quả
)

# Tải mô hình VGG16 đã được huấn luyện trước
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Đóng băng các lớp trong phần base model để không thay đổi trọng số khi huấn luyện
for layer in base_model.layers:
    layer.trainable = False

# Thêm các lớp mới để tạo thành mô hình hoàn chỉnh
x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(len(train_generator.class_indices), activation='softmax')(x)

# Tạo mô hình tổng thể
model = Model(inputs=base_model.input, outputs=output_layer)

# Biên dịch mô hình với optimizer và learning rate có thể điều chỉnh
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Sử dụng ModelCheckpoint để lưu mô hình tốt nhất
checkpoint = ModelCheckpoint('best_model_vgg16.h5',
                             save_best_only=True,
                             monitor='val_accuracy',
                             mode='max')

# Sử dụng EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Huấn luyện mô hình
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=[checkpoint, early_stopping]
)

# Đánh giá mô hình trên tập kiểm tra
model.load_weights('best_model_vgg16.h5')  # Tải lại mô hình tốt nhất từ checkpoint
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc:.4f}")

# Dự đoán trên tập kiểm tra và tính toán các chỉ số đánh giá
y_true = test_generator.classes  # Nhãn thật
y_pred = model.predict(test_generator, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)

# In ra báo cáo chi tiết
print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=test_generator.class_indices.keys()))

# Tính toán các chỉ số khác (precision, recall, F1-score)
precision = precision_score(y_true, y_pred_classes, average='weighted')
recall = recall_score(y_true, y_pred_classes, average='weighted')
f1 = f1_score(y_true, y_pred_classes, average='weighted')
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# Tính toán ma trận nhầm lẫn
from sklearn.metrics import confusion_matrix
confusion_mtx = confusion_matrix(y_true, y_pred_classes)
print("Confusion Matrix:")
print(confusion_mtx)

# Hiển thị một số hình ảnh dự đoán
def plot_images(images, labels, predictions, start_idx, end_idx):
    plt.figure(figsize=(15, 5))
    for i in range(start_idx, end_idx):
        plt.subplot(1, 5, i + 1 - start_idx)
        plt.imshow(images[i])
        plt.title(f"True: {labels[i]}, Predicted: {predictions[i]}")
        plt.axis('off')
    plt.show()

# Lấy một batch dữ liệu từ tập test để hiển thị
X_test, y_test = next(test_generator)
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Hiển thị 5 hình ảnh đầu tiên
plot_images(X_test, y_test, y_pred_classes, 0, 5)

# Vẽ đường cong ROC
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_true.ravel(), y_pred.ravel())
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()