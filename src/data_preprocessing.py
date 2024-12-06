# data_preprocessing.py

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

# Hàm tiền xử lý tùy chỉnh (nếu cần)
def custom_preprocessing_function(img):
    img = img / 255.0  # Chuẩn hóa giá trị pixel
    return img

# Hàm tạo ImageDataGenerators cho tập huấn luyện, kiểm tra và đánh giá
def create_generators(data_dir, img_size, batch_size, validation_split, test_split):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=validation_split,
        preprocessing_function=custom_preprocessing_function
    )

    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split,
        preprocessing_function=custom_preprocessing_function
    )

    test_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=test_split,
        preprocessing_function=custom_preprocessing_function
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )

    test_generator = test_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    # Tính toán class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weights_dict = dict(zip(np.unique(train_generator.classes), class_weights))

    return train_generator, val_generator, test_generator, class_weights_dict
