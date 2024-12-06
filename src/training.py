# train.py

import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from config import IMG_SIZE, BATCH_SIZE, DATA_DIR, VALIDATION_SPLIT, TEST_SPLIT, LEARNING_RATE, EPOCHS
from data_preprocessing import create_generators
from model_setup import create_model

# Tạo các generator
train_generator, val_generator, test_generator, class_weights_dict = create_generators(
    DATA_DIR, IMG_SIZE, BATCH_SIZE, VALIDATION_SPLIT, TEST_SPLIT
)

# Tạo mô hình
model = create_model(IMG_SIZE, LEARNING_RATE, len(train_generator.class_indices))

# Thiết lập các callback
checkpoint = ModelCheckpoint('best_model_vgg16.h5',
                             save_best_only=True,
                             monitor='val_accuracy',
                             mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Huấn luyện mô hình
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[checkpoint, early_stopping],
    class_weight=class_weights_dict
)
