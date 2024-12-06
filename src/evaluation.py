# evaluate.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import load_model
from data_preprocessing import create_generators
from config import IMG_SIZE, BATCH_SIZE, DATA_DIR, VALIDATION_SPLIT, TEST_SPLIT

# Tạo các generator
train_generator, val_generator, test_generator, _ = create_generators(
    DATA_DIR, IMG_SIZE, BATCH_SIZE, VALIDATION_SPLIT, TEST_SPLIT
)

# Load mô hình tốt nhất
model = load_model('best_model_vgg16.h5')

# Đánh giá mô hình trên tập kiểm tra
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc:.4f}")

# Dự đoán và tính toán các chỉ số đánh giá
y_true = test_generator.classes
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

# Tính toán và vẽ ma trận nhầm lẫn
confusion_mtx = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Vẽ đường cong ROC cho từng lớp
from sklearn.preprocessing import label_binarize

y_true_bin = label_binarize(y_true, classes=list(test_generator.class_indices.values()))
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(test_generator.class_indices)):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], label=f'ROC curve class {i} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for each class')
plt.legend(loc="lower right")
plt.show()
