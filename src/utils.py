# utils.py
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Hàm vẽ hình ảnh
def plot_images(images, labels, predictions, start_idx, end_idx):
    plt.figure(figsize=(15, 5))
    for i in range(start_idx, end_idx):
        plt.subplot(1, 5, i + 1 - start_idx)
        plt.imshow(images[i])
        plt.title(f"True: {labels[i]}, Predicted: {predictions[i]}")
        plt.axis('off')
    plt.show()

# Hàm vẽ đường cong ROC
def plot_roc_curve(y_true, y_pred):
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
