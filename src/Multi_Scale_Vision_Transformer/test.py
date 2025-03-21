import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

INPUT_HEIGHT, INPUT_WIDTH = 512, 512
NUM_CLASSES = 2

MODEL_PATH = '/path/to/model'
DATASET_PATH = '/path/to/dataset'
TEST1_PATH = f"{DATASET_PATH}/test/time1/"  
TEST2_PATH = f"{DATASET_PATH}/test/time2/"  
LABEL_PATH = f"{DATASET_PATH}/test/label/"   

# Preprocessing
def preprocess_mask(img_msk):
    """
    Convert mask image to binary format.
    
    Parameters:
    - img_msk: Input mask image.
    
    Returns:
    - Binary mask with pixels as 0 or 1.
    """
    r, c, d = img_msk.shape
    mask = np.zeros((r, c))
    for i in range(r):
        for j in range(c):
            px = img_msk[i, j, :]
            if (px[0] == 255 and px[1] == 255 and px[2] == 255):
                mask[i, j] = 1
    return mask

def get_image_array(path1, path2, width, height):
    """
    Load and preprocess two images by applying CLAHE and HSV conversion.
    
    Parameters:
    - path1: First image path.
    - path2: Second image path.
    - width: Desired output width.
    - height: Desired output height.
    
    Returns:
    - Preprocessed image array with combined HSV channels.
    """
    img1 = cv2.imread(path1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1_hsv = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img1_hsv[:, :, 2] = clahe.apply(img1_hsv[:, :, 2])  
    img1_hsv = np.float32(img1_hsv) / 255

    img2 = cv2.imread(path2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2_hsv = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)
    img2_hsv[:, :, 2] = clahe.apply(img2_hsv[:, :, 2])  
    img2_hsv = np.float32(img2_hsv) / 255

    img = np.zeros((height, width, 6), dtype='float32')
    img[:, :, 0] = img1_hsv[:, :, 0]  
    img[:, :, 1] = img1_hsv[:, :, 1]  
    img[:, :, 2] = img1_hsv[:, :, 2]  
    img[:, :, 3] = img2_hsv[:, :, 0]  
    img[:, :, 4] = img2_hsv[:, :, 1]  
    img[:, :, 5] = img2_hsv[:, :, 2]  

    return img

def get_segmentation_array(path, n_classes, width, height):
    """
    Convert segmentation mask to one-hot encoded array.
    
    Parameters:
    - path: Mask image path.
    - n_classes: Number of classes.
    - width: Desired output width.
    - height: Desired output height.
    
    Returns:
    - One-hot encoded segmentation array.
    """
    seg_labels = np.zeros((height, width, n_classes))
    img_msk = cv2.imread(path)
    img_msk = cv2.cvtColor(img_msk, cv2.COLOR_BGR2RGB)
    img_mask = preprocess_mask(img_msk)

    for c in range(n_classes):
        seg_labels[:, :, c] = (img_mask == c).astype(int)

    return seg_labels


X_test, y_test = [], []

test1_img = sorted(os.listdir(TEST1_PATH))[:len(os.listdir(TEST1_PATH))]
test2_img = sorted(os.listdir(TEST2_PATH))[:len(os.listdir(TEST2_PATH))]
test_label = sorted(os.listdir(LABEL_PATH))[:len(os.listdir(LABEL_PATH))]

print('Loading testing images and masks...')
for im1, im2, seg in zip(test1_img, test2_img, test_label):
    X_test.append(get_image_array(TEST1_PATH + im1, TEST2_PATH + im2, INPUT_WIDTH, INPUT_HEIGHT))
    y_test.append(get_segmentation_array(LABEL_PATH + seg, NUM_CLASSES, INPUT_WIDTH, INPUT_HEIGHT))

X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)

print(f'X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')

# Load the pre-trained MSViT model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Evaluate the model
predictions = model.predict(X_test)

# Calculate metrics
y_pred_classes = np.argmax(predictions, axis=-1)
y_true_classes = np.argmax(y_test, axis=-1)

accuracy = accuracy_score(y_true_classes.flatten(), y_pred_classes.flatten())
print(f'Test Accuracy: {accuracy:.4f}')

def mean_iou(y_true, y_pred, num_classes):
    """
    Calculate mean IoU.
    
    Parameters:
    - y_true: True labels.
    - y_pred: Predicted labels.
    - num_classes: Number of classes.
    
    Returns:
    - Mean IoU score.
    """
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    return np.mean(np.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm)))

mean_iou_score = mean_iou(y_true_classes.flatten(), y_pred_classes.flatten(), NUM_CLASSES)
print(f'Mean IoU: {mean_iou_score:.4f}')

def dice_coefficient(y_true, y_pred, num_classes):
    """
    Calculate Dice coefficient.
    
    Parameters:
    - y_true: True labels.
    - y_pred: Predicted labels.
    - num_classes: Number of classes.
    
    Returns:
    - Mean Dice coefficient.
    """
    dice_scores = []
    for c in range(num_classes):
        intersection = np.sum((y_true == c) & (y_pred == c))
        dice_score = 2 * intersection / (np.sum(y_true == c) + np.sum(y_pred == c) + 1e-6)
        dice_scores.append(dice_score)
    return np.mean(dice_scores)

dice_score = dice_coefficient(y_true_classes.flatten(), y_pred_classes.flatten(), NUM_CLASSES)
print(f'Dice Coefficient: {dice_score:.4f}')

# Plot ROC Curve
def plot_roc_curve(y_true, y_score, num_classes):
    """
    Plot the ROC curve for each class.
    
    Parameters:
    - y_true: True labels.
    - y_score: Predicted probabilities.
    - num_classes: Number of classes.
    """
    y_true_flat = y_true.flatten()
    y_score_reshaped = y_score.reshape(-1, num_classes)

    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_flat, y_score_reshaped[:, i], pos_label=i)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

plot_roc_curve(y_true_classes, predictions, NUM_CLASSES)
