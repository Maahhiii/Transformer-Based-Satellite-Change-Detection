import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

INPUT_HEIGHT, INPUT_WIDTH = 512, 512
NUM_CLASSES = 2

DATASET_PATH = '/content/drive/MyDrive/CLCD'
TEST1_PATH = f"{DATASET_PATH}/test/time1/"  
TEST2_PATH = f"{DATASET_PATH}/test/time2/"  
LABEL_PATH = f"{DATASET_PATH}/test/label/"
MODEL_PATH = '/path/to/model'   

def preprocess_mask(img_msk):
    """
    Preprocesses a mask image by converting it into a binary format.
    
    Parameters:
    - img_msk: The input mask image.
    
    Returns:
    - A binary mask where pixels are either 0 or 1.
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
    Loads and preprocesses two images by applying CLAHE and converting them into HSV format.
    
    Parameters:
    - path1: Path to the first image.
    - path2: Path to the second image.
    - width: The desired width of the output image.
    - height: The desired height of the output image.
    
    Returns:
    - A preprocessed image array with combined HSV channels from both images.
    """
    img1 = cv2.imread(path1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1_hsv = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img1_hsv[:, :, 2] = clahe.apply(img1_hsv[:, :, 2])  # Apply CLAHE on the V channel
    img1_hsv = np.float32(img1_hsv) / 255

    img2 = cv2.imread(path2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2_hsv = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)
    img2_hsv[:, :, 2] = clahe.apply(img2_hsv[:, :, 2])  # Apply CLAHE on the V channel
    img2_hsv = np.float32(img2_hsv) / 255

    img = np.zeros((height, width, 6), dtype='float32')
    img[:, :, 0] = img1_hsv[:, :, 0]  # H channel of img1
    img[:, :, 1] = img1_hsv[:, :, 1]  # S channel of img1
    img[:, :, 2] = img1_hsv[:, :, 2]  # V channel of img1
    img[:, :, 3] = img2_hsv[:, :, 0]  # H channel of img2
    img[:, :, 4] = img2_hsv[:, :, 1]  # S channel of img2
    img[:, :, 5] = img2_hsv[:, :, 2]  # V channel of img2

    return img

def get_segmentation_array(path, n_classes, width, height):
    """
    Converts a segmentation mask into a one-hot encoded array.
    
    Parameters:
    - path: Path to the segmentation mask image.
    - n_classes: Number of classes in the segmentation task.
    - width: The desired width of the output array.
    - height: The desired height of the output array.
    
    Returns:
    - A one-hot encoded segmentation array.
    """
    seg_labels = np.zeros((height, width, n_classes))
    img_msk = cv2.imread(path)
    img_msk = cv2.cvtColor(img_msk, cv2.COLOR_BGR2RGB)
    img_mask = preprocess_mask(img_msk)

    for c in range(n_classes):
        seg_labels[:, :, c] = (img_mask == c).astype(int)

    return seg_labels

class VisionTransformer(tf.keras.Model):
    """
    A Vision Transformer model for image segmentation tasks.
    
    Attributes:
    - num_patches: Number of patches in the input image.
    - projection_dim: Dimension of the patch embeddings.
    - transformer_layers: Number of transformer layers.
    - num_heads: Number of attention heads.
    - mlp_dim: Dimension of the MLP in the transformer blocks.
    - num_classes: Number of classes in the segmentation task.
    """
    def __init__(self, num_patches, projection_dim, transformer_layers, num_heads, mlp_dim, num_classes, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.transformer_layers = transformer_layers
        self.num_classes = num_classes

        # Patching and embedding
        self.patch_embedding = layers.Conv2D(projection_dim, (16, 16), strides=(16, 16))  

        # Create transformer blocks
        self.transformer_blocks = [self.build_transformer_block() for _ in range(transformer_layers)]

        # Classification head
        self.classification_head = layers.Dense(num_classes, activation='softmax')

    def build_transformer_block(self):
        """
        Builds a single transformer block.
        
        Returns:
        - A transformer block model.
        """
        inputs = layers.Input(shape=(None, self.projection_dim))  
        # Multi-head attention
        x = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.projection_dim)(inputs, inputs)
        # Layer normalization and residual connection
        x = layers.LayerNormalization(epsilon=1e-6)(inputs + x)
        # Feed-forward network (MLP)
        x_ffn = layers.Dense(self.mlp_dim, activation='relu')(x)
        x = layers.Dense(self.projection_dim)(x_ffn)
        return tf.keras.Model(inputs=inputs, outputs=x)

    def call(self, x):
        """
        Defines the forward pass through the model.
        
        Parameters:
        - x: Input to the model.
        
        Returns:
        - Output of the model.
        """
        # Patch the input and project
        x = self.patch_embedding(x)
        x = tf.reshape(x, (tf.shape(x)[0], -1, self.projection_dim))  

        # Pass through the transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x)

        # Classification head
        x = self.classification_head(x)
        return x

    def get_config(self):
        config = super(VisionTransformer, self).get_config()
        config.update({
            'num_patches': self.num_patches,
            'projection_dim': self.projection_dim,
            'transformer_layers': self.transformer_layers,
            'num_heads': self.num_heads,
            'mlp_dim': self.mlp_dim,
            'num_classes': self.num_classes
        })
        return config

    @classmethod
    def from_config(cls, config):
        config.pop('name', None)
        return cls(**config)

def load_testing_data(test1_path, test2_path, label_path, input_width, input_height, num_classes):
    """
    Loads and preprocesses testing data.
    
    Parameters:
    - test1_path: Path to the first set of test images.
    - test2_path: Path to the second set of test images.
    - label_path: Path to the test labels.
    - input_width: Desired width of the input images.
    - input_height: Desired height of the input images.
    - num_classes: Number of classes in the segmentation task.
    
    Returns:
    - Preprocessed testing data (X_test, y_test).
    """
    X_test, y_test = [], []

    test1_img = sorted(os.listdir(test1_path))[:int(len(os.listdir(test1_path)))]
    test2_img = sorted(os.listdir(test2_path))[:int(len(os.listdir(test2_path)))]
    test_label = sorted(os.listdir(label_path))[:int(len(os.listdir(label_path)))]

    print('Loading testing images and masks...')
    for im1, im2, seg in zip(test1_img, test2_img, test_label):
        X_test.append(get_image_array(test1_path + im1, test2_path + im2, input_width, input_height))
        y_test.append(get_segmentation_array(label_path + seg, num_classes, input_width, input_height))

    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    print(f'X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')
    return X_test, y_test

def calculate_metrics(y_true, y_pred, num_classes):
    """
    Calculates accuracy, mean IoU, and Dice coefficient.
    
    Parameters:
    - y_true: True labels.
    - y_pred: Predicted labels.
    - num_classes: Number of classes.
    
    Returns:
    - Accuracy, mean IoU, and Dice coefficient.
    """
    accuracy = accuracy_score(y_true.flatten(), y_pred.flatten())
    
    def mean_iou(y_true, y_pred, num_classes):
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
        return np.mean(np.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm)))

    mean_iou_score = mean_iou(y_true.flatten(), y_pred.flatten(), num_classes)

    def dice_coefficient(y_true, y_pred, num_classes):
        dice_scores = []
        for c in range(num_classes):
            intersection = np.sum((y_true == c) & (y_pred == c))
            dice_score = 2 * intersection / (np.sum(y_true == c) + np.sum(y_pred == c) + 1e-6)
            dice_scores.append(dice_score)
        return np.mean(dice_scores)

    dice_score = dice_coefficient(y_true.flatten(), y_pred.flatten(), num_classes)
    
    return accuracy, mean_iou_score, dice_score

def plot_roc_curve(y_true, y_score, num_classes):
    """
    Plots the ROC curve for each class.
    
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

X_test, y_test = load_testing_data(TEST1_PATH, TEST2_PATH, LABEL_PATH, INPUT_WIDTH, INPUT_HEIGHT, NUM_CLASSES)

with tf.keras.utils.custom_object_scope({'VisionTransformer': VisionTransformer}):
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

predictions = model.predict(X_test)

# Calculate metrics
y_pred_classes = np.argmax(predictions, axis=-1)
y_true_classes = np.argmax(y_test, axis=-1)

# Calculate accuracy
accuracy, mean_iou_score, dice_score = calculate_metrics(y_true_classes, y_pred_classes, NUM_CLASSES)
print(f'Test Accuracy: {accuracy:.4f}')
print(f'Mean IoU: {mean_iou_score:.4f}')
print(f'Dice Coefficient: {dice_score:.4f}')

# Plot ROC Curve
plot_roc_curve(y_true_classes, predictions, NUM_CLASSES)