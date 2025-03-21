import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report

MODEL_PATH = "/path/to/save/model/STA.h5"
MODEL_WEIGHTS_PATH = "/path/to/save/modelweights/STA.weights.h5"

DATASET_PATH = "/path/to/dataset"
IMG_TRAIN1 = f"{DATASET_PATH}/train/time1/"
IMG_TRAIN2 = f"{DATASET_PATH}/train/time2/"
LABEL_TRAIN = f"{DATASET_PATH}/train/label/"
IMG_VAL1 = f"{DATASET_PATH}/val/time1/"
IMG_VAL2 = f"{DATASET_PATH}/val/time2/"
LABEL_VAL = f"{DATASET_PATH}/val/label/"

INPUT_HEIGHT, INPUT_WIDTH = 512, 512
NUM_CLASSES = 2

def preprocess_mask(img_msk):
    """
    Converts a mask image into a binary format.
    
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
    Loads and preprocesses two images using CLAHE and HSV conversion.
    
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
    Converts a segmentation mask into a one-hot encoded array.
    
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

# Load Training and Validation Data

X_train, y_train = [], []
X_val, y_val = [], []

train1_img = sorted(os.listdir(IMG_TRAIN1))[:len(os.listdir(IMG_TRAIN1))]
train2_img = sorted(os.listdir(IMG_TRAIN2))[:len(os.listdir(IMG_TRAIN2))]
train_label = sorted(os.listdir(LABEL_TRAIN))[:len(os.listdir(LABEL_TRAIN))]

val1_img = sorted(os.listdir(IMG_VAL1))[:len(os.listdir(IMG_VAL1))]
val2_img = sorted(os.listdir(IMG_VAL2))[:len(os.listdir(IMG_VAL2))]
val_label = sorted(os.listdir(LABEL_VAL))[:len(os.listdir(LABEL_VAL))]

print('Loading training images and masks...')
for im1, im2, seg in zip(train1_img, train2_img, train_label):
    X_train.append(get_image_array(IMG_TRAIN1 + im1, IMG_TRAIN2 + im2, INPUT_WIDTH, INPUT_HEIGHT))
    y_train.append(get_segmentation_array(LABEL_TRAIN + seg, NUM_CLASSES, INPUT_WIDTH, INPUT_HEIGHT))

print('Loading validation images and masks...')
for im1, im2, seg in zip(val1_img, val2_img, val_label):
    X_val.append(get_image_array(IMG_VAL1 + im1, IMG_VAL2 + im2, INPUT_WIDTH, INPUT_HEIGHT))
    y_val.append(get_segmentation_array(LABEL_VAL + seg, NUM_CLASSES, INPUT_WIDTH, INPUT_HEIGHT))

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)
X_val = np.array(X_val, dtype=np.float32)
y_val = np.array(y_val, dtype=np.float32)

# U-Net Model
def unet_base(input_shape, num_classes):
    """
    Defines a U-Net model for image segmentation.
    
    Parameters:
    - input_shape: Shape of the input images.
    - num_classes: Number of classes in the segmentation task.
    
    Returns:
    - U-Net model.
    """
    inputs = Input(shape=input_shape)

    # Encoder
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    # Bottleneck
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c9)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# STA model
class SpatialTemporalAttention(layers.Layer):
    """
    Spatial-Temporal Attention layer.
    
    Parameters:
    - channels: Number of channels in the input.
    
    Returns:
    - Spatial-temporal attention layer.
    """
    def __init__(self, channels, **kwargs):
        super(SpatialTemporalAttention, self).__init__(**kwargs)
        self.channels = channels
        self.conv1 = layers.Conv2D(channels, (1, 1), padding='same')
        self.conv2 = layers.Conv2D(channels, (1, 1), padding='same')

    def call(self, x):
        spatial_attention = tf.nn.softmax(self.conv1(x), axis=-1)
        temporal_attention = tf.nn.softmax(self.conv2(x), axis=-2)
        return x * spatial_attention * temporal_attention

# Complete Model with Attention
def create_model(input_shape, num_classes):
    """
    Define the complete model with spatial-temporal attention.
    
    Parameters:
    - input_shape: Shape of the input images.
    - num_classes: Number of classes in the segmentation task.
    
    Returns:
    - Complete model with attention.
    """
    base_model = unet_base(input_shape, num_classes)
    inputs = base_model.input
    x = base_model.output
    x = SpatialTemporalAttention(channels=num_classes)(x)
    model = Model(inputs, x)
    return model

# Compile Model
model = create_model((INPUT_HEIGHT, INPUT_WIDTH, 6), NUM_CLASSES)
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Training with Generators
def load_data_in_batches(image_paths1, image_paths2, labels, batch_size, input_width, input_height):
    """
    Load data in batches.
    
    Parameters:
    - image_paths1: Paths to the first set of images.
    - image_paths2: Paths to the second set of images.
    - labels: Corresponding labels.
    - batch_size: Batch size for training.
    - input_width: Desired input width.
    - input_height: Desired input height.
    
    Yields:
    - Batches of preprocessed images and labels.
    """
    batch_X = []
    batch_y = []
    for i in range(len(image_paths1)):
        print(f"Loading images: {image_paths1[i]} and {image_paths2[i]}")
        image_array = get_image_array(image_paths1[i], image_paths2[i], input_width, input_height)
        if image_array is not None:
            batch_X.append(image_array)
            batch_y.append(labels[i])  

        if len(batch_X) == batch_size:
            yield np.array(batch_X), np.array(batch_y)  
            batch_X = [] 
            batch_y = []  
    if len(batch_X) > 0:
        yield np.array(batch_X), np.array(batch_y) 

train_gen = load_data_in_batches(IMG_TRAIN1, IMG_TRAIN2, LABEL_TRAIN, batch_size=2, input_width=256, input_height=256)
val_gen = load_data_in_batches(IMG_VAL1, IMG_VAL2, LABEL_VAL, batch_size=2, input_width=256, input_height=256)

# Model Training with Generators
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=2)

# Save the model weights and architecture
model.save(MODEL_PATH)
model.save(MODEL_WEIGHTS_PATH)

# Evaluation
def plot_metrics(history):
    """
    Plot training and validation metrics.
    
    Parameters:
    - history: Training history object.
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

plot_metrics(history)