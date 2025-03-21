import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, concatenate, Dense, BatchNormalization, Activation, GlobalAveragePooling2D, Reshape, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import Dense
from sklearn.cluster import KMeans

MODEL_PATH = "/path/to/save/model/MSViT.h5"
MODEL_WEIGHTS_PATH = "/path/to/save/modelweights/MSViT.weights.h5"

DATASET_PATH = "/path/to/dataset"
IMG_TRAIN1 = f"{DATASET_PATH}/train/time1/"
IMG_TRAIN2 = f"{DATASET_PATH}/train/time2/"
LABEL_TRAIN = f"{DATASET_PATH}/train/label/"
IMG_VAL1 = f"{DATASET_PATH}/val/time1/"
IMG_VAL2 = f"{DATASET_PATH}/val/time2/"
LABEL_VAL = f"{DATASET_PATH}/val/label/"

INPUT_HEIGHT, INPUT_WIDTH = 512, 512
NUM_CLASSES = 2
IMG_SIZE = (512, 512)

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

def cluster_image(I):
    """
    Cluster image using K-means.
    
    Parameters:
    - I: Input image.
    
    Returns:
    - Clustered image.
    """
    I2 = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    a = np.asarray(I2, dtype=np.float32)
    x, y = a.shape
    a1 = a.reshape(x * y, 1)
    k_means = KMeans(n_clusters=7)
    k_means.fit(a1)
    labels = k_means.labels_
    img_clustered = k_means.cluster_centers_[labels].reshape(x, y)
    return img_clustered

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
    def load_and_process_image(path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_hsv[:, :, 2] = clahe.apply(img_hsv[:, :, 2])
        return np.float32(img_hsv) / 255

    img1_hsv = load_and_process_image(path1)
    img2_hsv = load_and_process_image(path2)

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

# Load Training and Validation Data
def load_data(img_train1, img_train2, label_train, img_val1, img_val2, label_val):
    """
    Load and preprocess training and validation data.
    
    Parameters:
    - img_train1: Path to the first set of training images.
    - img_train2: Path to the second set of training images.
    - label_train: Path to the training labels.
    - img_val1: Path to the first set of validation images.
    - img_val2: Path to the second set of validation images.
    - label_val: Path to the validation labels.
    
    Returns:
    - Preprocessed training and validation data.
    """
    X_train, y_train = [], []
    X_val, y_val = [], []

    train1_img = sorted(os.listdir(img_train1))[:len(os.listdir(img_train1))]
    train2_img = sorted(os.listdir(img_train2))[:len(os.listdir(img_train2))]
    train_label = sorted(os.listdir(label_train))[:len(os.listdir(label_train))]

    val1_img = sorted(os.listdir(img_val1))[:len(os.listdir(img_val1))]
    val2_img = sorted(os.listdir(img_val2))[:len(os.listdir(img_val2))]
    val_label = sorted(os.listdir(label_val))[:len(os.listdir(label_val))]

    print('Loading training images and masks...')
    for im1, im2, seg in zip(train1_img, train2_img, train_label):
        X_train.append(get_image_array(img_train1 + im1, img_train2 + im2, INPUT_WIDTH, INPUT_HEIGHT))
        y_train.append(get_segmentation_array(label_train + seg, NUM_CLASSES, INPUT_WIDTH, INPUT_HEIGHT))

    print('Loading validation images and masks...')
    for im1, im2, seg in zip(val1_img, val2_img, val_label):
        X_val.append(get_image_array(img_val1 + im1, img_val2 + im2, INPUT_WIDTH, INPUT_HEIGHT))
        y_val.append(get_segmentation_array(label_val + seg, NUM_CLASSES, INPUT_WIDTH, INPUT_HEIGHT))

    return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val)

X_train, y_train, X_val, y_val = load_data(IMG_TRAIN1, IMG_TRAIN2, LABEL_TRAIN, IMG_VAL1, IMG_VAL2, LABEL_VAL)

# MSViT Model
def ms_vit(input_shape, num_classes):
    """
    Define the MSViT model architecture.
    
    Parameters:
    - input_shape: Shape of the input images.
    - num_classes: Number of classes in the segmentation task.
    
    Returns:
    - MSViT model.
    """
    inputs = Input(shape=input_shape)

    # Initial Conv Layer
    x = Conv2D(64, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Multi-Scale Attention Block 1
    x1 = Conv2D(64, (1, 1), padding='same')(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    x2 = Conv2D(64, (3, 3), padding='same')(x)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)

    x3 = Conv2D(64, (5, 5), padding='same')(x)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)

    # Concatenate multi-scale features
    x = concatenate([x1, x2, x3], axis=-1)

    # Second Conv Layer
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 128))(x)  # Reshape for compatibility with U-Net

    # Adding a Conv2D layer to maintain spatial dimensions
    x = Conv2D(128, (1, 1), padding='same')(x)  
    x = UpSampling2D(size=(16, 16))(x)  

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# Combined Model
def combined_model(input_shape, num_classes):
    """
    Define the combined model architecture integrating MSViT with U-Net.
    
    Parameters:
    - input_shape: Shape of the input images.
    - num_classes: Number of classes in the segmentation task.
    
    Returns:
    - Combined model.
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
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5) 

    # MSViT integration
    ms_vit_model = ms_vit(input_shape=(c5.shape[1:]), num_classes=num_classes)
    x = ms_vit_model(c5) 

    # Decoder
    x = UpSampling2D(size=(4, 4))(x) 

    u6 = concatenate([x, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = UpSampling2D(size=(2, 2))(c6)  
    u7 = concatenate([u7, c3])  
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = UpSampling2D(size=(2, 2))(c7)  
    u8 = concatenate([u8, c2])  
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = UpSampling2D(size=(2, 2))(c8)
    u9 = concatenate([u9, c1]) 
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c9)

    return Model(inputs=[inputs], outputs=[outputs])

# Compile and train the combined model
input_shape = (INPUT_HEIGHT, INPUT_WIDTH, 6)  
model = combined_model(input_shape, NUM_CLASSES)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=2, epochs=10, validation_data=(X_val, y_val))

# Save model
model.save_weights(MODEL_WEIGHTS_PATH)
model.save(MODEL_PATH)

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
