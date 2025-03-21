import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

MODEL_PATH = "/path/to/save/model/DET.h5"
MODEL_WEIGHTS_PATH = "/path/to/save/modelweights/DET.weights.h5"

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

# U-Net Model
def unet_base(input_shape, num_classes):
    """
    Define the U-Net model architecture.
    
    Parameters:
    - input_shape: Shape of the input images.
    - num_classes: Number of classes in the segmentation task.
    
    Returns:
    - U-Net model.
    """
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c4)

    # Decoder
    u5 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c5)

    u6 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c7)

    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c7)

    return Model(inputs=[inputs], outputs=[outputs])

# Dual Encoder Transformer Model
def dual_encoder_transformer(input_shape):
    """
    Define the dual encoder transformer model architecture.
    
    Parameters:
    - input_shape: Shape of the input images.
    
    Returns:
    - Dual encoder transformer model.
    """
    inputs = layers.Input(shape=input_shape)

    # Define the first encoder with fewer dense layers
    x1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x1 = layers.MaxPooling2D(pool_size=(2, 2))(x1)
    x1 = layers.Flatten()(x1)
    x1 = layers.Dense(64, activation='relu')(x1)

    # Define the second encoder with fewer dense layers
    x2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x2 = layers.MaxPooling2D(pool_size=(2, 2))(x2)
    x2 = layers.Flatten()(x2)
    x2 = layers.Dense(64, activation='relu')(x2)

    # Concatenate the two encoders' outputs
    combined = layers.Concatenate()([x1, x2])
    combined = layers.Dense(128, activation='relu')(combined)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(combined)

    return Model(inputs=[inputs], outputs=[outputs])

# Combined Model
def combined_model(input_shape_unet, input_shape_transformer):
    """
    Define the combined model architecture integrating U-Net and dual encoder transformer.
    
    Parameters:
    - input_shape_unet: Shape of the input images for U-Net.
    - input_shape_transformer: Shape of the input images for the transformer.
    
    Returns:
    - Combined model.
    """
    unet_model = unet_base(input_shape_unet, NUM_CLASSES)
    transformer_model = dual_encoder_transformer(input_shape_transformer)

    # Freeze the weights of both models for initial training
    for layer in unet_model.layers:
        layer.trainable = False
    for layer in transformer_model.layers:
        layer.trainable = False

    # Combined input
    unet_input = layers.Input(shape=input_shape_unet)
    transformer_input = layers.Input(shape=input_shape_transformer)

    # Get outputs from models
    unet_output = unet_model(unet_input)
    if isinstance(unet_output, list):
        unet_output = unet_output[-1]

    transformer_output = transformer_model(transformer_input)
    if isinstance(transformer_output, list):
        transformer_output = transformer_output[-1]

    transformer_output = layers.Dense(512 * 512, activation='relu')(transformer_output)
    transformer_output_reshaped = layers.Reshape((512, 512, 1))(transformer_output)
    transformer_output_expanded = layers.Concatenate(axis=-1)([transformer_output_reshaped] * NUM_CLASSES)

    # Merge U-Net and transformer outputs
    combined_output = layers.Concatenate(axis=-1)([unet_output, transformer_output_expanded])

    final_output = layers.Conv2D(NUM_CLASSES, (1, 1), activation='softmax')(combined_output)

    return Model(inputs=[unet_input, transformer_input], outputs=[final_output])

# Initialize and compile the combined model
combined_input_shape_unet = (512, 512, 6)  
combined_input_shape_transformer = (512, 512, 6)  

model = combined_model(combined_input_shape_unet, combined_input_shape_transformer)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Training the Combined Model
history = model.fit(
    [X_train, X_train],  
    y_train,
    validation_data=([X_val, X_val], y_val),
    epochs=10,
    batch_size=4
)

# Save the Combined Model
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
