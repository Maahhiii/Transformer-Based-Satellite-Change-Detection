import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

DATA_DIR = '/content/drive/MyDrive/CLCD'
MODEL_SAVE_DIR = '/content/drive/MyDrive/Models'
BATCH_SIZE = 4
EPOCHS = 10
INPUT_HEIGHT, INPUT_WIDTH = 512, 512
NUM_CLASSES = 2

def load_images(img_path1, img_path2, label_path, load_fraction=1.0):
    """
    Loads and preprocesses images from specified paths.
    
    Parameters:
    - img_path1: Path to the first set of images.
    - img_path2: Path to the second set of images.
    - label_path: Path to the labels.
    - load_fraction: Fraction of the dataset to load.
    
    Returns:
    - Preprocessed images and labels.
    """
    img_list1 = sorted(os.listdir(img_path1))
    img_list2 = sorted(os.listdir(img_path2))
    label_list = sorted(os.listdir(label_path))

    # Load only a specified fraction of the dataset
    fraction_len = int(len(img_list1) * load_fraction)
    img_list1 = img_list1[:fraction_len]
    img_list2 = img_list2[:fraction_len]
    label_list = label_list[:fraction_len]

    images1, images2, labels = [], [], []

    for img1, img2, lbl in zip(img_list1, img_list2, label_list):
        img1_array = img_to_array(load_img(os.path.join(img_path1, img1), target_size=(INPUT_HEIGHT, INPUT_WIDTH))) / 255.0
        img2_array = img_to_array(load_img(os.path.join(img_path2, img2), target_size=(INPUT_HEIGHT, INPUT_WIDTH))) / 255.0
        label_array = img_to_array(load_img(os.path.join(label_path, lbl), target_size=(INPUT_HEIGHT, INPUT_WIDTH), color_mode="grayscale"))

        images1.append(img1_array)
        images2.append(img2_array)
        labels.append(label_array)

    return np.array(images1), np.array(images2), np.array(labels)

def load_dataset():
    """
    Loads the dataset for training and validation.
    
    Returns:
    - Training and validation images and labels.
    """
    img_train1 = f'{DATA_DIR}/train/time1'
    img_train2 = f'{DATA_DIR}/train/time2'
    label_train = f'{DATA_DIR}/train/label'

    img_val1 = f'{DATA_DIR}/val/time1'
    img_val2 = f'{DATA_DIR}/val/time2'
    label_val = f'{DATA_DIR}/val/label'

    train_images1, train_images2, train_labels = load_images(img_train1, img_train2, label_train, load_fraction=1.0)
    val_images1, val_images2, val_labels = load_images(img_val1, img_val2, label_val, load_fraction=1.0)

    train_labels = np.squeeze(train_labels)  
    val_labels = np.squeeze(val_labels)  

    train_labels[train_labels == 255] = 1  
    val_labels[val_labels == 255] = 1

    train_labels_one_hot = np.eye(NUM_CLASSES)[train_labels.astype(int)]
    val_labels_one_hot = np.eye(NUM_CLASSES)[val_labels.astype(int)]

    train_labels_one_hot = train_labels_one_hot.reshape((train_labels.shape[0], INPUT_HEIGHT, INPUT_WIDTH, NUM_CLASSES))
    val_labels_one_hot = val_labels_one_hot.reshape((val_labels.shape[0], INPUT_HEIGHT, INPUT_WIDTH, NUM_CLASSES))

    return train_images1, train_images2, train_labels_one_hot, val_images1, val_images2, val_labels_one_hot

# U-Net Model
def unet_model(input_shape, num_classes):
    """
    Defines a U-Net model for image segmentation.
    
    Parameters:
    - input_shape: Shape of the input images.
    - num_classes: Number of classes in the segmentation task.
    
    Returns:
    - U-Net model.
    """
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c9)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

# Vision Transformer Model
class VisionTransformer(tf.keras.Model):
    """
    Defines a Vision Transformer model for image segmentation.
    
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

        self.patch_embedding = layers.Conv2D(projection_dim, (16, 16), strides=(16, 16))  

        self.transformer_blocks = [self.build_transformer_block() for _ in range(transformer_layers)]

        self.classification_head = layers.Dense(num_classes, activation='softmax')

    def build_transformer_block(self):
        """
        Builds a single transformer block.
        
        Returns:
        - Transformer block model.
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

# Hybrid U-Net and Vision Transformer Model
def hybrid_model(input_shape, num_classes):
    """
    Defines a hybrid model combining U-Net and Vision Transformer.
    
    Parameters:
    - input_shape: Shape of the input images.
    - num_classes: Number of classes in the segmentation task.
    
    Returns:
    - Hybrid model.
    """
    unet = unet_model(input_shape, num_classes)
    inputs = layers.Input(shape=input_shape)

    unet_output = unet(inputs)

    vit = VisionTransformer(num_patches=(INPUT_HEIGHT // 16) * (INPUT_WIDTH // 16),
                            projection_dim=64,
                            transformer_layers=4,
                            num_heads=4,
                            mlp_dim=128,
                            num_classes=num_classes)

    vit_output = vit(inputs)

    vit_output_reshaped = layers.Reshape((INPUT_HEIGHT // 16, INPUT_WIDTH // 16, num_classes))(vit_output)
    vit_output_resized = layers.UpSampling2D(size=(16, 16))(vit_output_reshaped)  

    combined_output = layers.add([unet_output, vit_output_resized])

    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(combined_output)

    return tf.keras.Model(inputs=[inputs], outputs=[outputs])

if __name__ == "__main__":
    train_images1, train_images2, train_labels_one_hot, val_images1, val_images2, val_labels_one_hot = load_dataset()

    hybrid = hybrid_model((INPUT_HEIGHT, INPUT_WIDTH, 3), NUM_CLASSES)
    hybrid.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the hybrid model
    hybrid_history = hybrid.fit(train_images1, train_labels_one_hot,
                                validation_data=(val_images1, val_labels_one_hot),
                                epochs=EPOCHS, batch_size=BATCH_SIZE)

    # Save the model
    model_path = os.path.join(MODEL_SAVE_DIR, 'vit_model.h5')
    hybrid.save(model_path)

    # Plot training and validation accuracy and loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(hybrid_history.history['accuracy'])
    plt.plot(hybrid_history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(hybrid_history.history['loss'])
    plt.plot(hybrid_history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()
