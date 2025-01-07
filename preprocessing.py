import os
import tensorflow as tf

IMG_SIZE = 128
CATEGORIES = ['damage', 'no_damage']
DIRECTORY = 'archive'

def augment_image(image):
    """Apply augmentations to the image."""
    image = tf.image.random_flip_left_right(image)  # Random horizontal flip
    image = tf.image.random_brightness(image, max_delta=0.3)  # Random brightness
    image = tf.image.random_contrast(image, lower=0.7, upper=1.3)  # Random contrast
    image = tf.image.random_flip_up_down(image)  # Random vertical flip
    return image

def create_training_data():
    """Create a TensorFlow dataset from images with augmentation for one class."""
    file_paths = []
    labels = []

    # Gather file paths and labels
    for category in CATEGORIES:
        path = os.path.join(DIRECTORY, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            file_paths.append(os.path.join(path, img))
            labels.append(class_num)

    # Create initial dataset
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    def process_and_augment(file_path, label):
        """Process a single image and optionally augment."""
        image = tf.io.read_file(file_path)  # Read the image from file
        image = tf.image.decode_jpeg(image, channels=3)  # Decode as RGB
        image = tf.image.rgb_to_grayscale(image)  # Convert RGB to grayscale
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))  # Resize image
        image = tf.cast(image, tf.float32) / 255.0  # Normalize image
        
        if label == 0:  # If 'damage' class, augment the image
            augmented_image = augment_image(image)
            return tf.data.Dataset.from_tensors((image, label)).concatenate(
                tf.data.Dataset.from_tensors((augmented_image, label))
            )
        else:
            return tf.data.Dataset.from_tensors((image, label))

    # Apply preprocessing and augmentation
    dataset = dataset.flat_map(process_and_augment)

    # Batch, shuffle, and prefetch
    dataset = dataset.batch(16).shuffle(1000).repeat().prefetch(tf.data.AUTOTUNE)
    return dataset

# Create training dataset
train_dataset = create_training_data()
print("Created training data successfully")

# Debug dataset
for images, labels in train_dataset.take(1):
    print(f"Images shape: {images.shape}")  # Expected: (32, 128, 128, 1)
    print(f"Labels shape: {labels.shape}")  # Expected: (32,)
