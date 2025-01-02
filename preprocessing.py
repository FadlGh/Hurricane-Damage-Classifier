import os
import tensorflow as tf

IMG_SIZE = 256
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

    def process_image(file_path, label):
        """Process a single image and apply grayscale and resizing."""
        image = tf.io.read_file(file_path)  # Read the image from file
        image = tf.image.decode_jpeg(image, channels=3)  # Decode as RGB
        image = tf.image.rgb_to_grayscale(image)  # Convert RGB to grayscale
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))  # Resize image
        image = tf.cast(image, tf.float32) / 255.0  # Normalize image
        return image, label

    def augment_if_needed(image, label):
        """Augment the image if it belongs to the 'damage' class."""
        if label == 0:  # If 'damage' class
            augmented_image = augment_image(image)
            # Return both the original and augmented image
            return tf.data.Dataset.from_tensors((image, label)).concatenate(
                tf.data.Dataset.from_tensors((augmented_image, label))
            )
        else:
            # Return the original image for 'no_damage' class
            return tf.data.Dataset.from_tensors((image, label))

    # Map preprocessing and augmentation
    dataset = dataset.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Augment the 'damage' class
    dataset = dataset.flat_map(augment_if_needed)

    dataset = dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = create_training_data()
print("Created training data successfully")

for images, labels in train_dataset.take(1):
    print(images.shape)  # Should print (batch_size, 256, 256, 1)
    print(labels.shape)  # Should print (batch_size,)
