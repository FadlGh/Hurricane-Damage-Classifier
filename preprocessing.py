import os
import tensorflow as tf

IMG_SIZE = 256
CATEGORIES = ['damage', 'no_damage']
DIRECTORY = 'archive'

def augment_image(image):
    image = tf.image.random_flip_left_right(image)  # Random horizontal flip
    image = tf.image.random_brightness(image, max_delta=0.3)  # Random brightness
    image = tf.image.random_contrast(image, lower=0.7, upper=1.3)  # Random contrast
    image = tf.image.random_flip_up_down(image)  # Random vertical flip
    return image

def create_training_data():
    file_paths = []
    labels = []
    
    for category in CATEGORIES:
        path = os.path.join(DIRECTORY, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
               file_paths.append(os.path.join(path, img))
               labels.append(class_num)
            except Exception as e:
                print(e)
    
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    
    def process_image(file_path, label):
        image = tf.io.read_file(file_path)  # Read the image from file
        image = tf.image.decode_jpeg(image, channels=3)  # Decode as RGB
        image = tf.image.rgb_to_grayscale(image)  # Convert RGB to grayscale
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))  # Resize image
        image = tf.cast(image, tf.float32) / 255.0  # Normalize image
        # Augment only for 'damage' (category 1) to solve inbalance
        if label == 0:
            image = augment_image(image) # Apply augmentation
        return image, label

    dataset = dataset.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
     
    dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = create_training_data()
print("Created training data successfully")