import os
import cv2
import random
import pickle
import numpy as np

IMG_SIZE = 256
CATEGORIES = ['damage', 'no_damage']
DIRECTORY = 'archive'

def rotate_image(image):
    angle = random.randint(-30, 30)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))

def flip_image(image):
    return cv2.flip(image, 1) 

def adjust_brightness(image):
    factor = random.uniform(0.7, 1.3)
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

augmentations = [rotate_image, flip_image, adjust_brightness]

# Set chunk size to process data in smaller batches
chunk_size = 1000  # Adjust as needed

# Create lists to store the images and labels
X = []
y = []

chunk_counter = 0

def create_training_data():
    global chunk_counter  # Explicitly declare global to modify the counter
    for category in CATEGORIES:
        path = os.path.join(DIRECTORY, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                new_array = new_array / 255.0
                X.append(new_array)
                y.append(class_num)

                # Augment only for 'damage' (category 1) to solve imbalance
                if class_num == 0:  # Skip augmentation for 'no_damage'
                    continue
                
                augmentation_func = random.choice(augmentations)
                augmented_image = augmentation_func(new_array)
                X.append(augmented_image)
                y.append(class_num)
            except Exception as e:
                continue

            # If we've reached the chunk size, process and save
            if len(X) >= chunk_size:
                # Convert X and y to numpy arrays with appropriate dtype
                X_chunk = np.array(X, dtype=np.float32)
                y_chunk = np.array(y, dtype=np.float32)

                # Save the chunk to pickle with correct naming
                with open(f"X_chunk_{chunk_counter}.pickle", "wb") as pickle_out:
                    pickle.dump(X_chunk, pickle_out)

                with open(f"y_chunk_{chunk_counter}.pickle", "wb") as pickle_out:
                    pickle.dump(y_chunk, pickle_out)

                # Increment the chunk counter
                chunk_counter += 1

                # Clear the lists to start the next chunk
                X.clear()
                y.clear()

# Call the function to create training data
create_training_data()

# If any remaining data is left after the last chunk, save it
if X:
    X_chunk = np.array(X, dtype=np.float32)
    y_chunk = np.array(y, dtype=np.int32)
    with open(f"X_chunk_{chunk_counter}.pickle", "wb") as pickle_out:
        pickle.dump(X_chunk, pickle_out)

    with open(f"y_chunk_{chunk_counter}.pickle", "wb") as pickle_out:
        pickle.dump(y_chunk, pickle_out)

print(f"Data processing complete. Total images processed in chunks.")
