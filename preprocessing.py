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

training_data = []
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DIRECTORY, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                new_array = new_array / 255.0
                training_data.append([new_array, class_num])

                # Augment only for 'damage' (category 1) to solve imbalance
                if class_num == 0:  # Skip augmentation for 'no_damage'
                    continue
                
                augmentation_func = random.choice(augmentations)
                augmented_image = augmentation_func(new_array)
                training_data.append([augmented_image, class_num])
            except Exception as e:
                continue

            random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

create_training_data()
print(f"Total images: {len(training_data)}")

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()