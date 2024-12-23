import os
import cv2

resize_height = 256
resize_width = 256

directory = 'archive'

for folder, subfolders, filenames in os.walk(directory):
    for filename in filenames:
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Failed to load {img_path}")
            continue

        resized_img = cv2.resize(img, (resize_width, resize_height))
        resized_img = resized_img / 255.0

        cv2.imwrite(img_path, resized_img * 255)
        print(resized_img.shape) 
        