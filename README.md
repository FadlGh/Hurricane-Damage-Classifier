# Hurricane Damage Classifier

This project utilizes deep learning techniques to classify satellite images into two categories: **Damage** and **No Damage**. The classifier is built using a Convolutional Neural Network (CNN) implemented with TensorFlow and Keras. Hyperparameter tuning was conducted using Keras Tuner, and the model was trained on a custom dataset sourced from Kaggle. 
Best validation accuracy: **93%**

## Table of Contents

- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Fine-Tuning](#fine-tuning)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [References](#references)

## Dataset

The dataset comprises satellite images from Texas after Hurricane Harvey, divided into two categories: **damage** and **no_damage**. The dataset includes:

- **Damaged Images**: 14,000
- **Non-Damaged Images**: 7,000

The images are split into training and validation sets.

*Note: The dataset is available on Kaggle under the title "Satellite Images of Hurricane Damage".*  
[Kaggle Dataset](https://www.kaggle.com/datasets/kmader/satellite-images-of-hurricane-damage)

## Model Architecture

The CNN model is constructed with the following layers:

1. **Input Layer**: Accepts images resized to 128x128 pixels with 3 color channels (RGB).
2. **Convolutional Layers**: Three convolutional layers with 32, 64, and 128 filters, respectively, each followed by ReLU activation and max-pooling.
3. **Flatten Layer**: Converts the 2D feature maps into a 1D feature vector.
4. **Fully Connected Layers**: Two dense layers with 128 and 64 units, respectively, both with ReLU activation.
5. **Output Layer**: A dense layer with a single unit and sigmoid activation for binary classification.

## Preprocessing

The preprocessing steps include:

- **Resizing**: All images are resized to 128x128 pixels.
- **Normalization**: Pixel values are scaled to the range [0, 1].
- **Data Augmentation**:  
  To increase dataset diversity, transformations such as rotation, zoom, horizontal flipping, and other modifications were applied to generate unique new images.

## Training

The model is compiled with the following configurations:

- **Loss Function**: Binary Cross-Entropy
- **Optimizer**: Adam
- **Metrics**: Accuracy

Training is conducted over 20 epochs with a batch size of 32. Early stopping is implemented to prevent overfitting.

## Fine-Tuning

Hyperparameter tuning is performed using Keras Tuner to identify the optimal number of units in the dense layers and the learning rate for the optimizer. This process involves:

1. Defining a hypermodel that specifies the search space for hyperparameters.
2. Utilizing the Hyperband algorithm to efficiently search for the best hyperparameter combinations.
3. Training multiple models with different hyperparameters and selecting the best-performing model based on validation accuracy.

## Usage

To utilize the trained model for prediction:

1. Ensure all dependencies are installed as listed in the `requirements.txt` file.
2. Load the trained model using Keras's `load_model` function.
3. Preprocess the input image to match the training data format.
4. Use the model's `predict` method to obtain the classification result.

## References

- Kaggle Dataset: [Satellite Images of Hurricane Damage](https://www.kaggle.com/datasets/kmader/satellite-images-of-hurricane-damage)
- Keras Documentation: [Keras Tuner](https://keras.io/keras_tuner/)

