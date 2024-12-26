import numpy as np
import pickle
from tensorflow.python.keras.models import Sequential
from tensorflow.python.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
from tensorflow.python.keras.utils import to_categorical

# Load the pickle files
X = []
y = []

# Load X and y chunks
for i in range(3):  # Adjust based on how many chunks you have
    X_chunk = pickle.load(open(f'X_chunk_{i}.pickle', 'rb'))
    y_chunk = pickle.load(open(f'y_chunk_{i}.pickle', 'rb'))
    
    X.extend(X_chunk)
    y.extend(y_chunk)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Normalize X to the range [0, 1]
X = X / 255.0

# Reshape X to add the channel dimension (grayscale images)
X = X.reshape(-1, 256, 256, 1)  # -1 will automatically infer the batch size

# If needed, convert y to categorical
# y = to_categorical(y, num_classes=2)

# Build the CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(256, 256, 1)),  # Correct input shape
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(64),
    Dense(1),
    Activation('sigmoid')  # Binary classification
])

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
model.fit(X, y, batch_size=16, epochs=10, validation_split=0.1)
