from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.regularizers import l2
import numpy as np
import pickle

X = []
y = []

for i in range(16):  # Edit number based on how many chunks you have
    with open(f'X_chunk_{i}.pickle', 'rb') as f:
        X_chunk = pickle.load(f)
    with open(f'y_chunk_{i}.pickle', 'rb') as f:
        y_chunk = pickle.load(f)
    
    X.extend(X_chunk)
    y.extend(y_chunk)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32).flatten()

X = X.reshape(-1, 256, 256, 1)

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(256, 256, 1),
           kernel_regularizer=l2(0.01)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(32, kernel_size=(3, 3), activation='relu', 
           kernel_regularizer=l2(0.01)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=7, validation_split=0.15)
