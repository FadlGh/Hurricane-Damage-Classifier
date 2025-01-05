from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from preprocessing import train_dataset

# Split the dataset into training and validation sets
val_dataset = train_dataset.take(3600)
model_dataset = train_dataset.skip(3600)

for batch_images, batch_labels in val_dataset.take(1):
    print("Validation batch shape:", batch_images.shape)
    
# Model definition
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(256, 256, 1), kernel_regularizer=l2(0.01)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
model.fit(
    model_dataset,
    validation_data=val_dataset,
    steps_per_epoch=1500,
    epochs=20,  # Allow up to 20 epochs
    callbacks=[early_stopping]
)

# Validate dataset and model compatibility
for images, labels in train_dataset.take(1):
    predictions = model(images)  # Forward pass
    print("Model passed dataset compatibility test!")
