from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from preprocessing import train_dataset
import keras_tuner as kt

# Split the dataset into training and validation sets
val_dataset = train_dataset.take(3600)
model_dataset = train_dataset.skip(3600)

for batch_images, batch_labels in val_dataset.take(1):
    print("Validation batch shape:", batch_images.shape)

# Model definition
def build_model(hp):
    kernel_size_choice = hp.Choice("kernel_size", values=[3, 5]) 
    
    model = Sequential()
    
    # Input layer
    model.add(Conv2D(
        filters=hp.Int('filters_1', min_value=32, max_value=128, step=32),
        kernel_size=(kernel_size_choice, kernel_size_choice),
        activation='relu',
        input_shape=(128, 128, 1)
    ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Add multiple Conv2D layers based on hyperparameter choice
    for i in range(hp.Int('num_conv_layers', min_value=1, max_value=5, step=1)):
        model.add(Conv2D(
            filters=hp.Int(f'filters_{i+2}', min_value=32, max_value=128, step=32),
            kernel_size=(kernel_size_choice, kernel_size_choice),
            activation='relu'
        ))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Flatten and Dense layers
    model.add(Flatten())
    model.add(Dense(
        units=hp.Int('dense_units', min_value=64, max_value=256, step=64),
        activation='relu'
    ))
    model.add(Dropout(rate=hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(
        optimizer=Adam(hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5])),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,  # Number of hyperparameter combinations to try
    directory='my_dir',
    project_name='hyperparameter_tuning'
)

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
tuner.search(
    model_dataset,
    validation_data=val_dataset,
    steps_per_epoch=1500,
    epochs=10,
    callbacks=[early_stopping]
)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

best_model = tuner.hypermodel.build(best_hps)
history = best_model.fit(
    model_dataset,
    validation_data=val_dataset,
    steps_per_epoch=1500,
    epochs=20,
    callbacks=[early_stopping]
)

# Evaluate the model
val_loss, val_accuracy = best_model.evaluate(val_dataset)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

best_model.save('best_model.h5')
