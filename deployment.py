from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

model = load_model('best_model.h5')

def make_prediction(image_path):
    # Preprocess the image
    img = image.load_img(image_path, target_size=(128, 128), color_mode='grayscale')
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize

    prediction = model.predict(img_array)
    print("Prediction: ", prediction)

    # Convert prediction to actual class
    predicted_class = 'No Damage' if prediction[0] > 0.5 else 'Damage'
    print("Predicted Class: ", predicted_class)

image_name = str(input('Image name: '))
make_prediction(image_name)
