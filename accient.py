import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

class Accient:
    def __init__(self, image_file):
        self.image_file = image_file

    def process_image(self):
        # Load the model
        model = load_model('C:/Users/cnkum/OneDrive/Documents/Medical_Diagnosis-main/models/CardioGPT.h5')

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(self.image_file.filename))
        self.image_file.save(file_path)

        # Load and preprocess the image
        test_image = image.load_img(file_path, target_size=(224, 224))  # Ensure this matches model input size
        test_image = image.img_to_array(test_image)
        test_image = test_image / 255  # Normalize the image
        test_image = np.expand_dims(test_image, axis=0)  # Expand dimensions to fit the model input

        # Clear any session to avoid conflicts
        from tensorflow.keras.backend import clear_session
        clear_session()

        # Make prediction
        preds = model.predict(test_image)

        # Decode prediction
        preds = np.argmax(preds, axis=1)
        if preds == 0:
            return "Atrial Fibrillation"
        elif preds == 1:
            return "Atrial Flutter"
        elif preds == 2:
            return "Atrial Tachycardia"
        elif preds == 3:
            return "Sinus Arrhythmia"
        elif preds == 4:
            return "Sinus Bradycardia"
        elif preds == 5:
            return "Sinus Tachycardia"
