from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from flask_cors import CORS
import os
from predict import accient

# Set environment variables for encoding
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Helper function for predictions
def predict(values, dic):
    try:
        model = None
        if len(values) == 8:
            model = pickle.load(open('models/diabetes.pkl', 'rb'))
        elif len(values) == 26:
            model = pickle.load(open('models/breast_cancer.pkl', 'rb'))
        elif len(values) == 13:
            model = pickle.load(open('models/heart.pkl', 'rb'))
        elif len(values) == 18:
            model = pickle.load(open('models/kidney.pkl', 'rb'))
        elif len(values) == 10:
            model = pickle.load(open('models/liver.pkl', 'rb'))

        if model:
            values = np.asarray(values)
            return model.predict(values.reshape(1, -1))[0]
        else:
            return "Invalid input data for prediction."
    except Exception as e:
        return f"Prediction error: {str(e)}"

# Routes
@app.route("/")
def home():
    return render_template('home.html')

@app.route("/About")
def aboutPage():
    return render_template('About.html')

@app.route("/contact")
def contactPage():
    return render_template('contact.html')

@app.route("/diabetes")
def diabetesPage():
    return render_template('diabetes.html')

@app.route("/cancer")
def cancerPage():
    return render_template('breast_cancer.html')

@app.route("/heart")
def heartPage():
    return render_template('heart.html')

@app.route("/kidney")
def kidneyPage():
    return render_template('kidney.html')

@app.route("/liver")
def liverPage():
    return render_template('liver.html')

@app.route("/malaria")
def malariaPage():
    return render_template('malaria.html')

@app.route("/pneumonia")
def pneumoniaPage():
    return render_template('pneumonia.html')

@app.route("/cardio")
def cardioPage():
    return render_template('cardio.html')

# Prediction route for forms
@app.route("/predict", methods=['POST'])
def predictPage():
    try:
        to_predict_dict = request.form.to_dict()
        to_predict_list = list(map(float, to_predict_dict.values()))
        pred = predict(to_predict_list, to_predict_dict)
        return render_template('predict.html', pred=pred)
    except Exception as e:
        return render_template("home.html", message=f"Error: {str(e)}")

# Malaria prediction route
@app.route("/malariapredict", methods=['POST'])
def malariapredictPage():
    try:
        if 'image' in request.files:
            img = Image.open(request.files['image'])
            img = img.resize((36, 36))
            img = np.asarray(img).reshape((1, 36, 36, 3)).astype(np.float64)
            model = load_model("models/malaria.h5")
            pred = np.argmax(model.predict(img)[0])
            return render_template('malaria_predict.html', pred=pred)
        else:
            return render_template('malaria.html', message="No image uploaded.")
    except Exception as e:
        return render_template('malaria.html', message=f"Error: {str(e)}")

# Pneumonia prediction route
@app.route("/pneumoniapredict", methods=['POST'])
def pneumoniapredictPage():
    try:
        if 'image' in request.files:
            img = Image.open(request.files['image']).convert('L')
            img = img.resize((36, 36))
            img = np.asarray(img).reshape((1, 36, 36, 1)) / 255.0
            model = load_model("models/pneumonia.h5")
            pred = np.argmax(model.predict(img)[0])
            return render_template('pneumonia_predict.html', pred=pred)
        else:
            return render_template('pneumonia.html', message="No image uploaded.")
    except Exception as e:
        return render_template('pneumonia.html', message=f"Error: {str(e)}")

# Cardio prediction route
@app.route("/cardiopredict", methods=['POST'])
def cardiopredictPage():
    try:
        if 'image' in request.files:
            image_file = request.files['image']
            classifier = accient()  # Assuming accient class handles predictions
            result = classifier.accientImage(image_file)
            return render_template('cardio_predict.html', pred=result)
        else:
            return render_template('cardio.html', message="No image uploaded.")
    except Exception as e:
        return render_template('cardio.html', message=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
