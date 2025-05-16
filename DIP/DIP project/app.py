
#Import necessary libraries
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
app = Flask(__name__)

# Ensure 'uploads' folder exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
model = load_model('tumor_detector.h5')

def predict_tumor(image_path):
    img = cv2.imread(image_path, 0)  # Load in grayscale
    if img is None:
        return "Invalid image"
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = img.reshape(1, 128, 128, 1)
    prediction = model.predict(img)[0][0]
    return 'Tumor Detected' if prediction > 0.5 else 'No Tumor'

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('result.html', result='No file part')
        file = request.files['image']
        if file.filename == '':
            return render_template('result.html', result='No selected file')
        # Basic file extension check (optional)
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            return render_template('result.html', result='Unsupported file format')
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        result = predict_tumor(filepath)
        return render_template('result.html', result=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)