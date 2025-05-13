from flask import Flask, request, render_template, flash, redirect, url_for
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import io
import base64
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'your-secret-key'  # Replace with a secure key
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Class names for predictions
CLASS_NAMES = ['Clean', 'Dusty', 'Bird-drop', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']

# Load models
MODELS = {
    'ResNet50': tf.keras.models.load_model('models/best_model_stage2.keras'),
    'MobileNetV2': tf.keras.models.load_model('models/mobilenetv2_model.keras')
}

def preprocess_image(image_path,model_name, target_size=(224, 224)):
    """Preprocess an image for model prediction."""
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    if model_name == 'ResNet50':
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    else:
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

def predict_image(image_path, model_name):
    """Predict the class of an image using the specified model."""
    img_array = preprocess_image(image_path,model_name)
    model = MODELS[model_name]
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100
    return predicted_class, confidence

def predict_both_models(image_path):
    """Run both models and return the prediction with the highest confidence."""
    results = {}
    for model_name in MODELS:
        predicted_class, confidence = predict_image(image_path, model_name)
        results[model_name] = {'class': predicted_class, 'confidence': confidence}
    
    # Determine the model with the highest confidence
    best_model = max(results, key=lambda x: results[x]['confidence'])
    return best_model, results[best_model]['class'], results[best_model]['confidence']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get selected model (if any)
        model_name = request.form.get('model')
        
        # Check if files are uploaded
        if 'images' not in request.files:
            flash('No images uploaded.', 'error')
            return redirect(url_for('index'))

        files = request.files.getlist('images')
        if not files or all(f.filename == '' for f in files):
            flash('No images selected.', 'error')
            return redirect(url_for('index'))

        results = []
        for file in files:
            if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Save the uploaded image with a unique filename
                filename = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()) + file.filename)
                file.save(filename)

                # Predict based on model selection
                if model_name and model_name in MODELS:
                    predicted_class, confidence = predict_image(filename, model_name)
                    best_model = model_name
                else:
                    # If no model selected or "Both Models" chosen, use both and pick the best
                    best_model, predicted_class, confidence = predict_both_models(filename)

                # Convert image to base64 for display
                with open(filename, 'rb') as img_file:
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
                img_src = f"data:image/jpeg;base64,{img_data}"

                results.append({
                    'filename': file.filename,
                    'predicted_class': predicted_class,
                    'confidence': f"{confidence:.2f}%",
                    'model_used': best_model,
                    'img_src': img_src
                })

                # Clean up uploaded file
                os.remove(filename)
            else:
                flash(f'Invalid file: {file.filename}. Only PNG, JPG, JPEG allowed.', 'error')

        # Pass model_name for display in results header
        return render_template('index.html', results=results, model_name=model_name)

    return render_template('index.html', results=None)

if __name__ == '__main__':
    app.run(debug=True)