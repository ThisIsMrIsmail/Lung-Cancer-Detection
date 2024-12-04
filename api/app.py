from flask import Flask, render_template, request
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import numpy as np
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

IMAGE_UPLOAD_DIR = os.path.join('api', 'images')
MODEL_PATH = os.path.join('api', 'models', 'v2-cnn-100epochs.h5')
LABELS = ["Benign", "Malignant", "Normal"]

os.makedirs(IMAGE_UPLOAD_DIR, exist_ok=True)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    prediction = "Error: Unable to process the image."
    try:
        if 'image' not in request.files:
            return render_template('index.html', pred="No image uploaded.")
        
        image = request.files['image']
        if image.filename == '':
            return render_template('index.html', pred="No file selected.")
        
        image_path = os.path.join(IMAGE_UPLOAD_DIR, image.filename)
        image.save(image_path)

        test_image = load_img(image_path, target_size=(64, 64))
        test_image = img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        cnn = load_model(MODEL_PATH)
        result = cnn.predict(test_image)

        if result is not None and len(result) > 0:
            prediction = LABELS[np.argmax(result[0])]
        else:
            prediction = "Error in prediction"

    except Exception as e:
        prediction = f"Error: {e}"
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

    return render_template('index.html', pred=prediction)

if __name__ == '__main__':
    app.run(debug=True)