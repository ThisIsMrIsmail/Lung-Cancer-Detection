import flask
import keras
import numpy as np
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
app = flask.Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return flask.render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    try:
        image = flask.request.files['image']
        image_path = os.path.join('api', 'images', image.filename)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        image.save(image_path)

        labels = ["Bengin", "Malignant", "Normal"]

        # Load and preprocess the image
        test_image = keras.preprocessing.image.load_img(image_path, target_size=(64, 64))
        test_image = keras.preprocessing.image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        # Load the model
        cnn = keras.models.load_model('api/models/v2-cnn-100epochs.h5')

        # Perform the prediction
        result = cnn.predict(test_image)

        if result is not None and len(result) > 0:
            prediction = labels[np.argmax(result[0])]
        else:
            prediction = "Error in prediction"

        os.remove(image_path)

    except Exception as e:
        prediction = f"Error: {e}"

    return flask.render_template('index.html', pred=prediction)