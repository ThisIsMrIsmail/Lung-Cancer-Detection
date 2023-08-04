import os
import keras
import flask

app = flask.Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return flask.render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    image = flask.request.files['image']
    image_path = "./images/" + image.filename
    image.save(image_path)

    cnn = keras.models.load_model("model.h5")

    import numpy as np

    classes = ["Adenocarcinoma", "Large cell carcinoma", "Normal", "Squamous cell carcinoma"]
    img = keras.utils.load_img(image_path, target_size=(64, 64))
    norm_img = keras.utils.img_to_array(img) / 255
    input_arr_img = np.array([norm_img])
    pred = np.argmax(cnn.predict(input_arr_img))
    print(classes[pred])
    os.remove(image_path)

    return flask.render_template('index.html', pred=classes[pred])


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)