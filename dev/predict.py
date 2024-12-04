# Making a Single prediction
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from keras import saving
import os
import random

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

dataset_path = "datasets/v2-data"
labels = ["Bengin", "Malignant", "Normal"]

test_folders = os.listdir(f'{dataset_path}/test')

random_folder = random.choice(test_folders)
folder_path = os.path.join(f'{dataset_path}/test', random_folder)
images = os.listdir(folder_path)
random_image = random.choice(images)
image_path = os.path.join(folder_path, random_image)

test_image = image.load_img(image_path, target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

cnn = load_model('models/v2-cnn.h5')

# load cnn from the following: saving.save_model(cnn, 'cnn_model.keras')
cnn = saving.load_model('cnn_model.keras')

result = cnn.predict(test_image)
prediction = labels[np.argmax(result[0])]

print(image_path)
print(result)
print(prediction)