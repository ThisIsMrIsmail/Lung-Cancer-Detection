# %%
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# %%
# Training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
training_set = train_datagen.flow_from_directory(
    'cancer_dataset/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

# %%
# Test
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
    'cancer_dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

# %%
# Initialising the CNN
cnn = tf.keras.models.Sequential()

# %%
# Step1 - Convolution
cnn.add(tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=3,
    activation='relu',
    input_shape=[64, 64, 3]))

# %%
# Step2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# %%
# 2nd Convolutional layer
cnn.add(tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=3,
    activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# %%
# 3rd Convolutional layer
cnn.add(tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=3,
    activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# %%
# Step3 - Fallenting
cnn.add(tf.keras.layers.Flatten())

# %%
# Step4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# %%
# Step5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=4, activation='softmax'))

# %%
# Compile the CNN
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# %%
# Training the CNN on the Training set
# and evaluation it on the Test set
cnn.fit(x = training_set, validation_data=test_set, epochs=25)

# %%
# Making a Single prediction
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('cancer_dataset/prediction/squamous.cell.carcinoma.left.hilum/000119.png', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 0:
    prediction = 'Normal'
elif result[0][0] == 1:
    prediction = 'Adenocarcinoma'
elif result[0][0] == 2:
    prediction = 'Large carcinoma'
elif result[0][0] == 3:
    prediction = 'Squanmous carcinoma'

# %%
# last cell
print(prediction)

# %%
