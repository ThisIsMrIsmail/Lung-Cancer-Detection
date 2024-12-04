
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import saving

dataset_path = "datasets/v2-data"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

training_set = train_datagen.flow_from_directory(
    f'{dataset_path}/train',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# Test
val_datagen = ImageDataGenerator(rescale=1./255)
val_set = val_datagen.flow_from_directory(
    f'{dataset_path}/test',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# Initialising the CNN
cnn = tf.keras.models.Sequential()

# 1st Convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# 2nd Convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# 3rd Convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step3 - Fallenting
cnn.add(tf.keras.layers.Flatten())

# Step4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step5 - Dropout Layer
cnn.add(tf.keras.layers.Dropout(0.5))  # Dropout to prevent overfitting

# Step6 - Output Layer
cnn.add(tf.keras.layers.Dense(units=3, activation='softmax'))

# Compile the CNN
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

cnn.summary()

cnn.fit(x = training_set, validation_data=val_set, epochs=50)

print(training_set.class_indices)

saving.save_model(cnn, 'cnn_model.keras')
# cnn.save('models/v2-cnn.h5')