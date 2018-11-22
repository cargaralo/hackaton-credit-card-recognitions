# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# PROCESS DATA

# plt.figure()
# plt.imshow(test_images[0])
# plt.imshow(train_images[0])
# plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

# SETUP LAYERS
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# COMPILE
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# TRAIN
# An epoch is an iteration over the entire x and y data provided.
model.fit(train_images, train_labels, epochs=5)

# EVALUATE
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# MAKE PREDICTIONS
predictions = model.predict(test_images)
