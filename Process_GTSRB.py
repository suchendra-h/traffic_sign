from __future__ import absolute_import, division, print_function, unicode_literals
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow_hub as hub
import matplotlib
import matplotlib.pyplot as plt
import zipfile
import csv
import random
import pathlib


training_data_dir = "C:/Users/sirch/PycharmProjects/TFTutorials/venv/dataset/training"
training_data_dir = pathlib.Path(training_data_dir)
image_count = len(list(training_data_dir.glob('*/*.jpg')))
training_class_names = np.array([item.name for item in training_data_dir.glob('*') if item.name != "LICENSE.txt"])
tmp_validation_data_dir = "C:/Users/sirch/PycharmProjects/TFTutorials/venv/dataset/validation"
tmp_validation_data_dir = pathlib.Path(tmp_validation_data_dir)
validation_class_names = np.array([item.name for item in tmp_validation_data_dir.glob('*') if item.name != "GT-final_test.csv"])

# The 1./255 is to convert from uint8 to float32 in range [0,1].
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
# image_gen_train = ImageDataGenerator(
#                     rescale=1./255,
#                     rotation_range=45,
#                     width_shift_range=.15,
#                     height_shift_range=.15,
#                     horizontal_flip=True,
#                     zoom_range=0.5
#                     )
validation_image_generator = ImageDataGenerator(rescale=1./255)
BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)
train_data_gen = image_generator.flow_from_directory(directory=str(training_data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes = list(training_class_names))
# train_data_gen = image_gen_train.flow_from_directory(directory=str(training_data_dir),
#                                                      batch_size=BATCH_SIZE,
#                                                      shuffle=True,
#                                                      target_size=(IMG_HEIGHT, IMG_WIDTH),
#                                                      classes = list(training_class_names))
validation_gen = validation_image_generator.flow_from_directory(directory=str(tmp_validation_data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True   ,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes = list(validation_class_names))
epochs = 30
model = models.Sequential()
model.add(layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
model.add(layers.MaxPooling2D())
model.add(layers.SpatialDropout2D(0.1))
model.add(layers.Conv2D(32, 3, padding='same', activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.SpatialDropout2D(0.1))
model.add(layers.Conv2D(64, 3, padding='same',activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.SpatialDropout2D(0.1))
model.add(layers.UpSampling2D())
model.add(layers.Conv2D(128, 3, padding='same', activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.SpatialDropout2D(0.1))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(43))
# ?, activation='softmax'
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()
# history = model.fit(labeled_training_ds, epochs=5, validation_data=(validation_images, validation_labels))
history = model.fit(train_data_gen, epochs=epochs, validation_data=validation_gen)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()