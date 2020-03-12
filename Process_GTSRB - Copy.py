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

BATCH_SIZE = 64
IMG_HEIGHT = 224
IMG_WIDTH = 224

def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  return parts[-2] == training_class_names

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

def valid_process_path(file_path):
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img


# Set directories to find training and validation images and labels
training_data_dir = "C:/Users/sirch/PycharmProjects/TFTutorials/venv/dataset/training"
training_data_dir = pathlib.Path(training_data_dir)

training_class_names = np.array([item.name for item in training_data_dir.glob('*') if item.name != "LICENSE.txt"])

tmp_validation_data_dir = "C:/Users/sirch/PycharmProjects/TFTutorials/venv/dataset/validation"
# tmp_validation_data_dir =pathlib.Path(tmp_validation_data_dir)
# valid_ds = tf.data.Dataset.list_files(str(tmp_validation_data_dir/'*'))
tmp_validation_data_files = [tmp_validation_data_dir + "/" + f for f in os.listdir(tmp_validation_data_dir) if f.endswith(".jpg")]
validation_images = []
# Load validation images
for idx, file in enumerate(tmp_validation_data_files):
    validation_images.append(decode_img(tf.io.read_file(file)))
    if idx >= 100:
        break
# validation_images = valid_ds.map(tf.io.read_file).map(decode_img)

# print("Validation images count:", len(validation_images))
#Load CSV labels for validation
tmp_validation_labels_csv = os.path.join("C:/Users/sirch/PycharmProjects/TFTutorials/venv/dataset/validation/GT-final_test.csv")
validation_labels = []
# STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

with open(tmp_validation_labels_csv) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    line_count = 0
    for row in csv_reader:
        if (line_count > 0):
            validation_labels.append(int(row[7]))
        line_count += 1
print("Test labels count:", len(validation_labels))

# validation_labels = np.array(validation_labels)
# validation_images = np.array(validation_images)
# The 1./255 is to convert from uint8 to float32 in range [0,1].
# image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
# validation_image_generator = ImageDataGenerator(rescale=1./255)

# train_data_gen = image_generator.flow_from_directory(directory=str(training_data_dir),
#                                                      batch_size=BATCH_SIZE,
#                                                      shuffle=True,
#                                                      target_size=(IMG_HEIGHT, IMG_WIDTH),
#                                                      classes = list(training_class_names),
#                                                      class_mode = "categorical")
list_ds = tf.data.Dataset.list_files(str(training_data_dir/'*/*'))


# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
labeled_training_ds = list_ds.map(process_path)

# labeled_validation_ds = (valid_ds.map(valid_process_path), np.array(validation_labels))
labeled_validation_ds = (np.array(validation_images), np.array(validation_labels))
print("HELLO")
def prepare_for_training(ds, cache=False, shuffle_buffer_size=1000):
  # This is a small dataset, only load it once, and keep it in memory.
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory.
  print("Training")
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  ds = ds.shuffle(buffer_size=shuffle_buffer_size)

  # Repeat forever
  ds = ds.repeat()

  ds = ds.batch(BATCH_SIZE)

  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  #ds = ds.prefetch(buffer_size=AUTOTUNE)

  return ds

train_ds = prepare_for_training(labeled_training_ds)

train_images, train_labels = next(iter(train_ds))


# validation_ds = prepare_for_training(labeled_validation_ds)
# test_images, test_labels = next(iter(tmp_ds))



model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(43, activation='softmax'))

# train = train_ds.next()[0]
# ret = train_ds.next()[1]
# print(train.shape, ret.shape)

# model.compile(optimizer='adam',
#               loss=tf.keras.losses.CategoricalCrossentropy(),
#               metrics=['accuracy'])
#history = model.fit(labeled_training_ds, epochs=5, validation_data=(validation_images, validation_labels))
# model.fit(train_ds, epochs=5,steps_per_epoch=STEPS_PER_EPOCH,
#           validation_data=labeled_validation_ds)
optimizer = tf.keras.optimizers.Adam()
losser = tf.keras.losses.CategoricalCrossentropy()
print(labeled_validation_ds[0].shape)
for train_X, train_y in train_ds:
    with tf.GradientTape() as tape:
        train_loss = losser(
            y_pred=model(train_X),
            y_true=train_y
        )
    grad = tape.gradient(train_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grad, model.trainable_variables))
    # print(labeled_validation_ds[0].shape)
    # pred = model(labeled_validation_ds[0])
    # print(pred)
    # val_loss = losser(labeled_validation_ds[1][:100], pred)
    # print('loss in validation : {}\n'
    #       'loss in train: {}'.format(val_loss, train_loss))
    print(
          'loss in train: {}'.format(train_loss))