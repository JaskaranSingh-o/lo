import pathlib

import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import RandomZoom, RandomRotation
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt

import tensorflow as tf
import utils
from classifier import *
from utils import *

# setting tensorflow variables
tf.get_logger().setLevel('ERROR')
print('TensorFlow version: ', tf.__version__)
print(f"Num of GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

# dataset paths
dataset_path = pathlib.Path('C:/Users/jaska/Downloads/real-vs-fake')
train_path = pathlib.Path('C:/Users/jaska/Downloads/real-vs-fake/Train')
val_path = pathlib.Path('C:/Users/jaska/Downloads/real-vs-fake/Valid')

batch_size = 32
IMG_SIZE = (224, 224)


# dataset generate function
def dataset():
    train_ = tf.keras.utils.image_dataset_from_directory(
        train_path,
        subset="training",
        validation_split=0.2,
        seed=100,
        image_size=IMG_SIZE,
        batch_size=batch_size)

    val_ = tf.keras.utils.image_dataset_from_directory(
        val_path,
        subset="validation",
        validation_split=0.2,
        seed=100,
        image_size=IMG_SIZE,
        batch_size=batch_size)

    return train_, val_


# normalizes the incoming data from the dataset function
def normalize(train):
    # standardize the dataset images
    norm_layer = layers.Rescaling(1. / 255)
    normalized_ = train.map(lambda x, y: (norm_layer(x), y))
    return normalized_


# trains the model on meso4
def train_meso4(epochs):
    # get the dataset
    train_, val_ = dataset()

    # normalize the dataset
    train_ = normalize(train_)

    # model definition
    model = Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Conv2D(8, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2), padding='same'),

        layers.Conv2D(8, (5, 5), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2), padding='same'),

        layers.Conv2D(16, (5, 5), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2), padding='same'),

        layers.Conv2D(16, (5, 5), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(4, 4), padding='same'),

        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(16),
        layers.LeakyReLU(alpha=0.1),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    # Adam
    learning_rate = 0.001
    optimizer = Adam(learning_rate=learning_rate)
    # compile the model with the Adam optimizer
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
    model.summary()
    print(len(model.layers))

    # gets the model training history
    history = model.fit(
        train_,
        validation_data=val_,
        epochs=epochs
    )

    # save the weights of the model
    model.save_weights(f"Meso4_New{epochs}_2.weights.h5")

    # plots the history
    utils.his_ploter("Meso4", history, epochs)


# plots the history on the graph


def his_ploter(name, history, epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

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
    plt.title(f'Training and Validation Loss - {name}')
    plt.show()