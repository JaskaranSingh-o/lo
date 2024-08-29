import os
import numpy as np
import pandas as pd
from PIL import Image
from keras import Sequential
from matplotlib import pyplot as plt
from training import *
from classifier import *

# converse the images from path to an array
def image_to_array(img_path, w, h):
    img = Image.open(img_path)
    img = img.resize((w, h))
    img_array = np.asarray(img).reshape(1, w, h, 3)
    return img_array

# meso4 model visualization - for visualization
def meso4_model_Vis():
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
        layers.Dense(1, activation='sigmoid'),

    ])

    return model


# helper function to create a directory if it doesn't exist
def check_create_dir(path):
    try:
        isExist = os.path.exists(path)

        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(path)
            print(f"Directory {path} created.")

    except Exception as e:
        print(e)


# save the classification report to the file path provided
def classi_to_file(report, path, model_sel):
    check_create_dir(path)

    with open(f"{path}Classification_Report-{model_sel}.txt", "w+") as text_file:
        text_file.write(report)
