# imports
import os

from keras import layers
from keras.models import Model
from tensorflow.keras.layers import Input, Activation, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, \
    Dropout, LeakyReLU
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.optimizers import Adam

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'


# classifier class to access certin model methods
class Classifier:
    def __init__(self):
        self.model = 0

    # predict on the data
    def predict(self, x):
        return self.model.predict(x)

    # load the pre-trained weights
    def load(self, path):
        self.model.load_weights(path)

    # return the summary of the model (layers)
    def summary(self):
        return self.model.summary()

    # save the trained weights to file path
    def save_weights(self, path):
        self.model.save_weights(path)

    # return the filters
    def filters(self, layer_num):
        outputs = [self.model.layers[i].output for i in layer_num]
        return Model(inputs=self.model.inputs, outputs=outputs)

    # return the feature maps and layer numbers
    def get_conv_layers(self):
        conv_features = []
        layer_ = []
        for i, layer in enumerate(self.model.layers):
            # check for convolutional layer
            if 'conv' not in layer.name:
                continue
            conv_features.append(i)
            layer_.append(layer)
        return conv_features, layer_

class Meso4(Classifier):
    def __init__(self, learning_rate=0.001):
        self.model = self.init_model()
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

    # initializing the model with Adam optimizer
    def init_model(self):
        x = Input(shape=(256, 256, 3)) # image input layer

        # block 1 - convolution layer, normalization and maxpooling
        x1 = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)

        # block 2 - convolution layer, normalization and maxpooling
        x2 = Conv2D(8, (5, 5), padding='same', activation='relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)

        # block 3 - convolution layer, normalization and maxpooling
        x3 = Conv2D(16, (5, 5), padding='same', activation='relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)

        # block 4 - convolution layer, normalization and maxpooling
        x4 = Conv2D(16, (5, 5), padding='same', activation='relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)

        # flatten the input
        y = Flatten()(x4)
        y = Dropout(0.5)(y) # dropout 50% of neurons
        y = Dense(16)(y) # folly connected layer of 16 classes
        y = LeakyReLU(alpha=0.1)(y) # applied a small gradient when network isn't active
        y = Dropout(0.5)(y) # dropout 50% of neurons
        y = Dense(1, activation='sigmoid')(y) # fully connected layer of just 1 class, either fake or real

        # return the model
        return KerasModel(inputs=x, outputs=y)
