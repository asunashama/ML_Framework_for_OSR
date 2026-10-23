from omni import *

import keras.layers

from omni import *


def convolution(inputs, k: Optional[int] = None):
    if k is None:
        k = keras.backend.int_shape(inputs)[-1]  # The number of channels

    shortcut = keras.layers.Conv1D(k * 2, 1, padding='same', use_bias=False)(inputs)
    shortcut = keras.layers.BatchNormalization(axis=-1)(shortcut)

    block = inputs

    block = keras.layers.Conv1D(k, 1, padding='same', use_bias=False)(block)
    block = keras.layers.BatchNormalization(axis=-1)(block)
    block = keras.layers.Activation('relu')(block)

    block = keras.layers.SeparableConv1D(k, 3, padding='same', use_bias=False)(block)
    block = keras.layers.BatchNormalization(axis=-1)(block)
    block = keras.layers.Activation('relu')(block)

    block = keras.layers.Conv1D(2 * k, 1, padding='same', use_bias=False)(block)
    block = keras.layers.BatchNormalization(axis=-1)(block)

    block = keras.layers.Add()([block, shortcut])

    block = keras.layers.Activation('relu')(block)

    outputs = block

    return outputs


def downsampling(inputs):
    k = keras.backend.int_shape(inputs)[-1]

    shortcut = keras.layers.AveragePooling1D(padding='same')(inputs)

    shortcut = keras.layers.Conv1D(k, 1, padding='same', use_bias=False)(shortcut)
    shortcut = keras.layers.BatchNormalization(axis=-1)(shortcut)

    block = inputs

    block = keras.layers.Conv1D(k, 1, padding='same', use_bias=False)(block)
    block = keras.layers.BatchNormalization(axis=-1)(block)
    block = keras.layers.Activation('relu')(block)

    block = keras.layers.SeparableConv1D(k, 3, 2, padding='same', use_bias=False)(block)
    block = keras.layers.BatchNormalization(axis=-1)(block)
    block = keras.layers.Activation('relu')(block)

    block = keras.layers.Conv1D(k, 1, padding='same', use_bias=False)(block)
    block = keras.layers.BatchNormalization(axis=-1)(block)

    block = keras.layers.Add()([block, shortcut])

    block = keras.layers.Activation('relu')(block)

    outputs = block

    return outputs


def create_residual_block(length: int):
    inputs = keras.Input(shape=(length, 2))

    net = inputs

    net = keras.layers.Conv1D(64, 7, 2, padding='same', use_bias=False)(net)
    net = keras.layers.BatchNormalization(axis=-1)(net)
    net = keras.layers.Activation('relu')(net)

    net = convolution(net)
    net = downsampling(net)
    net = convolution(net)

    outputs = keras.layers.GlobalAveragePooling1D()(net)

    return keras.Model(inputs=inputs, outputs=outputs)


def CGDNN(length: int,
          num_of_classes: int = 10,
          *, Lt: int = 128):
    assert length % Lt == 0

    extractor = create_residual_block(Lt)
    Nt = length // Lt

    inputs = keras.layers.Input(shape=(length, 2))
    segmented_inputs = keras.layers.Reshape((Nt, Lt, 2))(inputs)

    net = [extractor(segmented_inputs[:, k, :, :]) for k in range(Nt)]

    net = keras.layers.Concatenate(axis=-2)(net)
    net = keras.layers.GRU(128)(net)
    net = keras.layers.Dense(num_of_classes)(net)
    outputs = keras.layers.Softmax()(net)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
