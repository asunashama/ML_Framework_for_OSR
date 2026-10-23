from tensorflow import keras
from typing import Optional


def convolution(inputs, k: Optional[int] = None):
    if k is None:
        k = keras.backend.int_shape(inputs)[-1]  # number of channels

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

    block = keras.layers.Activation('relu', name=None)(block)

    outputs = block

    return outputs


def ConvLSTM(length: int, nclass: int = 2):
    Lt = 128  # Lt -> each step will do conv on shape (Lt, 2)
    Nt = length // Lt  # Nt := N_timesteps

    inputs = keras.Input(shape=(length, 2))
    net = keras.layers.Reshape((Nt, 1, Lt, 2))(inputs)
    net = keras.layers.ConvLSTM2D(64, (1, 7), padding='same')(net)

    net = keras.layers.Reshape((Lt, 64))(net)

    net = convolution(net)
    net = downsampling(net)
    net = convolution(net)

    # net = downsampling(net)
    # net = convolution(net)

    net = keras.layers.GlobalAveragePooling1D(name='GAP')(net)

    outputs = keras.layers.Dense(nclass, activation='softmax')(net)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
