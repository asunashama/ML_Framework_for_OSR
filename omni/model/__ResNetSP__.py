from tensorflow import keras
from typing import Optional


def convolution(inputs, k: Optional[int] = None):
    if k is None:
        k = keras.backend.int_shape(inputs)[-1]  # number of channels

    shortcut = keras.layers.Conv2D(
        k * 2, (1, 1), padding='same', use_bias=False)(inputs)
    shortcut = keras.layers.BatchNormalization(axis=-1)(shortcut)

    block = inputs

    block = keras.layers.Conv2D(
        k, (1, 1), padding='same', use_bias=False)(block)
    block = keras.layers.BatchNormalization(axis=-1)(block)
    block = keras.layers.Activation('relu')(block)

    block = keras.layers.SeparableConv2D(
        k, (3, 3), padding='same', use_bias=False)(block)
    block = keras.layers.BatchNormalization(axis=-1)(block)
    block = keras.layers.Activation('relu')(block)

    block = keras.layers.Conv2D(
        2 * k, (1, 1), padding='same', use_bias=False)(block)
    block = keras.layers.BatchNormalization(axis=-1)(block)

    block = keras.layers.Add()([block, shortcut])

    block = keras.layers.Activation('relu')(block)

    outputs = block

    return outputs


def downsampling(inputs):
    k = keras.backend.int_shape(inputs)[-1]

    shortcut = keras.layers.AveragePooling2D()(inputs)

    shortcut = keras.layers.Conv2D(
        k, (1, 1), padding='same', use_bias=False)(shortcut)
    shortcut = keras.layers.BatchNormalization(axis=-1)(shortcut)

    block = inputs

    block = keras.layers.Conv2D(
        k, (1, 1), padding='same', use_bias=False)(block)
    block = keras.layers.BatchNormalization(axis=-1)(block)
    block = keras.layers.Activation('relu')(block)

    block = keras.layers.SeparableConv2D(
        k, (3, 3), (2, 2), padding='same', use_bias=False)(block)
    block = keras.layers.BatchNormalization(axis=-1)(block)
    block = keras.layers.Activation('relu')(block)

    block = keras.layers.Conv2D(
        k, (1, 1), padding='same', use_bias=False)(block)
    block = keras.layers.BatchNormalization(axis=-1)(block)

    block = keras.layers.Add()([block, shortcut])

    block = keras.layers.Activation('relu', name=None)(block)

    outputs = block

    return outputs


def ResNetSP(size: int, nclass: int = 2):
    inputs = keras.Input(shape=(size, size, 1))

    net = inputs
    net = keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', use_bias=False)(net)
    net = keras.layers.BatchNormalization(axis=-1)(net)
    net = keras.layers.Activation('relu')(net)

    net = convolution(net)
    net = downsampling(net)
    net = convolution(net)
    net = downsampling(net)
    net = convolution(net)

    net = keras.layers.GlobalAveragePooling2D(name='GAP')(net)

    outputs = keras.layers.Dense(nclass, activation='softmax')(net)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
