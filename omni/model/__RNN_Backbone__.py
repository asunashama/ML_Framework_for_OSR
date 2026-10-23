from omni import *


# CLDNN
# ===============================================================================================================
def create_conv_block(Lt: int):
    inputs = keras.Input(shape=(Lt, 2))

    block = keras.layers.Conv1D(64, 3, padding='same', use_bias=False)(inputs)
    block = keras.layers.BatchNormalization(axis=-1)(block)
    block = keras.layers.Activation('relu')(block)

    block = keras.layers.AveragePooling1D(pool_size=2)(block)

    block = keras.layers.Conv1D(128, 3, padding='same', use_bias=False)(block)
    block = keras.layers.BatchNormalization(axis=-1)(block)
    block = keras.layers.Activation('relu')(block)

    block = keras.layers.AveragePooling1D(pool_size=2)(block)

    block = keras.layers.Conv1D(256, 3, padding='same', use_bias=False)(block)
    block = keras.layers.BatchNormalization(axis=-1)(block)
    block = keras.layers.Activation('relu')(block)

    block = keras.layers.GlobalAveragePooling1D()(block)

    outputs = keras.layers.Reshape((1, keras.backend.int_shape(block)[-1]))(block)

    return keras.Model(inputs=inputs, outputs=outputs)


def RNN_Backbone(length: int, *, Lt: int = 128):
    """
    Lt: The input length for each timestep.
    Nt: The number of timesteps.
    ----------------------------------------------------------------------
    The input for each timestep is a signal segment of shape (Lt, 2).
    """
    assert length % Lt == 0
    Nt = length // Lt
    conv_block = create_conv_block(Lt)
    inputs = keras.layers.Input(shape=(length, 2))
    net = keras.layers.Reshape((Nt, Lt, 2))(inputs)
    net = [conv_block(net[:, k, :, :]) for k in range(Nt)]
    net = keras.layers.Concatenate(axis=-2)(net)
    outputs = keras.layers.LSTM(128)(net)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
