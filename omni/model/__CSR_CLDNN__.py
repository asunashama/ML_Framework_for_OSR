from omni import *
from omni.model import Backbone


# CLDNN
# ===============================================================================================================
def create_residual_extractor(length: int):
    inputs = keras.Input(shape=(length, 2))
    backbone = Backbone(length)
    block = backbone(inputs)
    outputs = keras.layers.Reshape((1, keras.backend.int_shape(block)[-1]))(block)
    return keras.Model(inputs=inputs, outputs=outputs)


def CSRCLDNN(length: int, num_of_classes: int = 10, *, len_of_segments: int = 128):
    extractor = create_residual_extractor(len_of_segments)

    inputs = keras.layers.Input(shape=(length, 2))

    _inputs = keras.layers.Reshape((length // len_of_segments, len_of_segments, 2))(inputs)
    _representations = [extractor(_inputs[:, k, :, :]) for k in range(length // len_of_segments)]
    net = keras.layers.Concatenate(axis=-2)(_representations)

    net = keras.layers.LSTM(256)(net)

    # net = keras.layers.Dense(256, use_bias=False)(net)
    # net = keras.layers.BatchNormalization(momentum=0.99)(net)
    # net = keras.layers.ReLU()(net)
    # net = keras.layers.Dense(128, name='feature', use_bias=False)(net)
    # net = keras.layers.BatchNormalization(momentum=0.99)(net)

    net = keras.layers.Dense(256)(net)
    net = keras.layers.ReLU()(net)
    net = keras.layers.Dense(128)(net)

    net = keras.layers.Dense(num_of_classes)(net)
    outputs = keras.layers.Softmax()(net)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
