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

    net = keras.layers.GlobalAveragePooling1D()(net)

    net = keras.layers.Dense(256)(net)
    net = keras.layers.ReLU()(net)
    outputs = keras.layers.Dense(128)(net)

    return keras.Model(inputs=inputs, outputs=outputs)


class SegmentSlicer(keras.layers.Layer):

    def __init__(self, length: int, num_of_segments: int):
        super().__init__()
        self._length = length
        self._num_of_segments = num_of_segments
        self._len_of_segments = length // num_of_segments

    def call(self, inputs, **kwargs):
        outputs = tf.reshape(
            inputs, shape=(-1, self._num_of_segments, self._len_of_segments, 2))
        return outputs


class SegmentEncoder(keras.layers.Layer):

    def __init__(self, length: int, num_of_segments: int, dim_of_projection: int):
        super().__init__()
        self._num_of_segments = num_of_segments
        self._len_of_segments = length // num_of_segments
        self._projector = create_residual_block(self._len_of_segments)
        self._dim_of_projection = self._projector.output.shape[-1]

    def call(self, segments, **kwargs):
        segments = tf.reshape(
            segments, shape=(-1, self._len_of_segments, 2))
        encoded_segments = self._projector(segments)
        encoded_segments = tf.reshape(
            encoded_segments, shape=(-1, self._num_of_segments, self._dim_of_projection))
        return encoded_segments


def MLP(inputs, structure, *, dropout: float = 0.1):
    block = inputs
    for num_of_units in structure:
        block = keras.layers.Dense(num_of_units, activation='gelu')(block)
        block = keras.layers.Dropout(dropout)(block)
    outputs = block
    return outputs


def CSTNN(length: int,
          num_of_classes: int,
          num_of_segments: int = 8,
          num_of_heads: int = 8,
          dim_of_projection: int = 128):
    assert length % num_of_segments == 0

    inputs = keras.layers.Input(shape=(length, 2))

    segments = SegmentSlicer(length, num_of_segments)(inputs)

    encoded_patches = SegmentEncoder(length, num_of_segments, dim_of_projection)(segments)

    for _ in range(1):
        # Layer Normalization
        block = keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        # Create a Multi-Head Attention
        attention_outputs = keras.layers.MultiHeadAttention(
            num_of_heads, dim_of_projection, dropout=0.1)(block, block)

        # Skip Connection
        shortcut = keras.layers.Add()([attention_outputs, encoded_patches])

        # Layer Normalization
        block = keras.layers.LayerNormalization(epsilon=1e-6)(shortcut)

        # MLP
        block = MLP(block, [dim_of_projection * 2, dim_of_projection])

        # Skip Connection
        encoded_patches = keras.layers.Add()([block, shortcut])

    # Create a [batch, dim_of_projection] Tensor
    representation = keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = keras.layers.Flatten()(representation)
    representation = keras.layers.Dropout(0.1)(representation)

    features = MLP(representation, [256, 128], dropout=0.1)

    outputs = keras.layers.Dense(num_of_classes, activation='softmax')(features)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
