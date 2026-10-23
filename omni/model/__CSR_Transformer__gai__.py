from omni import *
from omni.model import Backbone

#
def create_residual_block(length: int):
    inputs = keras.Input(shape=(length, 2))

    backbone = Backbone(length)
    _, dim_of_representation = backbone.output.shape

    net = backbone(inputs)

    net = keras.layers.Dense(dim_of_representation)(net)
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

    def __init__(self, length: int, num_of_segments: int):
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


def Transformer(length: int,
                num_of_classes: int,
                num_of_segments: int = 5,
                num_of_heads: int = 8,
                dim_of_projection: int = 128):
    assert length % num_of_segments == 0

    inputs = keras.layers.Input(shape=(length, 2))

    segments = SegmentSlicer(length, num_of_segments)(inputs)

    encoded_patches = SegmentEncoder(length, num_of_segments)(segments)

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

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization,Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# 位置编码函数
def positional_encoding(length, depth):
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (depths // 2)) / np.float32(depth))
    angle_rads = positions * angle_rates

    # 将 sin 应用于数组的偶数索引 (2i)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # 将 cos 应用于数组的奇数索引 (2i+1)
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return tf.convert_to_tensor(angle_rads, dtype=tf.float32)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Multi-Head Attention
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = x + inputs

    # Feed Forward Part
    x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = x + inputs

    # Normalize
    x = LayerNormalization()(x)
    return x

def transformer(length:int, num_of_classes: int):
    inputs = Input(shape=(length, 2), name='input')
    # 卷积层或线性层

    # 位置信息

    # Transformer Encoder 构建
    transformer_encoder1 = transformer_encoder(inputs, head_size=10, num_heads=4, ff_dim=128, dropout=0.5)
    transformer_encoder2 = transformer_encoder(inputs, head_size=10, num_heads=4, ff_dim=128, dropout=0.5)
    # transformer_encoder3 = transformer_encoder(inputs, head_size=10, num_heads=4, ff_dim=128, dropout=0.5)
    # transformer_encoder4 = transformer_encoder(inputs, head_size=10, num_heads=4, ff_dim=128, dropout=0.5)

    # 并行连接两个编码器的输出
    merged_output = Concatenate()([transformer_encoder1, transformer_encoder2])
    # Flatten 后接全连接层
    x = tf.keras.layers.Flatten()(merged_output)

    x = Dense(units=num_of_classes, activation='softmax')(x)

    # 创建模型
    model = Model(inputs=inputs, outputs=x)


    return model