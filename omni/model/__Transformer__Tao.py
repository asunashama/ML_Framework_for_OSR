import numpy as np
from omni import *
from omni.model import Backbone

# 位置编码函数
# def positional_encoding(length, depth):
#     positions = np.arange(length)[:, np.newaxis]
#     depths = np.arange(depth)[np.newaxis, :]
#     angle_rates = 1 / np.power(10000, (2 * (depths // 2)) / np.float32(depth))
#     angle_rads = positions * angle_rates
#
#     # 将 sin 应用于数组的偶数索引 (2i)
#     angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
#
#     # 将 cos 应用于数组的奇数索引 (2i+1)
#     angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
#
#     return tf.convert_to_tensor(angle_rads, dtype=tf.float32)


def create_residual_block(length: int):
    inputs = keras.Input(shape=(length, 2))

    backbone = Backbone(length)
    _, dim_of_representation = backbone.output.shape

    net = backbone(inputs)

    net = keras.layers.Dense(dim_of_representation)(net)
    net = keras.layers.ReLU()(net)

    outputs = keras.layers.Dense(128)(net)

    return keras.Model(inputs=inputs, outputs=outputs)





class Embedding(keras.layers.Layer):
    def __init__(self, num_of_token:int, num_of_features:int):
        super().__init__()
        self._projector = create_residual_block(num_of_token,num_of_features)

    def call(self, input, **kwargs):
        embeddings = self._projector(input)
        return embeddings


def MLP(inputs, structure, *, dropout: float = 0.1):
    block = inputs
    for num_of_units in structure:
        block = keras.layers.Dense(num_of_units, activation='gelu')(block)
        block = keras.layers.Dropout(dropout)(block)
    outputs = block
    return outputs


def Transformer(
                num_of_classes: int,
                num_of_heads: int = 8,
                dim_of_projection: int = 64,
                num_of_token:int = 1000,
                num_of_features:int = 4):

    inputs = keras.layers.Input(shape=(num_of_token, num_of_features))
    embeddings = Embedding(num_of_token, num_of_features)(inputs) #8,128

    for _ in range(1):
        # Layer Normalization
        block = keras.layers.LayerNormalization(epsilon=1e-6)(embeddings)
        # Create a Multi-Head Attention
        attention_outputs = keras.layers.MultiHeadAttention(
            num_of_heads, dim_of_projection, dropout=0.1)(block,block) # 做自注意力
        # Skip Connection
        shortcut = keras.layers.Add()([attention_outputs, embeddings])
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
