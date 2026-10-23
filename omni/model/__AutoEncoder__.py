import keras.layers
from omni import *
import tensorflow as tf

def convolution(inputs, k: Optional[int] = None):
    if k is None:
        k = keras.backend.int_shape(inputs)[-1]  # The number of channels 如果未手动传入输入通道数量，则直接获取通道数

    shortcut = keras.layers.Conv1D(k * 2, 1, padding='same', use_bias=False)(inputs)   # 用作残差层
    shortcut = keras.layers.BatchNormalization(axis=-1)(shortcut)

    block = inputs

    block = keras.layers.Conv1D(k, 1, padding='same', use_bias=False)(block)
    block = keras.layers.BatchNormalization(axis=-1)(block)
    block = keras.layers.Activation('relu')(block)

    block = keras.layers.Conv1D(k, 3, padding='same', use_bias=False)(block)
    block = keras.layers.BatchNormalization(axis=-1)(block)
    block = keras.layers.Activation('relu')(block)

    block = keras.layers.Conv1D(2 * k, 1, padding='same', use_bias=False)(block)
    block = keras.layers.BatchNormalization(axis=-1)(block)

    block = keras.layers.Add()([block, shortcut])  # resnet的相加层，w+f(w)，在搭建网络、反向求导时按此作计算图，赋值后就是矩阵对应相加
    block = keras.layers.Activation('relu')(block)
    outputs = block

    return outputs

def convolution2(inputs, k: Optional[int] = None):
    if k is None:
        k = keras.backend.int_shape(inputs)[-1]  # The number of channels 如果未手动传入输入通道数量，则直接获取通道数

    shortcut = keras.layers.Conv1D(k * 2, 1, padding='same', use_bias=False)(inputs)   # 用作残差层
    shortcut = keras.layers.BatchNormalization(axis=-1)(shortcut)

    block = inputs

    block = keras.layers.Conv1D(k, 1, padding='same', use_bias=False)(block)
    block = keras.layers.BatchNormalization(axis=-1)(block)
    block = keras.layers.Activation('relu')(block)

    block = keras.layers.Conv1D(k, 3, padding='same', use_bias=False)(block)
    block = keras.layers.BatchNormalization(axis=-1)(block)
    block = keras.layers.Activation('relu')(block)

    block = keras.layers.Conv1D(2 * k, 1, padding='same', use_bias=False)(block)
    block = keras.layers.BatchNormalization(axis=-1)(block)

    block = keras.layers.Add()([block, shortcut])  # resnet的相加层，w+f(w)，在搭建网络、反向求导时按此作计算图，赋值后就是矩阵对应相加
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

    block = keras.layers.Conv1D(k, 3, 2, padding='same', use_bias=False)(block)
    block = keras.layers.BatchNormalization(axis=-1)(block)
    block = keras.layers.Activation('relu')(block)

    block = keras.layers.Conv1D(k, 1, padding='same', use_bias=False)(block)
    block = keras.layers.BatchNormalization(axis=-1)(block)

    block = keras.layers.Add()([block, shortcut])

    block = keras.layers.Activation('relu')(block)

    outputs = block

    return outputs


def Encoder(length: int):
    inputs = keras.Input(shape=(length, 2))

    net = inputs

    net = keras.layers.Conv1D(128, 15, 2, padding='same', use_bias=False)(net)
    net = keras.layers.BatchNormalization(axis=-1)(net)
    net = keras.layers.Activation('relu')(net)

    net = convolution(net)
    net = downsampling(net)
    net = convolution(net)
    # net = Dropout(0.5)(net)


    net = downsampling(net)
    net = convolution(net)
    net = downsampling(net)
    # net = convolution(net)

    outputs = keras.layers.GlobalAveragePooling1D(name='gap')(net)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model






def deconvolution(inputs, k: Optional[int] = None):
    if k is None:
        k = keras.backend.int_shape(inputs)[-1] // 2  # 将通道数恢复到编码器的对应值
    # 残差层
    shortcut = keras.layers.Conv1DTranspose(k, 1, padding='same', use_bias=False)(inputs)
    shortcut = keras.layers.BatchNormalization(axis=-1)(shortcut)

    block = inputs

    # 转置卷积（对称于编码器的 1x1 卷积）
    block = keras.layers.Conv1DTranspose(k * 2, 1, padding='same', use_bias=False)(block)
    block = keras.layers.BatchNormalization(axis=-1)(block)
    block = keras.layers.Activation('relu')(block)

    # 转置分离卷积（对称于编码器的 SeparableConv1D）
    block = keras.layers.Conv1DTranspose(k, 3, padding='same', use_bias=False)(block)
    block = keras.layers.BatchNormalization(axis=-1)(block)
    block = keras.layers.Activation('relu')(block)

    # 转置卷积恢复到 k 通道
    block = keras.layers.Conv1DTranspose(k, 1, padding='same', use_bias=False)(block)
    block = keras.layers.BatchNormalization(axis=-1)(block)

    # 残差相加
    block = keras.layers.Add()([block, shortcut])
    block = keras.layers.Activation('relu')(block)
    outputs = block

    return outputs

def upsampling(inputs):
    k = keras.backend.int_shape(inputs)[-1]

    shortcut = keras.layers.Conv1DTranspose(k, 1, strides=2, padding='same', use_bias=False)(inputs)
    shortcut = keras.layers.BatchNormalization(axis=-1)(shortcut)

    block = inputs

    block = keras.layers.Conv1D(k, 1, padding='same', use_bias=False)(block)
    block = keras.layers.BatchNormalization(axis=-1)(block)
    block = keras.layers.Activation('relu')(block)

    block = keras.layers.Conv1DTranspose(k, 3, strides=2, padding='same', use_bias=False)(block)
    block = keras.layers.BatchNormalization(axis=-1)(block)
    block = keras.layers.Activation('relu')(block)

    block = keras.layers.Conv1D(k, 1, padding='same', use_bias=False)(block)
    block = keras.layers.BatchNormalization(axis=-1)(block)

    block = keras.layers.Add()([block, shortcut])

    block = keras.layers.Activation('relu')(block)

    outputs = block

    return outputs


def Backbone(length: int):
    inputs = keras.Input(shape=(length, 2))

    net = inputs

    net = keras.layers.Conv1D(128, 15, 2, padding='same', use_bias=False)(net)
    net = keras.layers.BatchNormalization(axis=-1)(net)
    net = keras.layers.Activation('relu')(net)

    net = convolution(net)
    net = downsampling(net)
    net = convolution(net)
    net = downsampling(net)
    net = downsampling(net)

    net = deconvolution(net)

    net = downsampling(net)
    net = upsampling(net)

    outputs = keras.layers.GlobalAveragePooling1D(name='gap')(net)
    net = keras.layers.Conv1DTranspose(2, 15, strides=2, padding='same', use_bias=False)(net)
    net = keras.layers.BatchNormalization(axis=-1)(net)
    net = keras.layers.Activation('relu')(net)
    outputs = net

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def Auto_encoder(length: int):
    inputs = keras.Input(shape=(length, 2))

    net = inputs

    net = keras.layers.Conv1D(128, 15, 2, padding='same', use_bias=False)(net)
    net = keras.layers.BatchNormalization(axis=-1)(net)
    net = keras.layers.Activation('relu')(net)

    net = convolution(net)
    net = downsampling(net)
    net = convolution(net)

    net = keras.layers.GlobalAveragePooling1D(name='gap')(net)



    embeddings = keras.layers.Dense(dim_of_representation, use_bias=False)(representations)
    embeddings = keras.layers.BatchNormalization(momentum=0.99)(embeddings)
    embeddings = keras.layers.ReLU()(embeddings)
    embeddings = keras.layers.Dense(64)(embeddings)
    embeddings = keras.layers.BatchNormalization(momentum=0.99)(embeddings)
    embeddings = keras.layers.ReLU()(embeddings)

    outputs1 = keras.layers.Dense(num_of_features)(embeddings)



    net = deconvolution(net)
    net = upsampling(net)
    net = deconvolution(net)

    net = keras.layers.Conv1DTranspose(2, 15, strides=2, padding='same', use_bias=False)(net)
    net = keras.layers.BatchNormalization(axis=-1)(net)
    net = keras.layers.Activation('relu')(net)
    outputs2 = net

    model = keras.Model(inputs=inputs, outputs=(outputs1,outputs2))
    return model





if __name__ == '__main__':
    inputs = tf.ones((1, 1024, 2))
    model = Backbone(1024)
    output = model(inputs)
    model.summary()