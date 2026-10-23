import keras.layers
from omni import *
import tensorflow as tf

class Dropout(keras.layers.Layer):
    def __init__(self, rate: float, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate  # Dropout 比率

    def call(self, inputs, training=None):
        if training:  # 只在训练阶段应用 Dropout
            random_tensor = tf.random.uniform(tf.shape(inputs))
            retain_prob = 1 - self.rate
            dropout_mask = (random_tensor >= retain_prob)
            return tf.where(dropout_mask, inputs / retain_prob, 0.0)
        else:
            return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({'rate': self.rate})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def convolution(inputs, k: Optional[int] = None):
    if k is None:
        k = keras.backend.int_shape(inputs)[-1]  # The number of channels 如果未手动传入输入通道数量，则直接获取通道数

    shortcut = keras.layers.Conv1D(k * 2, 1, padding='same', use_bias=False)(inputs)   # 用作残差层
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

    block = keras.layers.SeparableConv1D(k, 3, 2, padding='same', use_bias=False)(block)
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
    # net = Dropout(0.5)(net)


    # net = downsampling(net)
    # net = convolution(net)
    # net = downsampling(net)
    # net = convolution(net)

    outputs = keras.layers.GlobalAveragePooling1D(name='gap')(net)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model






def convolution_2d(inputs, k: Optional[int] = None):

    if k is None:
        k = keras.backend.int_shape(inputs)[-1]  # channels，应该是1

    shortcut = keras.layers.Conv2D(k * 2, (3,3), strides=1, padding='same', use_bias=False)(inputs)   # 用作残差层
    shortcut = keras.layers.BatchNormalization(axis=-1)(shortcut)

    block = inputs

    block = keras.layers.Conv2D(k, (3,3), strides=1, padding='same', use_bias=False)(block)
    block = keras.layers.BatchNormalization(axis=-1)(block)
    block = keras.layers.Activation('relu')(block)

    block = keras.layers.SeparableConv2D(k, (3,3), strides=1, padding='same', use_bias=False)(block)
    block = keras.layers.BatchNormalization(axis=-1)(block)
    block = keras.layers.Activation('relu')(block)

    block = keras.layers.Conv2D(2 * k, (3,3), strides=1 , padding='same', use_bias=False)(block)
    block = keras.layers.BatchNormalization(axis=-1)(block)

    block = keras.layers.Add()([block, shortcut])  # resnet的相加层，w+f(w)，在搭建网络、反向求导时按此作计算图，赋值后就是矩阵对应相加
    block = keras.layers.Activation('relu')(block)
    outputs = block

    return outputs


def downsampling_2d(inputs):
    k = keras.backend.int_shape(inputs)[-1]

    shortcut = keras.layers.AveragePooling2D(padding='same')(inputs)

    shortcut = keras.layers.Conv2D(k, (3,3),strides=1, padding='same', use_bias=False)(shortcut)
    shortcut = keras.layers.BatchNormalization(axis=-1)(shortcut)

    block = inputs

    block = keras.layers.Conv2D(k, (3,3),strides=1, padding='same', use_bias=False)(block)
    block = keras.layers.BatchNormalization(axis=-1)(block)
    block = keras.layers.Activation('relu')(block)

    block = keras.layers.SeparableConv2D(k, (3,3), 2, padding='same', use_bias=False)(block)
    block = keras.layers.BatchNormalization(axis=-1)(block)
    block = keras.layers.Activation('relu')(block)

    block = keras.layers.Conv2D(k, (3,3),strides=1, padding='same', use_bias=False)(block)
    block = keras.layers.BatchNormalization(axis=-1)(block)

    block = keras.layers.Add()([block, shortcut])

    block = keras.layers.Activation('relu')(block)

    outputs = block

    return outputs



def Backbone_2d(h:int = 15 ,w :int = 1024 ,channel:int = 1):
    inputs = keras.Input(shape=(h,w,channel))

    net = inputs

    net = keras.layers.Conv2D(128, (3,3), 2, padding='same', use_bias=False)(net)
    net = keras.layers.BatchNormalization(axis=-1)(net)
    net = keras.layers.Activation('relu')(net)

    net = convolution_2d(net)
    net = downsampling_2d(net)
    net = convolution_2d(net)

    # net = downsampling_2d(net)
    # net = convolution_2d(net)
    # net = downsampling_2d(net)
    # net = convolution_2d(net)

    outputs = keras.layers.GlobalAveragePooling2D(name='gap')(net)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model




#
#
# #-------------------------Picture-------------------------------
#
# def convolution(inputs, k: Optional[int] = None):
#     if k is None:
#         k = keras.backend.int_shape(inputs)[-1]  # The number of channels 如果未手动传入输入通道数量，则直接获取通道数
#
#     shortcut = keras.layers.Conv1D(k * 2, 1, padding='same', use_bias=False)(inputs)   # 用作残差层
#     shortcut = keras.layers.BatchNormalization(axis=-1)(shortcut)
#     block = inputs
#
#     block = keras.layers.Conv1D(k, 1, padding='same', use_bias=False)(block)
#     block = keras.layers.BatchNormalization(axis=-1)(block)
#     block = keras.layers.Activation('relu')(block)
#
#     block = keras.layers.SeparableConv1D(k, 3, padding='same', use_bias=False)(block)
#     block = keras.layers.BatchNormalization(axis=-1)(block)
#     block = keras.layers.Activation('relu')(block)
#
#     block = keras.layers.Conv1D(2 * k, 1, padding='same', use_bias=False)(block)
#     block = keras.layers.BatchNormalization(axis=-1)(block)
#
#     block = keras.layers.Add()([block, shortcut])  # resnet的相加层，w+f(w)，在搭建网络、反向求导时按此作计算图，赋值后就是矩阵对应相加
#
#     block = keras.layers.Activation('relu')(block)
#
#     outputs = block
#
#     return outputs
#
#
# def downsampling(inputs):
#     k = keras.backend.int_shape(inputs)[-1]
#
#     shortcut = keras.layers.AveragePooling1D(padding='same')(inputs)
#
#     shortcut = keras.layers.Conv1D(k, 1, padding='same', use_bias=False)(shortcut)
#     shortcut = keras.layers.BatchNormalization(axis=-1)(shortcut)
#
#     block = inputs
#
#     block = keras.layers.Conv1D(k, 1, padding='same', use_bias=False)(block)
#     block = keras.layers.BatchNormalization(axis=-1)(block)
#     block = keras.layers.Activation('relu')(block)
#
#     block = keras.layers.SeparableConv1D(k, 3, 2, padding='same', use_bias=False)(block)
#     block = keras.layers.BatchNormalization(axis=-1)(block)
#     block = keras.layers.Activation('relu')(block)
#
#     block = keras.layers.Conv1D(k, 1, padding='same', use_bias=False)(block)
#     block = keras.layers.BatchNormalization(axis=-1)(block)
#
#     block = keras.layers.Add()([block, shortcut])
#
#     block = keras.layers.Activation('relu')(block)
#
#     outputs = block
#
#     return outputs
#
#
# def Backbone(length: int):
#     inputs = keras.Input(shape=(length, 2))
#
#     net = inputs
#
#     net = keras.layers.Conv1D(128, 15, 2, padding='same', use_bias=False)(net)
#     net = keras.layers.BatchNormalization(axis=-1)(net)
#     net = keras.layers.Activation('relu')(net)
#
#     net = convolution(net)
#     net = downsampling(net)
#     net = convolution(net)
#
#     net = downsampling(net)
#     net = convolution(net)
#
#     outputs = keras.layers.GlobalAveragePooling1D(name='gap')(net)
#
#     model = keras.Model(inputs=inputs, outputs=outputs)
#     return model

if __name__ == '__main__':
    inputs = tf.ones((1, 1024, 2))
    model = Backbone(1024)
    output = model(inputs)
    model.summary()