from keras.layers import GlobalAveragePooling2D, Dense, Reshape, Multiply, GlobalMaxPooling2D, Conv2D, ReLU, Add, \
    Concatenate
from omni import *
from omni.model import Backbone
import keras
import tensorflow as tf

class ChannelAttention_MLP_2D(keras.layers.Layer):
    def __init__(self, reduction_ratio=16, **kwargs):
        super(ChannelAttention_MLP_2D, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio # 缩减率，缩减为原channel的 1/ratio

    def build(self, input_shape):
        channel_dim = input_shape[-1]
        self.global_avg_pool = GlobalAveragePooling2D()
        self.global_max_pool = GlobalMaxPooling2D()
        self.dense1 = Dense(units=channel_dim // self.reduction_ratio, activation='relu', use_bias=False) # //是整除 32//15 = 2
        self.dense2 = Dense(units=channel_dim, use_bias=False)
        self.reshape = Reshape((1, 1, channel_dim))
        super(ChannelAttention_MLP_2D, self).build(input_shape)

    def call(self, inputs):
        avg_pooled = self.global_avg_pool(inputs)
        max_pooled = self.global_max_pool(inputs)

        avg_fc1 = self.dense1(avg_pooled)
        avg_fc2 = self.dense2(avg_fc1)

        max_fc1 = self.dense1(max_pooled)
        max_fc2 = self.dense2(max_fc1)
        print(max_fc2.shape)
        print(avg_fc2.shape)

        # 注意力融合
        attention = tf.keras.layers.Add()([avg_fc2, max_fc2])  # 求和
        print(attention.shape)

        attention = tf.keras.layers.Activation('sigmoid')(attention)  # 激活
        attention = tf.keras.layers.Reshape((1, 1, inputs.shape[-1]))(attention)  # 调整形状
        print(attention.shape)

        return keras.layers.Multiply()([inputs, attention])

    def get_config(self):
        config = super(ChannelAttention_MLP_2D, self).get_config()
        config.update({
            "reduction_ratio": self.reduction_ratio,
        })
        return config







class ChannelAttention_Conv_2D(keras.layers.Layer):
    def __init__(self, reduction_ratio=16, **kwargs):
        super(ChannelAttention_Conv_2D, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channel_dim = input_shape[-1]
        reduced_channel = channel_dim // self.reduction_ratio  # 缩减后的通道数

        self.global_avg_pool = GlobalAveragePooling2D()
        self.global_max_pool = GlobalMaxPooling2D()
        self.conv1 = Conv2D(reduced_channel, kernel_size=1, activation='relu', use_bias=False)  # 降维
        self.conv2 = Conv2D(self.channel, kernel_size=1, activation='sigmoid', use_bias=False)  # 升维


        self.reshape = Reshape((1, 1, channel_dim))
        super(ChannelAttention_Conv_2D, self).build(input_shape)

    def call(self, inputs):
        avg_pooled = self.global_avg_pool(inputs)
        max_pooled = self.global_max_pool(inputs)

        avg_conv = self.conv1(avg_pooled)
        avg_conv = self.conv2(avg_conv)
        max_conv = self.conv1(max_pooled)
        max_conv = self.conv2(max_conv)

        # 注意力融合
        attention = tf.keras.layers.Add()([avg_conv, max_conv])  # 求和
        print(attention.shape)

        attention = tf.keras.layers.Activation('sigmoid')(attention)  # 激活
        attention = tf.keras.layers.Reshape((1, 1, inputs.shape[-1]))(attention)  # 调整形状
        print(attention.shape)

        return Multiply()([inputs, attention])

    def get_config(self):
        config = super(ChannelAttention_Conv_2D, self).get_config()
        config.update({
            "reduction_ratio": self.reduction_ratio,
        })
        return config


class SpatialAttention_Conv_2D(keras.layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):

        super(SpatialAttention_Conv_2D, self).__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        # 使用一个 7x7 的卷积层
        self.conv = Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            strides=1,
            padding='same',
            use_bias=False
        )
        super(SpatialAttention_Conv_2D, self).build(input_shape)

    def call(self, inputs, **kwargs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)  # tf目前没有通道平均池化函数，只能这样写了
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)   # 通道最大池化
        concat = Concatenate(axis=-1)([avg_pool, max_pool])        # 拼接
        attention = self.conv(concat)
        attention = tf.keras.layers.Activation('sigmoid')(attention)  # 激活


        return Multiply()([inputs, attention])

    def get_config(self):
        config = super(SpatialAttention_Conv_2D, self).get_config()
        config.update({
            "kernel_size": self.kernel_size,
        })
        return config

def CBAM(input_tensor, reduction_ratio=16, kernel_size=7):
    net = ChannelAttention_MLP_2D(reduction_ratio=reduction_ratio)(input_tensor)
    net = SpatialAttention_Conv_2D(kernel_size=kernel_size)(net)
    return net









def VMD_S_Backbone(inputs, k: Optional[int] = None):
    if k is None:
        k = keras.backend.int_shape(inputs)[-1]  # The number of channels 如果未手动传入输入通道数量，则直接获取通道数

    # 先7x1再3x3，再1x7再3x3 先基础卷积
    net = keras.layers.Conv2D(filters=2*k, kernel_size=(7,1), strides=1, padding=1, use_bias=False)(inputs)
    net = keras.layers.BatchNormalization(axis=-1)(net)
    net = keras.layers.Activation('relu')(net)

    net = keras.layers.Conv2D(filters=2*k, kernel_size=(3,3), strides=2, padding=1, use_bias=False)(net)
    net = keras.layers.BatchNormalization(axis=-1)(net)
    net = keras.layers.Activation('relu')(net)


    net = keras.layers.Conv2D(filters=2*k, kernel_size=(1,7), strides=1, padding=1, use_bias=False)(net)
    net = keras.layers.BatchNormalization(axis=-1)(net)
    net = keras.layers.Activation('relu')(net)

    net = keras.layers.Conv2D(filters=2*k, kernel_size=(3, 3), strides=1, padding=1, use_bias=False)(net)
    net = keras.layers.BatchNormalization(axis=-1)(net)
    net = keras.layers.Activation('relu')(net)

    # CBAM
    net = SpatialAttention_Conv_2D()(net)
    for i in range(4):
        # 第二层卷积
        net = keras.layers.Conv2D(filters=2 * k, kernel_size=(1, 3), strides=1, padding=1, use_bias=False)(net)
        net = keras.layers.BatchNormalization(axis=-1)(net)
        net = keras.layers.Activation('relu')(net)

        net = keras.layers.Conv2D(filters=2 * k, kernel_size=(3, 3), strides=2, padding=1, use_bias=False)(net)
        net = keras.layers.BatchNormalization(axis=-1)(net)
        net = keras.layers.Activation('relu')(net)

        net = keras.layers.Conv2D(filters=2 * k, kernel_size=(3, 1), strides=1, padding=1, use_bias=False)(net)
        net = keras.layers.BatchNormalization(axis=-1)(net)
        net = keras.layers.Activation('relu')(net)

        net = keras.layers.Conv2D(filters=2 * k, kernel_size=(3, 3), strides=1, padding=1, use_bias=False)(net)
        net = keras.layers.BatchNormalization(axis=-1)(net)
        net = keras.layers.Activation('relu')(net)

        net = keras.layers.Conv2D(filters=2 * k, kernel_size=(3, 3), strides=2, padding=1, use_bias=False)(net)
        net = keras.layers.BatchNormalization(axis=-1)(net)
        net = keras.layers.Activation('relu')(net)

    outputs = keras.layers.GlobalAveragePooling2D()(net)

    return outputs


def VMD_T_Backbone(inputs, k: Optional[int] = None):
    if k is None:
        k = keras.backend.int_shape(inputs)[-1]  # The number of channels 如果未手动传入输入通道数量，则直接获取通道数

    # 先7x1再3x3，再1x7再3x3 先基础卷积
    net = keras.layers.Conv2D(filters=2*k, kernel_size=(1,5), strides=1, padding=1, use_bias=False)(inputs)
    net = keras.layers.BatchNormalization(axis=-1)(net)
    net = keras.layers.Activation('relu')(net)

    net = keras.layers.Conv2D(filters=2*k, kernel_size=(1,3), strides=2, padding=1, use_bias=False)(net)
    net = keras.layers.BatchNormalization(axis=-1)(net)
    net = keras.layers.Activation('relu')(net)

    net = keras.layers.Conv2D(filters=2*k, kernel_size=(1,5), strides=1, padding=1, use_bias=False)(net)
    net = keras.layers.BatchNormalization(axis=-1)(net)
    net = keras.layers.Activation('relu')(net)

    net = keras.layers.Conv2D(filters=2*k, kernel_size=(1,3), strides=1, padding=1, use_bias=False)(net)
    net = keras.layers.BatchNormalization(axis=-1)(net)
    net = keras.layers.Activation('relu')(net)

    # CBAM
    net = SpatialAttention_Conv_2D(kernel_size=7)(net)
    for i in range(2):
        # 第二层卷积
        net = keras.layers.Conv2D(filters=2 * k, kernel_size=(1, 3), strides=1, padding=1, use_bias=False)(net)
        net = keras.layers.BatchNormalization(axis=-1)(net)
        net = keras.layers.Activation('relu')(net)

        net = keras.layers.Conv2D(filters=2 * k, kernel_size=(1, 3), strides=2, padding=1, use_bias=False)(net)
        net = keras.layers.BatchNormalization(axis=-1)(net)
        net = keras.layers.Activation('relu')(net)

        net = keras.layers.Conv2D(filters=2 * k, kernel_size=(1, 3), strides=1, padding=1, use_bias=False)(net)
        net = keras.layers.BatchNormalization(axis=-1)(net)
        net = keras.layers.Activation('relu')(net)

        net = keras.layers.Conv2D(filters=2 * k, kernel_size=(1, 3), strides=1, padding=1, use_bias=False)(net)
        net = keras.layers.BatchNormalization(axis=-1)(net)
        net = keras.layers.Activation('relu')(net)


    outputs = keras.layers.GlobalAveragePooling2D()(net)

    return outputs




def VMD_Backbone1(input_S,input_T):

    inputs = keras.Input(shape=(length, 2))

    S = VMD_S_Backbone(input_S)
    T = VMD_T_Backbone(input_T)

    fusion = tf.concat([S,T], axis=0)
    embeddings = keras.layers.Dense(128, use_bias=False)(fusion)
    embeddings = keras.layers.BatchNormalization(momentum=0.99)(embeddings)
    embeddings = keras.layers.Dense(10, use_bias=False)(embeddings)
    output = keras.layers.BatchNormalization(momentum=0.99)(embeddings)
    model = keras.Model(inputs=inputs, outputs=output)
    return model


def VMD_Backbone2(input_S,input_T):

    inputs = keras.Input(shape=(length, 2))

    S = VMD_S_Backbone(input_S)
    T = VMD_T_Backbone(input_T)

    fusion = tf.concat([S,T], axis=0)
    embeddings = keras.layers.Dense(128, use_bias=False)(fusion)
    embeddings = keras.layers.BatchNormalization(momentum=0.99)(embeddings)
    embeddings = keras.layers.Dense(10, use_bias=False)(embeddings)
    output = keras.layers.BatchNormalization(momentum=0.99)(embeddings)
    model = keras.Model(inputs=inputs, outputs=output)
    return model