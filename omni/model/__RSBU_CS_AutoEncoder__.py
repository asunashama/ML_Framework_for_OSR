import keras.layers
from omni import *
import tensorflow as tf
import keras
from keras.layers import Layer, Conv1D, BatchNormalization, Activation, Add, GlobalAveragePooling1D, Dense ,Conv2DTranspose,ReLU,Reshape,Conv2D,Lambda,Permute



# 阈值软化层
class SoftThresholding(Layer):
    def __init__(self, **kwargs):
        super(SoftThresholding, self).__init__(**kwargs)
    def call(self, inputs, threshold):
        threshold = tf.expand_dims(threshold, axis=1)  # 调整维度匹配 [batch, 1, 1]
        positive = tf.nn.relu(tf.abs(inputs) - threshold)
        return tf.sign(inputs) * positive

# 通道阈值共享层
class ChannelThreshold(Layer):
    def __init__(self, ratio=8, **kwargs):
        super(ChannelThreshold, self).__init__(**kwargs)
        self.ratio = ratio  # 全连接层缩减比例
        self.gap = GlobalAveragePooling1D()
        self.fc1 = Dense(units=max(1, 64 // self.ratio), activation='relu')  # 动态调整单元数
        self.fc2 = Dense(units=1, activation='sigmoid')
    def call(self, inputs):
        x_abs = tf.abs(inputs)  # 先算absolute，再gap
        gap = self.gap(x_abs)  # [batch, channels]
        # 生成缩放系数（0~1）
        scale = self.fc2(self.fc1(gap))  # [batch, 1]
        # 计算阈值：scale * mean_abs
        threshold = scale * tf.reduce_mean(gap, axis=-1, keepdims=True)  # [batch, 1]
        return threshold


# RSBU_CS块
def RSBU_CS(inputs, filters, kernel_size=3, strides=1):
    """1D残差收缩块（通道共享阈值）"""
    # 主路径
    x = Conv1D(filters, kernel_size, strides=strides, padding='same')(inputs)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)

    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization(axis=-1)(x)

    # 阈值生成子网络
    threshold_layer = ChannelThreshold()
    threshold = threshold_layer(x)  # [batch, 1]

    # 应用软阈值化
    st = SoftThresholding()
    x = st(x, threshold)

    # 跳跃连接（如果需要维度匹配）
    if strides != 1 or inputs.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, strides=strides, padding='same')(inputs)
        shortcut = BatchNormalization(axis=-1)(shortcut)
    else:
        shortcut = inputs

    # 残差连接
    outputs = Add()([x, shortcut])
    outputs = Activation('relu')(outputs)
    return outputs


def encoder(inputs,num_classes=10):
    x = Conv1D(64, 3, padding='same')(inputs)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)


    x = RSBU_CS(x, 64)  # 输出形状：[batch, timesteps, 64]
    x = RSBU_CS(x, 128, strides=2)  # 时间步长减半
    x = RSBU_CS(x, 256, strides=2)  # 时间步长再次减半
    x = RSBU_CS(x, 512, strides=2)  # 时间步长再次减半
    x = RSBU_CS(x, 1024, strides=2)  # 时间步长再次减半

    x = GlobalAveragePooling1D()(x)  # [batch, 256]
    embeddings = Dense(num_classes)(x)

    outputs_softmax = Dense(num_classes, activation='softmax')(x)
    return embeddings, outputs_softmax






def decoder(inputs):
    # 输入形状 (1024, 1)
    x = Dense(1024 * 4)(inputs)  # 扩展特征维度
    x = Reshape((16, 64, 4))(x)  # 调整为 channels_first 格式 (C, H, W)

    # 转置卷积层（逐步扩展高度到 1024，保持宽度为 4）
    x = Conv2DTranspose(64, kernel_size=(5, 3), strides=(2, 1), padding='same', data_format='channels_first')(x)
    x = BatchNormalization(axis=1)(x)
    x = ReLU()(x)

    x = Conv2DTranspose(64, kernel_size=(5, 3), strides=(2, 1), padding='same', data_format='channels_first')(x)
    x = BatchNormalization(axis=1)(x)
    x = ReLU()(x)

    x = Conv2DTranspose(64, kernel_size=(5, 3), strides=(2, 1), padding='same', data_format='channels_first')(x)
    x = BatchNormalization(axis=1)(x)
    x = ReLU()(x)

    x = Conv2DTranspose(64, kernel_size=(5, 3), strides=(2, 1), padding='same', data_format='channels_first')(x)
    x = BatchNormalization(axis=1)(x)
    x = ReLU()(x)

    # 最后一层 Conv2D：输出形状 (2, 1024, 4)
    x = Conv2D(2, kernel_size=(1, 1), strides=1, padding='same', activation='sigmoid',
                      data_format='channels_first')(x)

    # 调整宽度维度：从 4 压缩为 2
    x = Conv2D(2, kernel_size=(1, 4), strides=1, padding='valid', activation='sigmoid',
                      data_format='channels_first')(x)

    # 移除宽度维度（1）并调整通道顺序
    x = Lambda(lambda x: tf.squeeze(x, axis=-1))(x)  # 输出形状 (Batch, 2, 1024)
    x = Permute((2, 1))(x)  # 调整为 (Batch, 1024, 2)

    return x



def AE(length=1024,class_num=20):
    inputs = tf.keras.Input(shape=(length,2))  # 输入形状：[batch, timesteps, 2]
    embeddings, outputs = encoder(inputs,num_classes=class_num)
    restruct_signal = decoder(embeddings)
    model = keras.Model(inputs=inputs, outputs=(embeddings, outputs,restruct_signal))
    return model


if __name__ == '__main__':
    inputs = tf.ones((32, 1024, 2))
    model = AE()
    model.summary()
    output = model(inputs)


