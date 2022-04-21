import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# DCGAN-like generator and discriminator
class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()  # 初始化keras.model的属性
        self.fullConv = layers.Dense(3 * 3 * 512)

        self.convolution1 = layers.Conv2DTranspose(256, 3, 3, 'valid')
        #filters：整数,输出空间的维数(即卷积中的滤波器数).
        #kernel_size：一个元组或2个正整数的列表,指定过滤器的空间维度；可以是单个整数,以指定所有空间维度的相同值.
        #strides：一个元组或2个正整数的列表,指定卷积的步幅；可以是单个整数,以指定所有空间维度的相同值.
        self.bn1 = layers.BatchNormalization()
        #该方法出自Sergey Ioffe和Christian Szegedy的《Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift》一文，BN方法到底能否减少Internal Covariate Shift可能还无法下最终结论 (参见《How Does Batch Normalization Help Optimization?》)，
        # 但的确可以加速训练平稳收敛！
        self.convolution2 = layers.Conv2DTranspose(128, 5, 2, 'valid')
        self.bn2 = layers.BatchNormalization()

        self.convolution3 = layers.Conv2DTranspose(3, 4, 3, 'valid')

    def call(self, inputs, training=None):
        x = self.fullConv(inputs)
        x = tf.reshape(x, [-1, 3, 3, 512])
        x = tf.nn.leaky_relu(x)
        x = tf.nn.leaky_relu(self.bn1(self.convolution1(x), training=training))
        x = tf.nn.leaky_relu(self.bn2(self.convolution2(x), training=training))
        x = self.convolution3(x)
        x = tf.tanh(x)
        return x


class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.convolution1 = layers.Conv2D(64, 5, 3, 'valid')

        self.convolution2 = layers.Conv2D(128, 5, 3, 'valid')
        self.bn2 = layers.BatchNormalization()  # 标准化

        self.convolution3 = layers.Conv2D(256, 5, 3, 'valid')
        self.bn3 = layers.BatchNormalization()

        self.flatten = layers.Flatten()  # 压平，为了dense层
        self.fullConv = layers.Dense(1)

    def call(self, inputs, training=None):
        x = tf.nn.leaky_relu(self.convolution1(inputs))
        x = tf.nn.leaky_relu(self.bn2(self.convolution2(x), training=training))
        x = tf.nn.leaky_relu(self.bn3(self.convolution3(x), training=training))

        x = self.flatten(x)
        logits = self.fullConv(x)
        return logits
