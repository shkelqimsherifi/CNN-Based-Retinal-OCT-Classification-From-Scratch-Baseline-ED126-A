import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import tensorflow as tf
from keras import layers


@keras.saving.register_keras_serializable(package="Custom")
class ChannelAttention(layers.Layer):
    def __init__(self, reduction=16, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction
        self.mlp = None

    def build(self, input_shape):
        channels = int(input_shape[-1])
        self.mlp = keras.Sequential(
            [
                layers.Dense(channels // self.reduction, activation="relu"),
                layers.Dense(channels),
            ]
        )
        super().build(input_shape)

    def call(self, x):
        avg = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        avg = self.mlp(avg)

        max_val = tf.reduce_max(x, axis=[1, 2], keepdims=True)
        max_val = self.mlp(max_val)

        scale = tf.nn.sigmoid(avg + max_val)
        return x * scale

    def get_config(self):
        config = super().get_config()
        config.update({"reduction": self.reduction})
        return config


@keras.saving.register_keras_serializable(package="Custom")
class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = None

    def build(self, input_shape):
        self.conv = layers.Conv2D(1, self.kernel_size, padding="same", activation="sigmoid")
        super().build(input_shape)

    def call(self, x):
        avg = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_val = tf.reduce_max(x, axis=-1, keepdims=True)
        scale = self.conv(tf.concat([avg, max_val], axis=-1))
        return x * scale

    def get_config(self):
        config = super().get_config()
        config.update({"kernel_size": self.kernel_size})
        return config


@keras.saving.register_keras_serializable(package="Custom")
class CBAM(layers.Layer):
    def __init__(self, reduction=16, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction
        self.kernel_size = kernel_size
        self.channel_att = ChannelAttention(reduction=reduction)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def build(self, input_shape):
        self.channel_att.build(input_shape)
        self.spatial_att.build(input_shape)
        super().build(input_shape)

    def call(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"reduction": self.reduction, "kernel_size": self.kernel_size})
        return config


CUSTOM_OBJECTS = {
    "CBAM": CBAM,
    "ChannelAttention": ChannelAttention,
    "SpatialAttention": SpatialAttention,
}
