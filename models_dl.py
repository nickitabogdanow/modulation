import tensorflow as tf
from tensorflow.keras import layers, models


def conv_block_1d(x, filters, kernel_size=7, stride=1, dropout=0.0):
    x = layers.Conv1D(filters, kernel_size, strides=stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    return x


def build_cnn1d(input_len: int, num_channels: int, num_classes: int) -> tf.keras.Model:
    inp = layers.Input(shape=(input_len, num_channels), name="iq_input")

    x = conv_block_1d(inp, 32, 7, 1, 0.1)
    x = conv_block_1d(x, 32, 7, 2, 0.1)
    x = conv_block_1d(x, 64, 5, 1, 0.1)
    x = conv_block_1d(x, 64, 5, 2, 0.1)
    x = conv_block_1d(x, 128, 3, 1, 0.1)
    x = conv_block_1d(x, 128, 3, 2, 0.1)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inp, out, name="cnn1d_iq")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def conv_block_2d(x, filters, k=(3,3), s=(1,1), p="same", d=0.0):
    x = layers.Conv2D(filters, k, strides=s, padding=p, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    if d > 0:
        x = layers.Dropout(d)(x)
    return x


def build_cnn2d_spectrogram(input_shape, num_classes: int) -> tf.keras.Model:
    inp = layers.Input(shape=input_shape, name="spec_input")
    x = conv_block_2d(inp, 32, (3,3), (1,1), "same", 0.1)
    x = conv_block_2d(x, 32, (3,3), (2,2), "same", 0.1)
    x = conv_block_2d(x, 64, (3,3), (1,1), "same", 0.1)
    x = conv_block_2d(x, 64, (3,3), (2,2), "same", 0.1)
    x = conv_block_2d(x, 128, (3,3), (1,1), "same", 0.1)
    x = conv_block_2d(x, 128, (3,3), (2,2), "same", 0.1)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inp, out, name="cnn2d_spec")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
