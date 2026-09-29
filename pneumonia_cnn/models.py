"""Raw 0-255 RGB input models; preprocessing is saved inside each model."""

import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="pneumonia_cnn")
class ResNetPreprocessing(layers.Layer):
    def call(self, inputs):
        # ImageNet ResNet50 expects BGR channels and ImageNet channel centering.
        return tf.keras.applications.resnet50.preprocess_input(
            tf.cast(inputs, tf.float32)
        )


def build_model(kind: str, *, image_size: int = 224, seed: int = 42):
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="rgb_0_to_255")
    augmentation = tf.keras.Sequential([
        layers.RandomRotation(0.03, seed=seed),
        layers.RandomZoom(0.05, seed=seed + 1),
    ], name="training_augmentation")
    x = augmentation(inputs)
    if kind == "baseline":
        x = layers.Rescaling(1 / 255.0)(x)
        for filters in (32, 64, 128):
            x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
            x = layers.MaxPooling2D()(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation="relu")(x)
    elif kind == "resnet50":
        x = ResNetPreprocessing()(x)
        base = tf.keras.applications.ResNet50(
            weights="imagenet", include_top=False,
            input_shape=(image_size, image_size, 3),
        )
        base.trainable = False
        x = base(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation="relu")(x)
    else:
        raise ValueError(f"Unknown model: {kind}")
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="pneumonia_probability")(x)
    model = tf.keras.Model(inputs, outputs, name=f"pneumonia_{kind}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                 tf.keras.metrics.AUC(name="auc")],
    )
    return model
