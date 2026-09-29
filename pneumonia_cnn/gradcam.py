"""Grad-CAM of the positive (pneumonia) score from the original Keras artifact."""

import numpy as np
import tensorflow as tf


def heatmap(model, pixels: np.ndarray) -> np.ndarray:
    if pixels.shape[0] != 1:
        raise ValueError("Grad-CAM expects one RGB image")
    if model.name == "pneumonia_baseline":
        conv = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)][-1]
        probe = tf.keras.Model(model.inputs, [conv.output, model.output])
        with tf.GradientTape() as tape:
            features, score = probe(tf.convert_to_tensor(pixels), training=False)
            target = score[:, 0]
    elif model.name == "pneumonia_resnet50":
        base = next(layer for layer in model.layers if isinstance(layer, tf.keras.Model)
                    and layer.name.startswith("resnet50"))
        extractor = tf.keras.Model(base.inputs, [base.get_layer("conv5_block3_out").output, base.output])
        x = tf.convert_to_tensor(pixels)
        base_index = model.layers.index(base)
        with tf.GradientTape() as tape:
            for layer in model.layers[1:base_index]:
                x = layer(x, training=False)
            features, x = extractor(x, training=False)
            for layer in model.layers[base_index + 1:]:
                x = layer(x, training=False)
            target = x[:, 0]
    else:
        raise ValueError(f"No Grad-CAM implementation for {model.name}")
    gradients = tape.gradient(target, features)
    if gradients is None:
        raise RuntimeError("Cannot calculate gradients for the convolutional features")
    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    activation = tf.nn.relu(tf.reduce_sum(weights * features, axis=-1))[0]
    maximum = tf.reduce_max(activation)
    return np.asarray(activation / (maximum + 1e-8), dtype=np.float32)
