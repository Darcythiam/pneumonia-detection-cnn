"""Export the selected Keras model and check ONNX Runtime parity on validation pixels."""

from pathlib import Path
import numpy as np


def export_onnx(model_file: Path, output_file: Path, validation_pixels: np.ndarray) -> dict:
    import tensorflow as tf
    import onnxruntime as ort
    from . import models  # noqa: F401 -- register custom preprocessing.

    model = tf.keras.models.load_model(model_file, compile=False)
    # Random augmentation is an identity at inference, but its traced random
    # branches contain ops tf2onnx cannot convert. Build the inference graph
    # from the same trained layers, skipping augmentation explicitly.
    pixels = tf.keras.Input(shape=model.input_shape[1:], dtype=tf.float32,
                            name="rgb_0_to_255")
    x = pixels
    for layer in model.layers[2:]:
        x = layer(x, training=False)
    inference_model = tf.keras.Model(pixels, x)
    inputs = validation_pixels.astype(np.float32)
    expected = np.asarray(model(inputs, training=False)).reshape(-1)
    stripped = np.asarray(inference_model(inputs, training=False)).reshape(-1)
    if not np.allclose(expected, stripped, atol=1e-5, rtol=1e-5):
        raise RuntimeError("Removing inference-time augmentation changed Keras predictions")
    inference_model.export(output_file, format="onnx", verbose=False)
    session = ort.InferenceSession(str(output_file), providers=["CPUExecutionProvider"])
    actual = np.asarray(session.run(None, {session.get_inputs()[0].name: inputs})[0]).reshape(-1)
    delta = float(np.max(np.abs(expected - actual)))
    if not np.allclose(expected, actual, atol=1e-4, rtol=1e-4):
        output_file.unlink(missing_ok=True)
        raise RuntimeError(f"ONNX and Keras predictions disagree (max delta={delta:.6g})")
    return {"model_file": output_file.name, "input": "raw RGB float32 pixels [0,255]",
            "parity_samples": len(inputs), "max_abs_difference": delta}
