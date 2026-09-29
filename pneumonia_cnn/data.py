"""Decode X-rays to raw RGB pixels; model artifacts own preprocessing."""

import tensorflow as tf


def make_dataset(records, *, image_size: int, batch_size: int,
                 training: bool = False, seed: int = 42,
                 degradation: str | None = None):
    if degradation not in (None, "noise", "blur", "low_contrast"):
        raise ValueError(f"Unknown degradation: {degradation}")
    paths = [str(record.path) for record in records]
    labels = [float(record.label) for record in records]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def decode(path, label):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3,
                                   expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(tf.cast(image, tf.float32), (image_size, image_size))
        return image, label

    ds = ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    if degradation:
        def transform(index, item):
            image, label = item
            if degradation == "noise":
                noise = tf.random.stateless_normal(tf.shape(image), seed=[seed, tf.cast(index, tf.int32)])
                image = tf.clip_by_value(image + noise * 12.0, 0.0, 255.0)
            elif degradation == "blur":
                image = tf.nn.avg_pool2d(image[None], ksize=3, strides=1, padding="SAME")[0]
            else:
                image = tf.clip_by_value((image - 127.5) * 0.65 + 127.5, 0.0, 255.0)
            return image, label
        ds = ds.enumerate().map(transform, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
