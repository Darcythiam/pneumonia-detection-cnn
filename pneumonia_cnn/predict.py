"""Predict from a saved run without refitting or retuning the threshold."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    selected = report["selected_model"]
    threshold = report["candidates"][selected]["validation"]["threshold"]

    import numpy as np
    import tensorflow as tf
    from . import models  # noqa: F401 -- register ResNetPreprocessing for load_model.

    model = tf.keras.models.load_model(run_dir / report["candidates"][selected]["model_file"])
    image = tf.keras.utils.load_img(args.image, target_size=(report["config"]["image_size"],) * 2,
                                    color_mode="rgb")
    pixels = np.expand_dims(tf.keras.utils.img_to_array(image), axis=0)
    probability = float(model.predict(pixels, verbose=0).reshape(-1)[0])
    print(json.dumps({"class": "PNEUMONIA" if probability >= threshold else "NORMAL",
                      "pneumonia_probability": probability, "threshold": threshold,
                      "research_only": True}, indent=2))


if __name__ == "__main__":
    main()
