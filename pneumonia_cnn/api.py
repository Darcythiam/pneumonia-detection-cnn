"""Research-only ONNX inference service with optional Keras Grad-CAM overlay."""

import base64
from functools import lru_cache
from io import BytesIO
import json
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
import numpy as np
from PIL import Image, UnidentifiedImageError


MAX_IMAGE_BYTES = 10 * 1024 * 1024


def create_app(run_dir: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="Pneumonia CNN research inference", version="1.0")
    root = Path(run_dir or os.getenv("PNEUMONIA_RUN_DIR", "runs/latest")).resolve()

    @lru_cache(maxsize=1)
    def load_runtime():
        import onnxruntime as ort
        report = json.loads((root / "report.json").read_text(encoding="utf-8"))
        if "onnx" not in report:
            raise ValueError("This run has no ONNX export; train without --skip-onnx")
        onnx_path = root / report["onnx"]["model_file"]
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        return report, session

    @lru_cache(maxsize=1)
    def load_keras():
        import tensorflow as tf
        from . import models  # noqa: F401 -- register ResNet preprocessing.
        report, _ = load_runtime()
        path = root / report["candidates"][report["selected_model"]]["model_file"]
        return tf.keras.models.load_model(path, compile=False)

    @app.get("/health")
    def health():
        try:
            load_runtime()
        except (OSError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}") from exc
        return {"status": "ready", "research_only": True}

    @app.post("/predict")
    async def predict(image: UploadFile = File(...), grad_cam: bool = True):
        if image.content_type not in ("image/jpeg", "image/png"):
            raise HTTPException(status_code=415, detail="Upload a JPEG or PNG image")
        payload = await image.read(MAX_IMAGE_BYTES + 1)
        if len(payload) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Image exceeds 10 MiB")
        try:
            with Image.open(BytesIO(payload)) as source:
                source.verify()
            with Image.open(BytesIO(payload)) as source:
                rgb = source.convert("RGB")
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Invalid image") from exc
        try:
            report, session = load_runtime()
        except (OSError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}") from exc
        size = report["config"]["image_size"]
        pixels = np.asarray(rgb.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)[None]
        probability = float(np.asarray(session.run(None, {
            session.get_inputs()[0].name: pixels
        })[0]).reshape(-1)[0])
        threshold = report["candidates"][report["selected_model"]]["validation"]["threshold"]
        result = {"class": "PNEUMONIA" if probability >= threshold else "NORMAL",
                  "pneumonia_probability": probability, "threshold": threshold,
                  "research_only": True}
        if grad_cam:
            from .gradcam import heatmap
            model = load_keras()
            cam = heatmap(model, pixels)
            cam_image = Image.fromarray(np.uint8(cam * 255)).resize((size, size), Image.Resampling.BILINEAR)
            intensity = np.asarray(cam_image, dtype=np.float32) / 255
            overlay = np.asarray(rgb.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)
            red = np.zeros_like(overlay)
            red[..., 0] = 255
            overlay = Image.fromarray(np.uint8(np.clip(
                overlay * (1 - 0.5 * intensity[..., None]) + red * (0.5 * intensity[..., None]),
                0, 255)))
            buffer = BytesIO()
            overlay.save(buffer, format="PNG")
            result["grad_cam_overlay_png_base64"] = base64.b64encode(buffer.getvalue()).decode("ascii")
        return result

    return app


app = create_app()
