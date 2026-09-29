# Pneumonia detection from chest X-rays

An educational comparison of a custom CNN and an ImageNet-pretrained ResNet50 on the Kaggle Chest X-Ray Images (Pneumonia) dataset. `Project_5.ipynb` reads `data/chest_xray/` inside the project root and launches the current experiment from Jupyter. The notebook calls the `pneumonia_cnn` package, which contains the training, evaluation, ONNX export, and API code; keep that folder in the repository. The original course notebook and its old outputs are retained as `Project_5_legacy.ipynb` for historical reference. **This is not a clinical diagnostic tool.**

## Why the workflow changed

The original notebook trained with a 16-image validation directory and selected classification thresholds from the **test labels**. It then reported classification accuracy on that same test set. That thresholded accuracy is optimistic. Its ResNet50 path also scaled images to `[0,1]`, while ImageNet ResNet50 expects its own BGR/channel-centering preprocessing. The notebook's saved results do not support the old README's approximate headline values.

The new workflow:

1. Combines the original `train` and tiny `val` directories, then makes a deterministic, stratified validation split from that pool. It leaves the provided `test` directory untouched.
2. Removes byte-identical images from the combined training/validation pool before stratification and records discarded paths in each seed's `duplicate_report.json`. Identical files with conflicting labels, or copies shared between training/validation and the held-out test directory, still stop the run. These checks do not detect patient overlap or visually similar images.
3. Trains both models on the full training split with class weights calculated **only** from training labels. Data augmentation occurs during training. The model artifact contains its own preprocessing: `[0,1]` scaling for the CNN, ResNet-specific preprocessing for ResNet50.
4. Repeats the predeclared model comparison for seeds 42, 123 and 456 by default. Each model's classification threshold is chosen on its own validation split using Youden's J; both models are evaluated on the untouched test directory at every seed. The model/seed chosen for deployment is the highest validation AUC, never the highest test score.
5. Reports accuracy, ROC AUC, average precision, class recall/specificity, Brier score, and a 10-bin expected calibration error (ECE). ECE depends on the binning convention; Brier and ECE are measured on uncalibrated probabilities and do not imply clinical calibration.
6. Applies predetermined Gaussian noise (σ=12/255), a 3×3 mean blur, and 0.65 contrast to test images, reusing each model's fixed validation threshold. These synthetic shifts are diagnostic checks, not evidence of clinical robustness.
7. Exports the validation-selected Keras model to ONNX, checks probability agreement with ONNX Runtime on validation images, and offers a FastAPI inference endpoint. Grad-CAM overlays use the corresponding saved Keras model for gradients.

No new accuracy, AUC, sensitivity, or reproducibility claim is made until the new pipeline is run against the actual dataset.

## Dataset and setup

The images are **not** included in this repository. The [Kaggle Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) can be downloaded with KaggleHub, which returns the local path and caches the files outside this repo. The pipeline resolves the returned root or its `chest_xray` child. The expected directories are `train/NORMAL`, `train/PNEUMONIA`, `test/NORMAL`, and `test/PNEUMONIA`; the original `val` directory is optional. Respect the dataset terms and provide Kaggle credentials if your Kaggle environment requires them.

Use Python 3.11 or 3.12. TensorFlow installation and pretrained ResNet weights require a compatible environment and a network connection for the initial weights download.

For the notebook workflow, extract the dataset so `data/chest_xray/train/NORMAL`, `data/chest_xray/train/PNEUMONIA`, `data/chest_xray/test/NORMAL`, and `data/chest_xray/test/PNEUMONIA` exist. Install the requirements, run `jupyter lab` from this repository's main directory, and open `Project_5.ipynb`. The notebook prints the full local path and stops with a clear error if Jupyter starts in another directory or the data folders are missing. It does not call KaggleHub. Keep `pneumonia_cnn/` alongside it.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pneumonia_cnn.train --download-kaggle --models baseline resnet50 --seeds 42 123 456
```

Training ResNet50 may take considerable time on CPU. To validate the baseline workflow first:

```bash
python -m pneumonia_cnn.train --download-kaggle --models baseline --seeds 42 --epochs 3
```

You can also download the dataset separately and pass the returned path:

```bash
python - <<'PY'
import kagglehub
print(kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia"))
PY
python -m pneumonia_cnn.train --data-root /path/printed/above
```

Each run creates `runs/<UTC timestamp>/report.json` and `seed_<n>/` subdirectories containing the per-seed split manifest, duplicate report, Keras models, histories and metrics. The selected export is `selected.onnx`. An explicitly supplied `--output-dir` must be empty. Use `--seeds`, `--validation-fraction`, `--epochs`, `--batch-size`, and `--image-size` to configure training. `--skip-onnx` skips export for quick training checks; the API requires ONNX. The duplicate check reads every image once per seed; `--skip-duplicate-check` disables deduplication and leakage checks and must not be used for reported results.

To classify a single image with the saved threshold and preprocessing:

```bash
python -m pneumonia_cnn.predict --run-dir runs/<UTC timestamp> --image path/to/image.jpeg
```

This prints a research-only classification and probability. The output does not represent a calibrated clinical risk or a diagnosis.

For an HTTP prediction with an optional Grad-CAM overlay:

```bash
PNEUMONIA_RUN_DIR=runs/<UTC timestamp> uvicorn pneumonia_cnn.api:app --host 127.0.0.1 --port 8000
curl -F 'image=@path/to/image.jpeg' 'http://127.0.0.1:8000/predict?grad_cam=true'
```

`GET /health` checks that the ONNX model loads. `POST /predict` accepts JPEG/PNG (up to 10 MiB), returns a thresholded class, pneumonia probability, and, when `grad_cam=true`, a base64 PNG overlay in `grad_cam_overlay_png_base64`. Set `grad_cam=false` for ONNX-only inference without loading TensorFlow. The heatmap highlights regions influencing the model score; it does not establish pathology location or provide clinical justification. Both artifacts must remain together in the run directory. The server has no authentication and should be kept on localhost for research.

## Historical notebook results

The saved output in `Project_5_legacy.ipynb` shows the custom CNN at **0.8061 accuracy at Keras's default 0.5 threshold** and **0.9115 ROC AUC**. The ResNet50 output shows **0.3750 default-threshold accuracy** and **0.8297 ROC AUC**. The legacy notebook also prints roughly **0.85** and **0.78** classification accuracy after selecting thresholds on the test set itself; those two figures are not valid held-out thresholded accuracy estimates. The tiny validation set and mismatched ResNet preprocessing further limit this comparison. The notebook includes exploratory reruns, but the repo does not establish stability across seeds under the new protocol.

## Verification

The split and metric logic can be tested without downloading the dataset or installing TensorFlow if NumPy and scikit-learn are installed:

```bash
python -m unittest discover -s tests -v
```

The exact accuracy, AUC, calibration, and claim of consistent superiority in a resume must come from a real multi-seed run's `aggregate_test` and individual `runs` records, not this repo's historical notebook or synthetic smoke tests. One dataset and one frozen ResNet50 configuration cannot establish that ImageNet features generally transfer poorly to medical imaging. Identical hashes do not guarantee patient-level independence because the dataset does not provide reliable patient identifiers here. External validation and clinical evaluation would be necessary before any medical use.
