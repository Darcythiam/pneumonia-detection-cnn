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

The images are not tracked in this repository. Download the [Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and extract its `chest_xray` folder into `data/`:

```text
data/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── val/                 # Optional
```

Both the notebook and the terminal training command use `data/chest_xray/`. The original `train` and optional `val` images form the pool from which the pipeline creates a stratified validation split. The original `test` directory remains held out. The dataset comes from the work of [Kermany, Zhang, and Goldbaum](https://data.mendeley.com/datasets/rscbjbr9sj/2).

Use Python 3.11 or 3.12. From the project root, create an environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run from the notebook

Start Jupyter Lab from the project root and open `Project_5.ipynb`:

```bash
jupyter lab
```

The notebook checks for `data/chest_xray/` before launching training. It uses the code in `pneumonia_cnn/` and does not download the dataset.

### Run from the terminal

To compare the custom CNN and frozen ResNet50 across three seeds:

```bash
python -m pneumonia_cnn.train \
  --data-root data/chest_xray \
  --models baseline resnet50 \
  --seeds 42 123 456
```

ResNet50 training can take considerable time on CPU. To check the baseline training workflow with a shorter run:

```bash
python -m pneumonia_cnn.train \
  --data-root data/chest_xray \
  --models baseline \
  --seeds 42 \
  --epochs 3 \
  --skip-onnx
```

The first ResNet50 run may download pretrained weights if they are not already cached.

Each run creates a timestamped directory under `runs/`. Its `report.json` contains the selected model, per-seed results, and aggregate test metrics. The `seed_<n>/` directories contain split manifests, duplicate reports, model files, training histories, and metrics. A run with ONNX export also contains `selected.onnx`.

You can adjust training with `--seeds`, `--validation-fraction`, `--epochs`, `--batch-size`, and `--image-size`. If you supply `--output-dir`, that directory must be empty. Do not use `--skip-duplicate-check` for reported results: it disables the identical-image deduplication and cross-split leakage checks.

### Predict with a trained model

Set `RUN_DIR` to the directory printed when training finishes:

```bash
RUN_DIR="runs/REPLACE_WITH_RUN_TIMESTAMP"

python -m pneumonia_cnn.predict \
  --run-dir "$RUN_DIR" \
  --image /path/to/image.jpeg
```

The command uses the saved model and its validation-selected threshold. Its probability is a model output, not a calibrated clinical risk or diagnosis.

### Serve predictions over HTTP

Use a run that contains `selected.onnx`. Start the API from the project root:

```bash
RUN_DIR="runs/REPLACE_WITH_RUN_TIMESTAMP"
PNEUMONIA_RUN_DIR="$RUN_DIR" uvicorn pneumonia_cnn.api:app \
  --host 127.0.0.1 \
  --port 8000
```

In another terminal, check the model and submit an image:

```bash
curl http://127.0.0.1:8000/health

curl -F 'image=@/path/to/image.jpeg' \
  'http://127.0.0.1:8000/predict?grad_cam=true'
```

`GET /health` checks that the ONNX model loads. `POST /predict` accepts JPEG or PNG images up to 10 MiB and returns the predicted class, pneumonia probability, and saved threshold. With `grad_cam=true`, it also returns a base64 PNG overlay in `grad_cam_overlay_png_base64`. Use `grad_cam=false` for ONNX inference without loading the Keras model.

Grad-CAM highlights regions that influenced the model score; it does not establish the location of a disease or justify a clinical decision. The API has no authentication and is intended for local research use.

## Historical notebook results

The saved output in `Project_5_legacy.ipynb` shows the custom CNN at **0.8061 accuracy at Keras's default 0.5 threshold** and **0.9115 ROC AUC**. The ResNet50 output shows **0.3750 default-threshold accuracy** and **0.8297 ROC AUC**. The legacy notebook also prints roughly **0.85** and **0.78** classification accuracy after selecting thresholds on the test set itself; those two figures are not valid held-out thresholded accuracy estimates. The tiny validation set and mismatched ResNet preprocessing further limit this comparison. The notebook includes exploratory reruns, but the repo does not establish stability across seeds under the new protocol.

## Verification

The split and metric logic can be tested without downloading the dataset or installing TensorFlow if NumPy and scikit-learn are installed:

```bash
python -m unittest discover -s tests -v
```

The exact accuracy, AUC, calibration, and claim of consistent superiority in a resume must come from a real multi-seed run's `aggregate_test` and individual `runs` records, not this repo's historical notebook or synthetic smoke tests. One dataset and one frozen ResNet50 configuration cannot establish that ImageNet features generally transfer poorly to medical imaging. Identical hashes do not guarantee patient-level independence because the dataset does not provide reliable patient identifiers here. External validation and clinical evaluation would be necessary before any medical use.
