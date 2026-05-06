# Pneumonia Detection from Chest X-Rays using CNNs and Transfer Learning

Deep learning pipeline for automated pneumonia classification from chest X-ray images using a custom Convolutional Neural Network (CNN) and transfer learning with ResNet50.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue" />
  <img src="https://img.shields.io/badge/TensorFlow-Keras-orange" />
  <img src="https://img.shields.io/badge/Deep%20Learning-CNN-red" />
  <img src="https://img.shields.io/badge/Medical-Imaging-green" />
</p>

---

## Project Overview

Pneumonia is a serious respiratory infection that can often be identified through chest X-ray imaging. Manual interpretation of X-rays may be time-consuming and subject to variability, making automated image classification an important application of deep learning in medical vision.

This project implements and evaluates two approaches for binary chest X-ray classification:

- **Baseline CNN** trained from scratch
- **Transfer Learning model** using pretrained ResNet50

The project emphasizes:
- CNN architecture design
- Medical image preprocessing
- Transfer learning
- Model evaluation using ROC/AUC
- Handling class imbalance
- Reproducibility analysis

---

## Models Implemented

### 1️⃣ Baseline CNN
Custom CNN architecture built from scratch using TensorFlow/Keras.

#### Architecture Highlights
- Multiple Conv2D + MaxPooling blocks
- Increasing feature depth:
  - 32 filters
  - 64 filters
  - 128 filters
- Dense classification layers
- Dropout regularization to reduce overfitting

#### Purpose
The baseline model serves as a reference point to evaluate how effectively the network can learn directly from the chest X-ray dataset without pretrained weights.

---

### 2️⃣ Transfer Learning (ResNet50)

Transfer learning model built using:
- **ResNet50 pretrained on ImageNet**
- Frozen convolutional base
- Custom classification head:
  - GlobalAveragePooling2D
  - Dense layer (ReLU)
  - Dropout
  - Sigmoid output layer

#### Purpose
Leverages pretrained visual feature representations to improve convergence and classification performance on limited medical imaging data.

---

## Dataset

Dataset:
- **Chest X-ray Images (Pneumonia)** from Kaggle

### Classes
- NORMAL
- PNEUMONIA

### Dataset Structure

```text
data/
└── chest_xray/
    ├── train/
    ├── val/
    └── test/
```

---

## Data Preprocessing

The preprocessing pipeline includes:

### Image Processing
- Resize images to **224×224**
- Normalize pixel values to `[0,1]`

### Data Augmentation
Applied only to training images:
- Random rotations
- Horizontal flips
- Zoom transformations

### Why Augmentation?
Medical imaging datasets are relatively limited in size. Augmentation improves generalization and reduces overfitting by exposing the model to varied image transformations.

---

## Evaluation Metrics

Models were evaluated using multiple metrics:

- Accuracy
- ROC Curve
- AUC Score
- Confusion Matrix
- Classification Report

### Handling Class Imbalance
The dataset contains significantly more pneumonia images than normal images. To address this imbalance:

- Class weighting was applied during training
- Threshold optimization was performed using ROC analysis

This significantly improved prediction quality for the minority class.

---

## Final Results

| Model | Accuracy | AUC |
|------|------|------|
| Baseline CNN | **~0.87** | **~0.93** |
| Transfer Learning (ResNet50) | ~0.79 | ~0.84 |

### Key Observation
Although transfer learning is generally expected to outperform models trained from scratch, the baseline CNN achieved stronger empirical performance in this experiment.

Possible reasons include:
- Limited training epochs
- Frozen pretrained layers
- Domain mismatch between ImageNet and medical X-ray images

---

## Reproducibility Analysis

To evaluate reproducibility and model stability:
- Each model was trained twice using different random seeds
- Accuracy and AUC values remained consistent across runs

This indicates:
- Stable optimization behavior
- Low sensitivity to initialization randomness

---

## Sample Pipeline

```text
Chest X-ray Image
        ↓
Preprocessing & Augmentation
        ↓
CNN / ResNet50 Model
        ↓
Binary Classification
(NORMAL vs PNEUMONIA)
```

---

## Technologies Used

### Languages & Frameworks
- Python
- TensorFlow
- Keras

### Libraries
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## Repository Structure

```text
.
├── Project_5.ipynb
├── requirements.txt
├── README.md
└── data/
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/pneumonia-detection-cnn.git
cd pneumonia-detection-cnn
```

Create virtual environment:

```bash
python3 -m venv env
source env/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

---

## Key Learning Outcomes

This project demonstrates:
- CNN architecture design principles
- Transfer learning workflows
- Medical image preprocessing
- Handling imbalanced datasets
- ROC/AUC evaluation
- Model reproducibility analysis

---

## Disclaimer

This project was developed for educational and research purposes only. It is **not** a clinically validated diagnostic system and should not be used for medical decision-making.
