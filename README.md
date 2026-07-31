# Lightweight Deep Learning for Chest X-Ray Classification

This project compares lightweight deep learning architectures for multi-label classification of chest X-rays from the **ChestMNIST** dataset.

The goal is to evaluate how CNNs, attention mechanisms, Vision Transformers, and graph neural networks perform under the same resource-constrained setup. All models are trained from scratch on `64 × 64` grayscale images and predict 14 thoracic findings. :contentReference[oaicite:0]{index=0}

## Models

The project includes:

- Baseline CNN
- CNN with CBAM attention
- ViT-lite
- IEViT-lite hybrid CNN–Transformer
- GCN with dot-product fusion
- GCN with interaction fusion

The graph-based models use label co-occurrence relationships constructed with either conditional-probability thresholding or PMI.

## Pipeline

The main workflow includes:

1. ChestMNIST data loading and exploration
2. CLAHE contrast enhancement
3. Image normalization
4. Class-imbalance correction with weighted binary cross-entropy
5. Gaussian-noise augmentation
6. Model training and evaluation
7. Per-class threshold tuning
8. Ablation, robustness, and Grad-CAM analysis

## Main Outcome

The best overall macro AUC was achieved by the GCN model with interaction fusion and PMI adjacency, while the CNN with CBAM provided a very similar result with lower computational cost.

The full methodology, architecture descriptions, experiments, and results are available in the accompanying report.

## Repository Contents

```text
.
├── cxr-code.ipynb
├── cxr-report.pdf
└── README.md

├── project_code.ipynb
├── Kuzovkova.pdf
└── README.md
