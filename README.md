# AgriChainNet





![Python](https://img.shields.io/badge/Python-3.x-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.11-ee4c2c) ![CUDA](https://img.shields.io/badge/CUDA-12.8-76b900) ![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

Most deep learning work on plant disease treats **detection, segmentation and classification as separate tasks**. Few systems are tested on real field images, and none keeps a tamper-proof record of each diagnosis. **AgriChainNet** addresses all three problems in one end-to-end framework:

1. **Detection:** YOLOv8m locates diseased regions on the leaf.
2. **Segmentation:** U-Net delineates lesion boundaries at the pixel level and estimates the diseased-area percentage.
3. **Classification:** **AgriConvNet**, our proposed model, identifies the disease. It is a ConvNeXt-Small backbone with **four cascaded CBAM attention blocks**, one after each backbone stage.
4. **Blockchain:** every diagnosis is hashed (SHA-256) into a chained ledger. A smart contract then automatically triggers an insurance payout, a government alert, or a monitoring advisory.

```
Raw leaf image + farm metadata
        │
        ▼
┌───────────────┐   boxes   ┌───────────────┐  mask + lesion %  ┌──────────────────────┐  label + conf  ┌─────────────────────┐
│ YOLOv8m       │ ────────► │ U-Net         │ ────────────────► │ AgriConvNet          │ ─────────────► │ Blockchain ledger   │
│ Detection     │           │ Segmentation  │                   │ ConvNeXt-S + 4×CBAM  │                │ + Smart contract    │
└───────────────┘           └───────────────┘                   └──────────────────────┘                └─────────────────────┘
```

### Main contributions
- A **single multi-task pipeline** (detect → segment → classify) that follows the diagnostic sequence of an expert agronomist.
- **AgriConvNet**: multi-stage CBAM integration at channel widths 96 / 192 / 384 / 768, with a dual-pooling (GAP ⊕ GMP, 1536-d) classifier head.
- A **blockchain smart-contract layer** for immutable diagnostic records and automated institutional actions.
- **Multi-dataset evaluation** on four public datasets, including real-field images from Chengalpattu, Kanchipuram and Krishnagiri districts in Tamil Nadu, India.

---

## Results

### Classification (Table 6 and 7)
| Model | Accuracy | F1 | Precision | Recall | Params | FLOPs |
|---|---|---|---|---|---|---|
| VGG16 | 91.93% | 0.9166 | 0.9150 | 0.9193 | 138M | 15.5G |
| MobileNetV2 | 90.62% | 0.9050 | 0.9038 | 0.9062 | 3.4M | 0.3G |
| EfficientNetB0 | 92.91% | 0.9274 | 0.9261 | 0.9291 | 5.3M | 0.4G |
| ResNet50 | 98.75% | 0.9876 | 0.9879 | 0.9875 | 25M | 4.1G |
| **AgriConvNet (Ours)** | **98.90%** | **0.9891** | **0.9893** | **0.9890** | 51M | 8.9G |

5-fold stratified cross-validation: **98.77% ± 0.19%** validation accuracy, F1 0.9877 ± 0.0019.

### Detection and segmentation (Table 8)
| Module | Model | Dataset | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Dice | IoU | Pixel Acc |
|---|---|---|---|---|---|---|---|---|---|
| Detection | YOLOv8m | Multi-Crop TN | 70.99% | 49.59% | 73.70% | 68.12% | — | — | — |
| Segmentation | U-Net | Leaf Disease | — | — | — | — | 0.9948 | 0.9897 | 99.68% |

### Blockchain (Table 10)
| Avg latency | Throughput | Avg gas cost | Tamper detection | Blocks |
|---|---|---|---|---|
| 8.08 ms | 123.69 TPS | 31,541 units | 100% | 6 |

### Ablation and attention analysis (Tables 11 and 12)
| Configuration | Accuracy | F1 |
|---|---|---|
| ConvNeXt-Small without CBAM | 98.88% | 0.9888 |
| **AgriConvNet (full CBAM)** | **98.90%** | **0.9891** |

Grad-CAM attention concentration is **21.4% higher on average** with CBAM across all four datasets (0.1745 → 0.2119).

---

## Datasets

The datasets are **not included** in this repository. Download them from the original sources:

| Dataset | Images | Classes | Task | Conditions | Source |
|---|---|---|---|---|---|
| PlantVillage | 162,916 | 38 | Classification | Lab-controlled | [Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) |
| Multi-Crop Disease Dataset (Tamil Nadu) | 21,875 | 30+ | Detection | Real-field | [Mendeley Data](https://data.mendeley.com/datasets/6243z8r6t6/1) |
| Plant Leaf Disease Recognition | 25,077 | 4 | Segmentation | Mixed | [Mendeley Data](https://data.mendeley.com/datasets/5g238dv4ht/1) |
| Rice Leaf and Crop Disease Detection | 10,766 | 5 | Classification | Real-field | [Mendeley Data](https://data.mendeley.com/datasets/g7tcwvshff/1) |

**Preprocessing (Algorithm 1):**
- PIL integrity check and MD5 duplicate removal (1,080 images removed).
- Resize: 224 × 224 (classification), 640 × 640 (detection), 512 × 512 (segmentation).
- ImageNet normalisation.
- Stratified 70 / 15 / 15 split.
- Training-only augmentation: flips, ±15° rotation, ColorJitter, RandomErasing, and MixUp (p = 0.3, λ ~ Beta(0.2, 0.2)).
- The final training set contains 200,274 images; the total after augmentation is 227,474.

Place the extracted datasets in the project root under their original folder names.

---

## Repository structure

```
AgriChainNet/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── models/
│   ├── cbam.py                  # CBAM: channel (Eq. 5) + spatial (Eq. 6) attention
│   ├── agriconvnet.py           # AgriConvNet: ConvNeXt-Small + 4×CBAM + dual-pool head (Table 4)
│   └── unet.py                  # U-Net lesion segmentation (Sec 4.3)
├── utils/
│   ├── metrics.py               # Accuracy, F1, ROC-AUC, Dice, IoU, pixel acc, lesion %, severity
│   ├── losses.py                # BCE + Dice (Eq. 4), label-smoothed CE (Eq. 8)
│   └── transforms.py            # Preprocessing, augmentation, MixUp (Algorithm 1)
├── blockchain/
│   ├── ledger.py                # SHA-256 hash-chained ledger (Eq. 9), tamper detection
│   └── smart_contract.py        # Insurance / government alert / advisory rules (Eq. 10)
├── inference/
│   └── pipeline.py              # End-to-end detect → segment → classify → blockchain (Algorithm 2)
├── scripts/
│   ├── train_detection.py       # YOLOv8m training (Table 5 settings)
│   ├── train_segmentation.py    # U-Net training
│   ├── train_classification.py  # AgriConvNet training (--no-cbam for ablation)
│   └── blockchain_benchmark.py  # Latency, TPS, gas, tamper-detection test
├── experiments/                 # Original scripts that produced the paper results
├── datasets/README.md           # Download links and expected folder layout
├── docs/ARCHITECTURE.md         # Equations, layer table, smart-contract rules
├── results/                     # Metrics and logs (generated)
└── figures/                     # Paper figures
```

> **Code organisation.** `models/`, `utils/`, `blockchain/`, `inference/` and `scripts/` form a clean modular implementation of the method. The exact scripts used to generate the numbers reported in the paper are in `experiments/` (see `experiments/README.md`).

---

## Training configuration (Table 5)

| Parameter | Detection | Segmentation | Classification |
|---|---|---|---|
| Optimiser | AdamW | AdamW | AdamW |
| Learning rate | 1e-3 | 1e-4 | 1e-4 |
| Scheduler | Cosine | Cosine | Cosine warm restarts |
| Batch size | 16 | 8 | 64 |
| Epochs | 50 | 30 | 30 |
| Early stopping | 10 | 7 | 12 |
| Weight decay | 5e-4 | 1e-4 | 1e-4 |
| Image size | 640 | 512 | 224 |
| Label smoothing | — | — | 0.05 |

All modules used mixed precision (AMP).

**Hardware:** NVIDIA GeForce RTX 5050 Laptop GPU (8 GB), AMD Ryzen 7, 24 GB RAM, Windows 11, PyTorch 2.11, CUDA 12.8.

---

## Installation

```bash
git clone https://github.com/<your-username>/AgriChainNet.git
cd AgriChainNet
pip install -r requirements.txt
```

Install the CUDA build of PyTorch that matches your GPU from [pytorch.org](https://pytorch.org).

Quick check that the models build:
```bash
python -m models.agriconvnet     # Output shape: (2, 38)  |  Parameters: ~51M
python -m models.unet            # Output shape: (1, 1, 512, 512)
```

## Usage

```bash
# 1. Train the three modules
python -m scripts.train_detection      --data multicrop_fixed.yaml
python -m scripts.train_segmentation   --data final_datasets/segmentation
python -m scripts.train_classification --data final_datasets/classification

# 2. CBAM ablation (ConvNeXt-Small without CBAM)
python -m scripts.train_classification --data final_datasets/classification --no-cbam

# 3. Blockchain benchmark
python -m scripts.blockchain_benchmark

# 4. End-to-end inference on one image
python -m inference.pipeline --image leaf.jpg \
    --det weights/yolov8m_best.pt --seg weights/unet_best.pth --cls weights/agriconvnet_best.pth \
    --classes datasets/class_names.txt --farmer TN-001 --location Kanchipuram --crop Banana
```

Example output of the inference pipeline:
```json
{
  "record": {"disease": "...", "confidence": 0.97, "area_pct": 34.2, "severity": "Medium", "...": "..."},
  "block_hash": "9f2c...e41a",
  "action": {"action": "INSURANCE_PAYOUT", "amount": 17100.0, "notify_government": true}
}
```

## Smart contract logic

| Condition | Action |
|---|---|
| confidence > 0.85 **and** lesion area > 30% | Insurance payout + government notification |
| confidence > 0.95 | Government alert + early warning |
| otherwise | Monitoring advisory to farmer |

Severity levels: Low < 20%, Medium 20–40%, High > 40% diseased leaf area.



## Citation

```bibtex
@article{anandan2026agrichainnet,
  title   = {A Secure and Transparent Automated Crop Disease Analysis Framework Using Blockchain-Assisted Attention-Enhanced Multi-Task Deep Learning for Precision Agriculture},
  author  = {Anandan, Ranjith Kumar},
  journal = {<Journal name>},
  year    = {2026}
}
```

## License

Released under the MIT License. The datasets remain under their original licenses.

## Contact

Ranjith Kumar Anandan · SRM Institute of Science and Technology, Ramapuram, Chennai · ranjithdr.kumar@gmail.com
