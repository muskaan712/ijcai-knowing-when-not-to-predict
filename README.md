# Self-Supervised Learning for Medical Representation Learning

A PyTorch framework for self-supervised pretraining and downstream fine-tuning of retinal fundus image encoders using contrastive learning approaches.

## Overview

This repository implements two self-supervised learning (SSL) methods for medical image representation learning:

- **SiCoVa (VICReg)** – Variance-Invariance-Covariance Regularization with an additional cross-correlation term for decorrelated embeddings
- **Triplet Loss** – All-triplet mining strategy with labeled supervision during pretraining for metric learning

Both approaches support:
- **Multi-stage jigsaw puzzle augmentation** for locality-aware feature learning
- **Contrast Limited Adaptive Histogram Equalization (CLAHE)** for improved retinal feature visibility
- **Downstream fine-tuning** with optional CAM (Class Activation Map) refinement and confidence-based abstention

## Repository Structure

```
├── src/                          # Modular source code
│   ├── __init__.py
│   ├── transforms.py             # Medical image augmentations & pipelines
│   ├── models.py                 # Encoder architectures for SSL & downstream
│   ├── losses.py                 # VICReg & triplet loss implementations
│   ├── datasets.py               # Custom dataset classes
│   └── optimizers.py             # LARS optimizer & LR scheduling
├── sicova_jigsaw_pt.py           # SiCoVa pretraining with jigsaw (main entry point)
├── triplet_jigsaw_pt.py          # Triplet pretraining with jigsaw (main entry point)
├── sicova_cam_ft.py              # SiCoVa downstream with CAM refinement
├── sicova_abs_ft.py              # SiCoVa downstream with abstention
├── triplet_cam_ft.py             # Triplet downstream with CAM refinement
├── triplet_abs_ft.py             # Triplet downstream with abstention
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
└── README.md                     # This file
```

## Installation

Create a Python environment (Python 3.9+ recommended) and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Methods

### Pretraining Approaches

#### VICReg (SiCoVa)

VICReg combines four complementary loss terms to learn invariant and decorrelated representations:

- **Variance Loss** – Encourages sufficient variance in each embedding dimension
- **Invariance Loss** – Minimizes MSE between augmented views (embedding similarity)
- **Covariance Loss** – Regularizes covariance off-diagonals to zero (dimension decorrelation)
- **Cross-correlation Loss** – Decorrelates embeddings across views

**Paper**: Bardes et al., "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning" (ICLR 2022)

#### Triplet Loss

All-triplet mining with labeled supervision during pretraining:

- Mines all valid triplets (anchor, positive, negative) from each batch
- Minimizes triplet margin loss: $\max(d(a, p) - d(a, n) + \text{margin}, 0)$
- L2-normalizes embeddings for cosine similarity
- Supports power-scaled distance ($d^\gamma$) for tunable hardness

### Downstream Tasks

#### CAM Refinement
- Extracts Class Activation Maps (CAM) from feature maps via 1×1 convolution
- Applies multi-threshold masking and self-attention for feature localization
- Regularizes CAM with L1 loss during training for improved interpretability

#### Confidence-based Abstention (Selective Prediction)
- Post-hoc uncertainty estimation via temperature scaling calibration
- Allows selective prediction by rejecting low-confidence predictions
- Trade-off between accuracy and coverage via abstention threshold sweep

## Usage

### Dataset Setup

Organize your retinal imaging data as follows:

**Pretraining (unsupervised):**
```
DATA_ROOT/
└── eyepacs/
    └── train/
        ├── 0_left.jpeg
        ├── 0_right.jpeg
        └── ...
```

**Fine-tuning (supervised):**
```
DATA_ROOT/
└── aptos/
    ├── train/
    │   ├── 0/  (class 0: no DR)
    │   ├── 1/  (class 1: mild)
    │   ├── 2/  (class 2: moderate)
    │   ├── 3/  (class 3: severe)
    │   └── 4/  (class 4: proliferative)
    └── test/
        ├── 0/ ... 4/
```

Update data paths in pretraining/fine-tuning scripts:
```python
DATA_ROOT = os.getenv("DATA_ROOT", "/path/to/data")
EXPERIMENTS_ROOT = os.getenv("EXPERIMENTS_ROOT", "/path/to/experiments")
```

### Pretraining

#### SiCoVa with Multi-stage Jigsaw

```bash
python sicova_jigsaw_pt.py
```

**Key features:**
- ResNet50 encoder + 3-layer MLP expander (→8192-d embeddings)
- 256 batch size, effective batch 2048 via gradient accumulation (8 steps)
- LARS optimizer with cosine annealing + linear warmup (10 epochs)
- Saves full checkpoint every 10 epochs, encoder-only every 10 epochs

#### Triplet Loss with Multi-stage Jigsaw

```bash
python triplet_jigsaw_pt.py
```

**Key features:**
- ResNet50 encoder + linear projection head (→128-d embeddings)
- L2-normalized embeddings for triplet mining
- All-triplet mining with margin=1.0
- SGD optimizer (lr=0.01, momentum=0.9)
- Saves encoder checkpoint every 10 epochs

### Fine-tuning

#### SiCoVa with CAM Refinement

```bash
python sicova_cam_ft.py
```

- Loads pretrained VICReg backbone
- Adds linear classification head (2048 → 5 classes)
- CAM extraction from res5 feature maps (7×7)
- Refinement via self-attention over multi-threshold masks
- Evaluates at epochs specified in `EVAL_EPOCHS`

#### SiCoVa with Confidence Abstention

```bash
python sicova_abs_ft.py
```

- Confidence-based selective prediction
- Temperature scaling calibration on hold-out split (10% of train)
- Dense abstention threshold sweep (0.5 → 0.95, Δ=0.05)
- Records accuracy, precision, recall, F1, Cohen's Kappa, abstention coverage

#### Triplet downstream approaches

```bash
python triplet_cam_ft.py      # CAM refinement
python triplet_abs_ft.py      # Confidence abstention
```

## Key Components

### Transforms (`src/transforms.py`)

**Medical-specific augmentations:**
- `CLAHETransform` – Adaptive histogram equalization in LAB space
- `RemoveBackgroundTransform` – Zero-out low-intensity pixels
- `RandomCropWithFallback` – Adaptive random/center cropping
- `MultiStageRandomChoiceJigsaw` – Multi-scale puzzle at [8, 4, 2, 1]

**Pipelines:**
- `get_pretraining_transform()` – Full augmentation + jigsaw for SSL
- `get_finetuning_transform()` – Light augmentation for downstream
- `ContrastiveLearningViewGenerator` – Multiple independent views
- `TwoViewTransform` – Two views for contrastive/triplet loss

### Losses (`src/losses.py`)

**VICReg components:**
- `var_loss()` – Variance regularization
- `invar_loss()` – Invariance (MSE)
- `cov_loss()` – Covariance regularization
- `cross_corr_loss()` – Cross-correlation decorrelation
- `VICRegLoss` – Combined loss module

**Triplet loss:**
- `LabeledTripletLoss` – All-triplet mining with pairwise distance matrix
- Efficient masking for valid triplets
- Power-scaled distances ($d^\gamma$)

### Models (`src/models.py`)

**Pretraining encoders:**
- `SiCoVa` – ResNet50 + 3-layer MLP for VICReg
- `ResNet50TripletSelfSup` – ResNet50 + projection + L2-norm for triplet

**Fine-tuning models:**
- `VICRegNet` – Feature extractor for VICReg downstream
- `LinearClassifier` – Simple linear head on pooled features
- `CAMFinetuneVICReg` – Classification + CAM extraction/refinement
- `CAMFinetuneTriplet` – Triplet encoder + CAM downstream
- `ABSFinetuneVICReg` – Temperature-calibrated classifier
- `ABSFinetuneTriplet` – Triplet with abstention

**CAM modules:**
- `CAMExtractor` – 1×1 conv → normalized activation map
- `RefinementCAM` – Multi-threshold masking + self-attention

### Optimizers (`src/optimizers.py`)

- `LARS` – Layer-wise Adaptive Rate Scaling for large-batch training
- `exclude_bias_and_norm()` – Filter for excluding bias/norm from LARS/decay
- `adjust_learning_rate()` – Cosine annealing + linear warmup

## Performance Notes

**Effective Batch Sizes:**
- SiCoVa: 2048 (256 loader batch × 8 accumulation steps)
- Triplet: 256 (paired samples, 128 anchor-positive pairs + negatives)

**Recommended Hardware:**
- GPU: NVIDIA A100 or H100 (40GB+ VRAM)
- CPU: Multi-core for efficient data loading (num_workers=16-32)

**Training Times (approx):**
- SiCoVa: 200 epochs → 48-72 hours on A100
- Triplet: 200 epochs → 24-36 hours on A100

## Datasets

Public retinal fundus imaging datasets:

- [EyePACS (Kaggle)](https://www.kaggle.com/c/diabetic-retinopathy-detection)  
- [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection)  
- [Messidor-2](https://www.adcis.net/en/third-party/messidor2/)
- [Fundus Dataset (Zenodo)](https://zenodo.org/records/4647952)

## Configuration & Customization

Edit the following in pretraining/fine-tuning scripts:

```python
# Data paths
DATA_ROOT = os.getenv("DATA_ROOT", "/path/to/data")
EXPERIMENTS_ROOT = os.getenv("EXPERIMENTS_ROOT", "/path/to/experiments")

# Training hyperparameters
BATCH_SIZE = 256
TOTAL_DS_EPOCHS = 200
LR = 0.003
WEIGHT_DECAY = 1e-4

# Model parameters
EMBEDDING_DIM = 8192  # SiCoVa
NUM_CLASSES = 5       # DR classification (0-4)
```

For SLURM job arrays (distributed fine-tuning over multiple checkpoints):

```bash
sbatch --array=0-9 fine_tune_job.sh
```

Automatic checkpoint slicing via:
```python
CHUNK_SIZE = 5
CHUNK_IDX = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
```

## Reproducibility

To ensure reproducible results:

```python
import torch
import random
import numpy as np

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
```

## License

This project is released under the [MIT License](LICENSE).

<!-- ## Citation

If you find this repository useful in your research, please cite:

```bibtex
@inproceedings{vicreg2022,
  title={VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning},
  author={Bardes, Adrien and Ponce, Jean and LeCun, Yann},
  booktitle={ICLR},
  year={2022}
}

@inproceedings{facenet2015,
  title={FaceNet: A Unified Embedding for Face Recognition and Clustering},
  author={Schroff, Florian and Kalenichenko, Dmitry and Philbin, James},
  booktitle={CVPR},
  year={2015}
}
``` -->

## Acknowledgements

The codebase builds upon PyTorch, torchvision, and OpenCV. Special thanks to researchers in self-supervised learning and medical image analysis communities for inspiration and guidance.

For questions or issues, please open an issue on the repository.
