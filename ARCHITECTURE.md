# Architecture Overview

## Repository Statistics

### Code Organization

```
Total Lines of Code: 3,714

Modular Source Code (src/):     1,303 lines (35%)
├── transforms.py              344 lines  (Medical image augmentations)
├── models.py                  464 lines  (SSL & downstream model architectures)
├── losses.py                  289 lines  (VICReg & triplet loss implementations)
├── optimizers.py              134 lines  (LARS & learning rate scheduling)
├── datasets.py                 55 lines  (Dataset utilities)
└── __init__.py                 17 lines  (Package metadata)

Main Scripts:                   2,411 lines (65%)
├── sicova_jigsaw_pt.py        184 lines  (VICReg pretraining)
├── triplet_jigsaw_pt.py        96 lines  (Triplet pretraining)
├── sicova_cam_ft.py           320 lines  (VICReg + CAM downstream)
├── sicova_abs_ft.py           630 lines  (VICReg + abstention)
├── triplet_cam_ft.py          415 lines  (Triplet + CAM)
└── triplet_abs_ft.py          766 lines  (Triplet + abstention)
```

## Module Dependency Graph

```
Main Scripts
    │
    ├─→ src.transforms
    │       └─→ PIL, cv2, numpy, torch, torchvision
    │
    ├─→ src.models
    │       ├─→ src.transforms (indirectly via composition)
    │       └─→ torch, torchvision.models
    │
    ├─→ src.losses
    │       └─→ torch, F (torch.nn.functional)
    │
    ├─→ src.optimizers
    │       └─→ torch, math
    │
    ├─→ src.datasets
    │       ├─→ numpy, PIL
    │       └─→ skimage.io
    │
    └─→ sklearn.metrics, pandas
```

## Class Hierarchy

### Transforms (`src/transforms.py`)

```
Object
├── RandomCropWithFallback
│   └── Randomly crops image with center crop fallback
├── CLAHETransform
│   └── CLAHE enhancement in LAB color space
├── RemoveBackgroundTransform
│   └── Removes low-intensity background pixels
├── MultiStageRandomChoiceJigsaw
│   └── Multi-scale jigsaw puzzle augmentation
├── ContrastiveLearningViewGenerator
│   └── Generates multiple augmented views
└── TwoViewTransform
    └── Generates two paired augmented views

Functions:
├── jigsaw_puzzle()
├── get_pretraining_transform()
└── get_finetuning_transform()
```

### Losses (`src/losses.py`)

```
torch.nn.Module
├── VICRegLoss
│   └── Combined VICReg (variance + invariance + covariance + cross-corr)
└── LabeledTripletLoss
    └── All-triplet mining with distance-based loss

Functions:
├── off_diagonal()       → Extract off-diagonal elements
├── var_loss()          → Variance regularization
├── invar_loss()        → Invariance (MSE)
├── cov_loss()          → Covariance regularization
└── cross_corr_loss()   → Cross-correlation loss
```

### Models (`src/models.py`)

```
VICReg Pretraining:
├── SiCoVa (nn.Module)
│   └── ResNet50 + 3-layer MLP → 8192-d embeddings

VICReg Downstream:
├── VICRegNet (nn.Module)
│   └── Feature extractor with expander
├── LinearClassifier (nn.Module)
│   └── Simple linear head on pooled features
└── CAMFinetuneVICReg (nn.Module)
    ├── Classification head
    ├── CAM extraction
    └── CAM refinement

Triplet Pretraining:
└── ResNet50TripletSelfSup (nn.Module)
    └── ResNet50 + projection → 128-d L2-normalized embeddings

Triplet Downstream:
├── CAMFinetuneTriplet (nn.Module)
│   └── Loads encoder checkpoint + adds CAM
└── ABSFinetuneTriplet (nn.Module)
    └── Temperature-calibrated classification

CAM Modules:
├── CAMExtractor (nn.Module)
│   └── 1×1 convolution → normalized attention map
└── RefinementCAM (nn.Module)
    ├── Multi-threshold masking
    └── Self-attention refinement

Other:
├── ABSFinetuneVICReg (nn.Module)
│   └── Temperature-scaled classifier
```

### Optimizers (`src/optimizers.py`)

```
torch.optim.Optimizer
└── LARS
    └── Layer-wise Adaptive Rate Scaling

Functions:
├── exclude_bias_and_norm() → Filter for LARS exclusion
└── adjust_learning_rate()  → Cosine annealing + warmup
```

### Datasets (`src/datasets.py`)

```
torch.utils.data.Dataset
└── CustomDataset
    └── Load images from paths with augmentation
```

## Data Flow

### Pretraining Pipeline

```
Raw Images
    ↓
[RandomCropWithFallback]
    ↓
[CLAHETransform]
    ↓
[RandomHorizontalFlip, ColorJitter, GaussianBlur]
    ↓
[MultiStageRandomChoiceJigsaw]
    ↓
[ToTensor, Normalize]
    ↓
[View1, View2] ← TwoViewTransform or ContrastiveLearningViewGenerator
    ↓
Batch [B, 3, H, W]
    ↓
SiCoVa/ResNet50Triplet
    ↓
Embeddings [B, D]
    ↓
VICRegLoss / LabeledTripletLoss
    ↓
Backward + LARS + Checkpointing
```

### Fine-tuning Pipeline

```
Labeled Images
    ↓
[RemoveBackgroundTransform]
    ↓
[CLAHETransform]
    ↓
[RandomHorizontalFlip]
    ↓
[Resize, Normalize]
    ↓
Batch [B, 3, 224/512, 224/512]
    ↓
Pretrained Backbone (VICRegNet / ResNet50TripletSelfSup)
    ↓
Feature Maps [B, 2048, H, W] + Pooled [B, 2048]
    ↓
Classification Head / CAM Extraction / Temperature Scaling
    ↓
Logits [B, 5]
    ↓
CrossEntropyLoss + CAMRefinementLoss (optional)
    ↓
Evaluation (Accuracy, F1, Cohen's Kappa, etc.)
```

## Training Workflows

### Workflow 1: VICReg Pretraining
```
sicova_jigsaw_pt.py
├── Load images from eyepacs/train
├── Apply get_pretraining_transform() + ContrastiveLearningViewGenerator
├── Initialize SiCoVa model
├── Initialize LARS optimizer
├── Initialize VICRegLoss
├── Train 200 epochs:
│   ├── accumulate gradients over 8 batches (eff batch 2048)
│   ├── compute VICReg loss
│   ├── LARS step with cosine annealing
│   └── save checkpoint every 10 epochs
└── Output: encoder-only checkpoints
```

### Workflow 2: Triplet Pretraining
```
triplet_jigsaw_pt.py
├── Load images from eyepacs/train (ImageFolder format)
├── Apply get_pretraining_transform() + TwoViewTransform
├── Initialize ResNet50TripletSelfSup
├── Initialize LabeledTripletLoss
├── Initialize SGD optimizer
├── Train 200 epochs:
│   ├── all-triplet mining per batch
│   ├── compute triplet loss
│   ├── backward + SGD step
│   └── save encoder checkpoint every 10 epochs
└── Output: encoder checkpoints
```

### Workflow 3: VICReg + CAM Fine-tuning
```
sicova_cam_ft.py
├── Load APTOS train/test (ImageFolder format)
├── Apply get_finetuning_transform()
├── Load pretrained VICRegNet backbone
├── Initialize CAMFinetuneVICReg
├── Fine-tune on labeled data:
│   ├── Extract feature maps + pooled features
│   ├── Classification head
│   ├── CAM extraction (1×1 conv)
│   ├── CAM refinement (multi-threshold + self-attention)
│   ├── Loss = CrossEntropy + CAMRefinement
│   └── Save checkpoint at eval epochs
└── Output: classification metrics + CAM visualizations
```

### Workflow 4: VICReg + Abstention Fine-tuning
```
sicova_abs_ft.py
├── Load APTOS train/test
├── Split train into train (90%) + calibration (10%)
├── Load pretrained VICRegNet
├── Initialize ABSFinetuneVICReg
├── Fine-tune on train split:
│   ├── CrossEntropy loss
│   ├── Adam optimizer
│   └── Save checkpoint at eval epochs
├── Temperature scaling calibration on calib split
├── Abstention sweep (0.5 → 0.95, Δ=0.05):
│   ├── For each threshold:
│   │   ├── Compute calibrated probabilities
│   │   ├── Count abstained samples
│   │   ├── Compute metrics on non-abstained
│   │   └── Log results
└── Output: accuracy vs. abstention coverage curve
```

## Extensibility Points

### Adding a New Transform
1. Create class in `src/transforms.py` inheriting from standard pattern
2. Add to pipeline functions
3. Use in scripts

### Adding a New Loss Function
1. Create function or class in `src/losses.py`
2. Implement `forward()` method for modules
3. Use in training loop

### Adding a New Model Architecture
1. Create class in `src/models.py`
2. Inherit from `nn.Module`
3. Implement `forward()` method
4. Use in scripts via import

### Adding a New Optimizer
1. Create class in `src/optimizers.py` inheriting from `torch.optim.Optimizer`
2. Implement `step()` method
3. Use in training initialization

## Performance Characteristics

### Memory Usage
- SiCoVa batch (256): ~22 GB (with grad accumulation over 8 steps)
- Triplet batch (256): ~12 GB
- Fine-tuning batch (32): ~4-6 GB

### Computational Complexity
- SiCoVa loss computation: O(B × D) where D=8192
- Triplet loss: O(B² × D) for distance matrix → typically B=256
- CAM refinement: O(B × H × W × D) with self-attention

### Scalability
- Modular design enables distributed training extensions
- Can be extended with torch.nn.parallel.DataParallel
- LARS optimizer designed for large-batch scaling

## Quality Assurance

### Code Standards
- ✓ Type hints in signatures
- ✓ Docstrings (Google style) for all public APIs
- ✓ PEP 8 compliant naming
- ✓ No circular imports
- ✓ Single responsibility per module

### Testing Opportunities
Each module can be tested independently:
- `test_transforms.py` – Augmentation correctness
- `test_losses.py` – Loss computation shapes
- `test_models.py` – Forward/backward pass validation
- `test_optimizers.py` – LARS step correctness

### Documentation
- ✓ Comprehensive README
- ✓ Detailed module docstrings
- ✓ Usage examples in README
- ✓ Hyperparameter guide
- ✓ Dataset organization examples
