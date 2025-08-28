# Self-Supervised Learning for Medical Representation Learning

This repository provides self-supervised pretraining pipelines for retinal imaging and a downstream classifier used for evaluation. Two families of methods are implemented:

* **SiCoVa** – a variance/invariance/covariance framework with an additional cross-correlation term.
* **Triplet** – a supervised contrastive setup that optimises a triplet margin loss.

Both approaches support an optional multi-stage jigsaw puzzle augmentation to encourage locality-aware representations.

## Repository layout

```
├── SiCoVa_loss/          # Pretraining with SiCoVa loss
│   ├── jigsaw/           # SiCoVa + jigsaw augmentation
│   └── without_jigsaw/   # SiCoVa without jigsaw
├── triplet_loss/         # Triplet-loss pretraining
│   ├── jigsaw/
│   └── without_jigsaw/
├── fine_tuning/          # Downstream classification and CAM refinement
├── utils/                # Shared augmentations, models and losses
├── requirements.txt      # Python dependencies
└── LICENSE
```

## Installation

Create a Python environment (Python 3.9+ recommended) and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Datasets

Pretraining scripts expect images organised either as a flat folder of JPEG files (SiCoVa) or in `torchvision.datasets.ImageFolder` format (triplet). Paths to the data are defined near the top of each `pretrain.py` and should be edited to match your local copy of EyePACS or another retinal dataset.

### Download Links

- [EyePACS (Kaggle Diabetic Retinopathy Detection)](https://www.kaggle.com/c/diabetic-retinopathy-detection)  
- [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection)  
- [Messidor Dataset](https://www.adcis.net/en/third-party/messidor2/)
- [Fundus Dataset](https://zenodo.org/records/4647952#.YGNjXVUzbIU)  

Fine-tuning uses the APTOS dataset and likewise requires updating `TRAIN_PATH`, `VAL_PATH` and `PRETRAIN_DIR` in `fine_tuning/train.py` to point to your data and pretrained checkpoints.

## Pretraining

### SiCoVa

Run with jigsaw augmentation:

```bash
python SiCoVa_loss/jigsaw/pretrain.py
```

Run without jigsaw:

```bash
python SiCoVa_loss/without_jigsaw/pretrain.py
```

Checkpoints are written every ten epochs and include both encoder-only and full-model weights.

### Triplet loss

Run with jigsaw augmentation:

```bash
python triplet_loss/jigsaw/pretrain.py
```

Run without jigsaw:

```bash
python triplet_loss/without_jigsaw/pretrain.py
```

These scripts save encoder weights every ten epochs and write a final checkpoint after the last epoch.

## Fine-tuning

`fine_tuning/train.py` fine-tunes a CAM-regularised classifier on labelled data. The model loads a ResNet-based backbone, attaches a linear head and refines class activation maps through self-attention. Metrics and confusion matrices are written to a results directory while intermediate checkpoints are stored every ten epochs.

After adjusting dataset and checkpoint paths, launch fine-tuning with:

```bash
python fine_tuning/train.py
```

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgements

The codebase builds upon PyTorch and torchvision and was developed for research into self-supervised representation learning for medical images. Please cite this repository if you find it useful in your work.
