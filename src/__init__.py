"""
Self-Supervised Learning for Medical Representation Learning
============================================================

A PyTorch framework for self-supervised pretraining and downstream fine-tuning
of medical image encoders using VICReg and Triplet Loss approaches.

Key modules:
- transforms: Medical image augmentations (CLAHE, jigsaw puzzle, etc.)
- losses: Loss functions for VICReg and triplet learning
- models: ResNet50 encoders for SSL pretraining and downstream tasks
- datasets: Custom dataset implementations
- optimizers: LARS optimizer for large-batch training
"""

__version__ = "1.0.0"
__author__ = "Medical Imaging Research Group"
