"""
Medical image transformations and augmentations.

Provides specialized transforms for retinal imaging including:
- Contrast Limited Adaptive Histogram Equalization (CLAHE)
- Random cropping with fallback
- Multi-stage jigsaw puzzle augmentation
- Background removal
- Complete augmentation pipelines for contrastive learning
"""

import numpy as np
import cv2
import random
from PIL import Image
import torchvision.transforms as T


# =============================================================================
# Basic Transforms
# =============================================================================

class RandomCropWithFallback:
    """
    Randomly crop image to target size with center crop fallback.
    
    Falls back to center crop if image is smaller than target size.
    """
    def __init__(self, size: int):
        """
        Args:
            size (int): Target crop size (square).
        """
        self.size = size
        self.center_crop = T.CenterCrop(size)
        
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL.Image): Input image.
        
        Returns:
            PIL.Image: Cropped image.
        """
        width, height = img.size
        if width < self.size or height < self.size:
            return self.center_crop(img)
        left = np.random.randint(0, width - self.size + 1)
        top = np.random.randint(0, height - self.size + 1)
        return img.crop((left, top, left + self.size, top + self.size))


class CLAHETransform:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Enhances local contrast in the L channel (LAB color space) to improve
    visibility of retinal features in medical images.
    """
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
        """
        Args:
            clip_limit (float): Clip limit for CLAHE. Default: 2.0.
            tile_grid_size (tuple): Grid size for adaptive histogram. Default: (8, 8).
        """
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL.Image): Input image.
        
        Returns:
            PIL.Image: CLAHE-enhanced image.
        """
        np_img = np.array(img)
        if np_img.ndim == 3 and np_img.shape[2] == 3:
            lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            lab = cv2.merge((l, a, b))
            np_img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            np_img = self.clahe.apply(np_img)
        return Image.fromarray(np_img)


class RemoveBackgroundTransform:
    """
    Remove low-intensity background pixels from medical images.
    
    Converts RGB to grayscale, applies binary threshold to identify background,
    and sets background pixels to zero.
    """
    def __init__(self, threshold: int = 10):
        """
        Args:
            threshold (int): Gray value threshold for background detection. Default: 10.
        """
        self.threshold = threshold

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL.Image): Input image.
        
        Returns:
            PIL.Image: Image with background removed.
        """
        arr = np.array(img)
        if arr.ndim == 3:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        else:
            gray = arr
        _, mask = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        arr[mask == 0] = 0
        return Image.fromarray(arr)


# =============================================================================
# Jigsaw Puzzle Augmentation
# =============================================================================

def jigsaw_puzzle(np_img: np.ndarray, grid_size: int = 4) -> np.ndarray:
    """
    Split image into grid and shuffle tiles randomly.
    
    Divides an image into grid_size x grid_size tiles, shuffles them randomly,
    and reassembles into a scrambled image for self-supervised learning.
    
    Args:
        np_img (np.ndarray): Input image (H, W, 3).
        grid_size (int): Number of tiles per dimension. Default: 4.
    
    Returns:
        np.ndarray: Shuffled image with same shape as input.
    """
    H, W, C = np_img.shape
    tile_w = W // grid_size
    tile_h = H // grid_size

    tiles = []
    for gx in range(grid_size):
        for gy in range(grid_size):
            left = gx * tile_w
            right = (gx + 1) * tile_w
            top = gy * tile_h
            bottom = (gy + 1) * tile_h
            tile = np_img[top:bottom, left:right, :]
            tiles.append(tile)

    random.shuffle(tiles)

    new_img = np.zeros_like(np_img)
    idx = 0
    for gx in range(grid_size):
        for gy in range(grid_size):
            left = gx * tile_w
            right = (gx + 1) * tile_w
            top = gy * tile_h
            bottom = (gy + 1) * tile_h
            new_img[top:bottom, left:right, :] = tiles[idx]
            idx += 1

    return new_img


class MultiStageRandomChoiceJigsaw:
    """
    Apply random jigsaw puzzle at one of several grid sizes.
    
    Randomly selects a grid size from puzzle_sizes and applies jigsaw shuffling.
    Transformation applied with probability p.
    """
    def __init__(self, puzzle_sizes: list = [8, 4, 2, 1], p: float = 1.0):
        """
        Args:
            puzzle_sizes (list): Grid sizes to randomly choose from. Default: [8, 4, 2, 1].
            p (float): Probability of applying transform. Default: 1.0.
        """
        self.puzzle_sizes = puzzle_sizes
        self.prob = p

    def __call__(self, pil_img: Image.Image) -> Image.Image:
        """
        Args:
            pil_img (PIL.Image): Input image.
        
        Returns:
            PIL.Image: Jigsaw-augmented image or original if probability check fails.
        """
        if random.random() > self.prob:
            return pil_img
        grid_size = random.choice(self.puzzle_sizes)
        np_img = np.array(pil_img)
        puzzle_np = jigsaw_puzzle(np_img, grid_size=grid_size)
        return Image.fromarray(puzzle_np)


# =============================================================================
# Augmentation Pipelines
# =============================================================================

def get_pretraining_transform(
    resize: int = 300,
    crop_size: int = 256,
    use_jigsaw: bool = True,
    clahe_clip: float = 2.0
) -> T.Compose:
    """
    Create full augmentation pipeline for SSL pretraining.
    
    Pipeline includes:
    - Resizing and CLAHE enhancement
    - Random cropping and horizontal flipping
    - Color jittering, grayscale conversion, Gaussian blur
    - Optional multi-scale jigsaw puzzle augmentation
    - Normalization for ImageNet-pretrained models
    
    Args:
        resize (int): Initial resize dimension. Default: 300.
        crop_size (int): Random crop size. Default: 256.
        use_jigsaw (bool): Include jigsaw augmentation. Default: True.
        clahe_clip (float): CLAHE clip limit. Default: 2.0.
    
    Returns:
        T.Compose: Transformation pipeline.
    """
    transforms = [
        T.Resize(resize),
        CLAHETransform(clip_limit=clahe_clip, tile_grid_size=(8, 8)),
        RandomCropWithFallback(crop_size),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(0.4, 0.4, 0.2, 0.1),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(kernel_size=(23, 23), sigma=(0.1, 2.0)),
    ]
    
    if use_jigsaw:
        transforms.append(
            MultiStageRandomChoiceJigsaw(puzzle_sizes=[8, 4, 2, 1], p=1.0)
        )
    
    transforms.extend([
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    return T.Compose(transforms)


def get_finetuning_transform(
    resize: int = 224,
    remove_bg: bool = True,
    use_clahe: bool = True
) -> T.Compose:
    """
    Create augmentation pipeline for downstream fine-tuning.
    
    Pipeline includes:
    - Background removal (optional)
    - CLAHE enhancement (optional)
    - Resizing and basic augmentation
    - Normalization for ImageNet models
    
    Args:
        resize (int): Resize dimension. Default: 224.
        remove_bg (bool): Remove background pixels. Default: True.
        use_clahe (bool): Apply CLAHE enhancement. Default: True.
    
    Returns:
        T.Compose: Transformation pipeline.
    """
    transforms = [T.Resize((resize, resize))]
    
    if remove_bg:
        transforms.append(RemoveBackgroundTransform(threshold=10))
    
    if use_clahe:
        transforms.append(CLAHETransform())
    
    transforms.extend([
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    return T.Compose(transforms)


# =============================================================================
# View Generators
# =============================================================================

class ContrastiveLearningViewGenerator:
    """
    Generate multiple augmented views of the same image for contrastive learning.
    
    Applies the same base transformation multiple times independently
    to create different views for contrastive loss computation (VICReg).
    """
    def __init__(self, base_transform: T.Compose, n_views: int = 2):
        """
        Args:
            base_transform (T.Compose): Base transformation pipeline.
            n_views (int): Number of augmented views to generate. Default: 2.
        """
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x: Image.Image) -> list:
        """
        Args:
            x (PIL.Image): Input image.
        
        Returns:
            list: List of n_views augmented tensor views.
        """
        return [self.base_transform(x) for _ in range(self.n_views)]


class TwoViewTransform:
    """
    Generate two independent augmentations of the same image.
    
    Applies the same base transformation twice independently to create
    paired views for contrastive/triplet loss computation.
    """
    def __init__(self, base_transform: T.Compose):
        """
        Args:
            base_transform (T.Compose): Base transformation pipeline.
        """
        self.base_transform = base_transform

    def __call__(self, img: Image.Image) -> tuple:
        """
        Args:
            img (PIL.Image): Input image.
        
        Returns:
            tuple: (augmented_view_1, augmented_view_2) - both tensors.
        """
        return self.base_transform(img), self.base_transform(img)
