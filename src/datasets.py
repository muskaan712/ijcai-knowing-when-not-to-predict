"""
Custom dataset implementations for self-supervised learning.
"""

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from skimage import io


class CustomDataset(Dataset):
    """
    Load images from file paths with optional augmentation.
    
    Reads images from disk, normalizes pixel values, and applies
    transformation pipeline if provided. Supports various image formats
    and handles dtype conversions automatically.
    """
    def __init__(self, list_images: list, transform=None):
        """
        Args:
            list_images (list): List of image file paths.
            transform (callable): Optional transformation to apply. Default: None.
        """
        self.list_images = list_images
        self.transform = transform

    def __len__(self) -> int:
        """Returns number of images in dataset."""
        return len(self.list_images)

    def __getitem__(self, idx: int):
        """
        Load and transform a single image.
        
        Args:
            idx (int): Image index.
        
        Returns:
            tensor or list: Transformed image(s).
        """
        img_name = self.list_images[idx]
        image = io.imread(img_name)
        
        # Normalize pixel values to uint8
        if image.dtype != np.uint8:
            if image.max() <= 1:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        pil_img = Image.fromarray(image)
        if self.transform:
            pil_img = self.transform(pil_img)
        return pil_img
