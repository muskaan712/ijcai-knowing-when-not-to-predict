"""Data augmentation utilities for fine-tuning."""

import numpy as np
from PIL import Image
import torchvision.transforms as T


class RemoveBackgroundTransform:
    """Zero-out low-intensity background pixels."""

    def __init__(self, threshold: int = 10):
        """Store threshold below which pixels are removed."""
        self.threshold = threshold

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply background removal to ``img``."""
        import cv2

        arr = np.array(img)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        arr[mask == 0] = 0
        return Image.fromarray(arr)


class CLAHETransform:
    """Apply CLAHE to enhance local contrast."""

    def __init__(self, clip_limit: float = 2.0, tile_grid_size=(8, 8)):
        """Initialize CLAHE with given parameters."""
        import cv2

        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img: Image.Image) -> Image.Image:
        """Enhance image using CLAHE."""
        import cv2

        arr = np.array(img)
        if arr.ndim == 3 and arr.shape[2] == 3:
            lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            lab = cv2.merge((l, a, b))
            arr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            arr = self.clahe.apply(arr)
        return Image.fromarray(arr)


def build_transform() -> T.Compose:
    """Create the transformation pipeline for training/validation."""

    return T.Compose(
        [
            T.Resize((512, 512)),
            RemoveBackgroundTransform(10),
            CLAHETransform(clip_limit=2.0),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ]
    )
