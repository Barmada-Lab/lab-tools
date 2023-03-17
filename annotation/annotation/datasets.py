import random
from pathlib import Path

import torch
import tifffile
import numpy as np
from torch.cuda import current_blas_handle
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F

class RotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return F.rotate(x, angle)

class SegmentationDataset(Dataset):

    def __init__(self, images: list[Path], masks: list[Path], transform=None) -> None:
        assert len(images) > 0
        assert len(images) == len(masks)
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        mask_file = self.masks[idx]
        img_file = self.images[idx]

        # assuming there's not gonna be more than 256 classes...
        mask = tifffile.imread(mask_file).astype(np.int64)
        img = tifffile.imread(img_file).astype(np.float64)
        assert img.size == mask.size

        img = img / img.max()
        img = img[np.newaxis, ...]

        img = torch.as_tensor(img)
        mask = torch.as_tensor(mask)

        if self.transform is not None:
            img, mask = self.apply_transform(img, mask)

        return {
            "image": img,
            "mask": mask
        }

    def apply_transform(self, img, mask, current_transform=None):
        if current_transform is None:
            current_transform = self.transform

        if isinstance(current_transform, (transforms.Compose)):
            for transform in current_transform.transforms:
                img, mask = self.apply_transform(img, mask, transform)

        elif isinstance(current_transform, (transforms.RandomApply)):
            if current_transform.p >= random.random():
                img, mask = self.apply_transform(
                    img, mask, current_transform.transforms
                )

        elif isinstance(current_transform, (transforms.RandomChoice)):
            t = random.choice(current_transform.transforms)
            img, mask = self.apply_transform(img, mask, t)

        elif isinstance(current_transform, (transforms.RandomOrder)):
            order = list(range(len(current_transform.transforms)))
            random.shuffle(order)
            for i in order:
                img, mask = self.apply_transform(
                    img, mask, current_transform.transforms[i]
                )

        elif isinstance(
            current_transform,
            (
                transforms.CenterCrop,
                transforms.FiveCrop,
                transforms.TenCrop,
                transforms.ToTensor,
                transforms.Grayscale,
                transforms.Resize,
            ),
        ):
            img = current_transform(img)
            mask = current_transform(mask)

        elif isinstance(current_transform, (transforms.RandomHorizontalFlip)):
            if random.random() < current_transform.p:
                img = F.hflip(img)
                mask = F.hflip(mask)

        elif isinstance(current_transform, (transforms.RandomVerticalFlip)):
            if random.random() < current_transform.p:
                img = F.vflip(img)
                mask = F.vflip(mask)

        elif isinstance(current_transform, (transforms.RandomRotation)):
            angle = current_transform.get_params(current_transform.degrees)

            img = F.rotate(img, angle)
            mask = F.rotate(mask, angle)

        elif isinstance(current_transform, (RotationTransform)):
            angles = current_transform.angles
            angle = random.choice(angles)

            img = F.rotate(img, angle)
            mask = F.rotate(mask, angle)

        else:
            raise NotImplementedError(f"Not implemented: {current_transform}")
        return img, mask
