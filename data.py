import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


SUN_SUBSETS = {
    "Easy-Seen": "TestEasyDataset/Seen",
    "Easy-Unseen": "TestEasyDataset/Unseen",
    "Hard-Seen": "TestHardDataset/Seen",
    "Hard-Unseen": "TestHardDataset/Unseen",
}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def natural_key(path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(path))]


def image_tensor(image):
    array = np.asarray(image, dtype=np.float32) / 255
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


def transform_pair(images, mask, size, training):
    images = [image.resize((size, size), Image.Resampling.BICUBIC) for image in images]
    if training:
        mask = mask.resize((size, size), Image.Resampling.NEAREST)
        crop = random.uniform(0.85, 1)
        extent = max(1, round(size * crop))
        left, top = random.randint(0, size - extent), random.randint(0, size - extent)
        box = (left, top, left + extent, top + extent)
        images = [image.crop(box).resize((size, size), Image.Resampling.BICUBIC) for image in images]
        mask = mask.crop(box).resize((size, size), Image.Resampling.NEAREST)
        if random.random() < 0.5:
            images = [image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for image in images]
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        angle = random.uniform(-15, 15)
        images = [image.rotate(angle, resample=Image.Resampling.BICUBIC) for image in images]
        mask = mask.rotate(angle, resample=Image.Resampling.NEAREST)
        for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
            factor = random.uniform(0.8, 1.2)
            images = [enhancer(image).enhance(factor) for image in images]
    return [image_tensor(image) for image in images], torch.from_numpy(np.array(mask, dtype=np.int64))


class ImageDataset(Dataset):
    def __init__(self, config, split, fold=0):
        self.root = Path(config["data_root"])
        manifest = json.loads(Path(config["split_file"]).read_text())
        if "folds" in manifest:
            manifest = manifest["folds"][f"fold{fold}"]
        self.samples = manifest[split]
        self.size = config["image_size"]
        self.classes = config["classes"]
        self.training = split == "train"
        if not self.samples:
            raise ValueError(f"Empty {split} split")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_name, mask_name = self.samples[index]
        if Path(image_name).is_absolute() or Path(mask_name).is_absolute():
            raise ValueError("Split entries must be relative to data_root")
        with Image.open(self.root / image_name) as handle:
            image = handle.convert("RGB")
        with Image.open(self.root / mask_name) as handle:
            mask = handle.copy()
        if len(mask.getbands()) != 1 and self.classes == 1:
            mask = mask.convert("L")
        if len(mask.getbands()) != 1:
            raise ValueError("Masks must contain one integer label per pixel")
        if image.size != mask.size:
            raise ValueError(f"Image and mask sizes differ: {image_name}")
        images, labels = transform_pair([image], mask, self.size, self.training)
        labels = (labels > 0).long() if self.classes == 1 else labels
        if labels.min() < 0 or labels.max() > self.classes:
            raise ValueError("Mask labels must be in 0..classes")
        return {"image": images[0], "mask": labels, "name": image_name, "sequence": ""}


class SUNSeg(Dataset):
    def __init__(self, config, subset="train"):
        self.root = Path(config["data_root"])
        self.size = config["image_size"]
        self.training = subset == "train"
        directory = self.root / ("TrainDataset" if self.training else SUN_SUBSETS[subset])
        frames = directory / "Frame"
        masks = directory / "GT"
        self.samples = []
        self.clips = 0
        for clip in sorted(frames.iterdir(), key=natural_key):
            if not clip.is_dir():
                continue
            sequence = f"{subset}/{clip.name}"
            names = sorted((p for p in clip.iterdir() if p.suffix.lower() in EXTENSIONS), key=natural_key)
            for index, image in enumerate(names):
                mask = masks / clip.name / f"{image.stem}.png"
                if not mask.is_file():
                    raise FileNotFoundError(mask)
                previous = names[index - 1] if index else image
                self.samples.append((image, previous, mask, sequence, index > 0))
            self.clips += bool(names)
        if not self.samples:
            raise ValueError(f"No SUN-SEG frames in {frames}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, previous, mask_path, sequence, valid = self.samples[index]
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        with Image.open(mask_path) as handle:
            mask = handle.convert("L")
        images = [image]
        if self.training:
            with Image.open(previous) as handle:
                images.append(handle.convert("RGB"))
        images, labels = transform_pair(images, mask, self.size, self.training)
        sample = {"image": images[0], "mask": (labels > 0).long(),
                  "name": str(path.relative_to(self.root)), "sequence": sequence}
        if self.training:
            sample.update(previous=images[1], history_valid=torch.tensor(valid))
        return sample
