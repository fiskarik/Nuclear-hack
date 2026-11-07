"""Procedural, seed-separated image quartets; no external dataset is required."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import torch
from torch.utils.data import Dataset


def image_array(path, size=128):
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "L" or image.size != (size, size):
            raise ValueError(f"{path}: expected an 8-bit grayscale PNG ({size}x{size}), got {image.format}/{image.mode}/{image.size}")
        return np.asarray(image, dtype=np.uint8).copy()


def load_quartet(paths, size=128):
    if len(paths) != 4:
        raise ValueError("supply exactly four PNGs in increasing distance order")
    arrays = np.stack([image_array(path, size) for path in paths])
    if (arrays.max(axis=(-2, -1)) == 0).any():
        raise ValueError("completely black target planes are not supported")
    return torch.from_numpy(arrays.astype(np.float32) / 255)[None]


def procedural_image(seed, size=128):
    rng = np.random.default_rng(seed)
    scale = 2
    image = Image.new("L", (size * scale, size * scale), 0)
    draw = ImageDraw.Draw(image)
    kind = int(rng.integers(0, 4))
    if kind == 0:
        font = ImageFont.load_default(size=int(rng.integers(38, 80)) * scale)
        text = str(rng.integers(0, 10))
        box = draw.textbbox((0, 0), text, font=font)
        w, h = box[2] - box[0], box[3] - box[1]
        x = (size * scale - w) // 2 + int(rng.integers(-12, 13)) * scale
        y = (size * scale - h) // 2 - box[1] + int(rng.integers(-12, 13)) * scale
        draw.text((x, y), text, font=font, fill=255)
    else:
        for _ in range(int(rng.integers(1, 5))):
            x, y = rng.integers(size // 8, size * 3 // 4, 2)
            w, h = rng.integers(size // 8, size // 2, 2)
            bounds = tuple(int(v * scale) for v in (x, y, min(x + w, size - 5), min(y + h, size - 5)))
            value = int(rng.integers(140, 256))
            if kind == 1:
                draw.ellipse(bounds, fill=value)
            elif kind == 2:
                draw.rectangle(bounds, fill=value)
            else:
                draw.ellipse(bounds, outline=value, width=int(rng.integers(3, 8)) * scale)
        if kind == 3:
            draw.line([(20 * scale, 105 * scale), (100 * scale, 30 * scale)], fill=220, width=3 * scale)
    image = image.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.5, 1.6)) * scale))
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    array = np.array(image)
    array = np.rint(array.astype(np.float64) * (255 / max(int(array.max()), 1))).astype(np.uint8)
    return array


def generate_dataset(root, train=128, validation=24, test=24, seed=2026):
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing dataset: {root}")
    if min(train, validation, test) < 1:
        raise ValueError("all three splits must be nonempty")
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"source": "procedurally generated demonstration data, not official hackathon inputs",
                "seed": seed, "size": 128, "mode": "L", "splits": {}}
    for split_id, (split, count) in enumerate((("train", train), ("validation", validation), ("test", test))):
        records = []
        for index in range(count):
            sample = root / split / f"sample_{index:04d}"
            sample.mkdir(parents=True)
            seeds = []
            hashes = []
            for plane in range(4):
                image_seed = seed + split_id * 1_000_000 + index * 16 + plane * 101
                path = sample / f"plane_{plane + 1}.png"
                Image.fromarray(procedural_image(image_seed)).save(path)
                seeds.append(image_seed)
                hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
            records.append({"path": str(sample.relative_to(root)), "image_seeds": seeds, "sha256": hashes})
        manifest["splits"][split] = records
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


class QuartetDataset(Dataset):
    def __init__(self, root, split):
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text())
        self.samples = [root / item["path"] for item in manifest["splits"][split]]
        if not self.samples:
            raise ValueError(f"empty dataset split: {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return load_quartet([self.samples[index] / f"plane_{i}.png" for i in range(1, 5)])[0]
