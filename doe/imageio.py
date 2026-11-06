"""Чтение и запись 8-битных одноканальных PNG 128×128."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from PIL import Image


def load_gray(path: str | Path, size: int = 128) -> np.ndarray:
    """Загружает изображение как uint8 (size×size), приводя к одному каналу.

    Изображения по условию уже 128×128, 8 бит, grayscale. Если пришло RGB/RGBA
    или другой размер — изображение конвертируется с предупреждением.
    """
    img = Image.open(path)
    if img.mode not in ("L", "I;16", "I", "F"):
        warnings.warn(f"{path}: режим {img.mode} → преобразован в оттенки серого (L)")
    if img.mode in ("I;16", "I", "F"):
        arr = np.asarray(img, dtype=np.float64)
        arr = 255.0 * (arr - arr.min()) / max(arr.max() - arr.min(), 1e-12)
        img = Image.fromarray(arr.astype(np.uint8), mode="L")
    else:
        img = img.convert("L")
    if img.size != (size, size):
        warnings.warn(f"{path}: размер {img.size} → масштабирован до {size}×{size}")
        img = img.resize((size, size), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8).copy()


def save_gray(path: str | Path, arr: np.ndarray) -> None:
    """Сохраняет uint8-массив как 8-битный одноканальный PNG."""
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        raise TypeError("ожидается uint8")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(path, format="PNG")


def amplitude_to_uint8(amp: np.ndarray, target: np.ndarray | None = None,
                       mode: str = "max") -> np.ndarray:
    """Модуль поля → 8-битное изображение.

    mode="max" — нормировка на максимум (|K| / max|K| · 255);
    mode="fit" — масштаб по МНК относительно исходного изображения
                 (s = <A,R>/<R,R>, как в Peng et al., Neural Holography, 2020),
                 значения выше 255 насыщаются, как у камеры.
    """
    amp = np.asarray(amp, dtype=np.float64)
    if mode == "max" or target is None:
        scale = 255.0 / max(amp.max(), 1e-12)
    elif mode == "fit":
        t = np.asarray(target, dtype=np.float64)
        scale = float((amp * t).sum() / max((amp * amp).sum(), 1e-12))
    else:
        raise ValueError(mode)
    return np.clip(np.round(amp * scale), 0, 255).astype(np.uint8)
