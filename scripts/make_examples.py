"""Создаёт тестовые наборы по 4 объектных изображения (128×128, 8 бит, PNG).

Эти изображения НЕ участвуют в обучении (обучение идёт на процедурно
сгенерированных данных) и служат для проверки обобщающей способности.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from doe.data import font_files  # noqa: E402
from doe.imageio import save_gray  # noqa: E402

N = 128


def _font(size, bold=True):
    files = font_files()
    pref = [f for f in files if ("DejaVuSans-Bold.ttf" in f if bold else "DejaVuSans.ttf" in f)]
    return ImageFont.truetype((pref or files)[0], size)


def text_img(text, size):
    img = Image.new("L", (N, N), 0)
    d = ImageDraw.Draw(img)
    f = _font(size)
    b = d.textbbox((0, 0), text, font=f)
    d.text(((N - (b[2] - b[0])) / 2 - b[0], (N - (b[3] - b[1])) / 2 - b[1]), text, fill=255, font=f)
    return np.asarray(img)


def shape_img(kind):
    img = Image.new("L", (N, N), 0)
    d = ImageDraw.Draw(img)
    c = N / 2
    if kind == "circle":
        d.ellipse([c - 40, c - 40, c + 40, c + 40], outline=255, width=8)
    elif kind == "triangle":
        d.polygon([(c, 20), (20, 104), (108, 104)], fill=255)
    elif kind == "square":
        d.rectangle([30, 30, 98, 98], outline=255, width=10)
        d.rectangle([52, 52, 76, 76], fill=255)
    elif kind == "star":
        import math
        pts = []
        for i in range(10):
            r = 50 if i % 2 == 0 else 20
            a = -math.pi / 2 + i * math.pi / 5
            pts.append((c + r * math.cos(a), c + r * math.sin(a)))
        d.polygon(pts, fill=255)
    return np.asarray(img)


def natural_imgs():
    from skimage import data, transform
    out = []
    for name in ["camera", "moon", "coins", "astronaut"]:
        im = getattr(data, name)()
        if im.ndim == 3:
            im = (im[..., :3] @ np.array([0.299, 0.587, 0.114])).astype(np.uint8)
        h, w = im.shape
        s = min(h, w)
        im = im[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
        im = transform.resize(im, (N, N), anti_aliasing=True, preserve_range=True)
        out.append(np.clip(im, 0, 255).astype(np.uint8))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="examples")
    args = ap.parse_args()
    root = Path(args.out)
    sets = {
        "digits": [text_img(t, 96) for t in "1234"],
        "shapes": [shape_img(k) for k in ["circle", "triangle", "square", "star"]],
        "letters": [text_img(t, 92) for t in "МИФИ"],
        "words": [text_img(t, 40) for t in ["ДОЭ", "ПВМС", "532", "НИЯУ"]],
    }
    try:
        sets["natural"] = natural_imgs()
    except Exception as e:  # skimage не обязателен
        print("natural: пропущено:", e)
    for name, imgs in sets.items():
        for i, im in enumerate(imgs, 1):
            save_gray(root / name / f"object_{i}.png", np.ascontiguousarray(im, dtype=np.uint8))
        print(name, "→", root / name)


if __name__ == "__main__":
    main()
