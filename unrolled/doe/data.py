"""Синтетический набор данных для обучения нейросети.

Нейросеть обучается без учителя (physics-informed): ей не нужны «правильные»
ДОЭ, достаточно разнообразных объектных изображений — функция потерь считает
восстановление через дифференцируемую модель Френеля. Поэтому набор данных
генерируется процедурно и бесконечно:

* символы и короткие надписи (цифры, латиница, кириллица, знаки) разными шрифтами;
* геометрические фигуры (эллипсы, кольца, многоугольники, звёзды, линии, дуги);
* массивы точек и решётки;
* гладкие «природоподобные» текстуры (1/f-шум, пятна, градиенты);
* комбинации перечисленного, негативы, полутона.

Одна обучающая выборка — 4 независимых изображения (по одному на плоскость).
"""
from __future__ import annotations

import glob
import math
import os
import random
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SIZE = 128

_CHARS = ("0123456789" "ABCDEFGHIJKLMNOPQRSTUVWXYZ" "abcdefghijklmnopqrstuvwxyz"
          "АБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЭЮЯ" "абвгдежзиклмнопрстуфхцчшщэюя" "+-=*#@&%?!<>()[]{}λπΩΣΔ")
_WORDS = ["МИФИ", "ДОЭ", "ПВМС", "НИЯУ", "ЛАЗЕР", "ФОТОН", "OPTICS", "HOLO", "DOE", "SLM",
          "532", "2026", "λ/2", "AI", "NN", "π", "ЛаПлаз", "CGH", "3D", "LASER"]


@lru_cache(maxsize=1)
def font_files() -> tuple[str, ...]:
    """TTF/OTF шрифты: из matplotlib (есть везде, где есть matplotlib) и системные."""
    paths: list[str] = []
    try:
        import matplotlib  # noqa: WPS433

        paths += glob.glob(os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf", "*.ttf"))
    except Exception:  # pragma: no cover - matplotlib не обязателен
        pass
    for root in ("/usr/share/fonts", "/usr/local/share/fonts"):
        paths += glob.glob(os.path.join(root, "**", "*.ttf"), recursive=True)
        paths += glob.glob(os.path.join(root, "**", "*.otf"), recursive=True)
    good = []
    for p in sorted(set(paths)):
        name = os.path.basename(p).lower()
        if any(s in name for s in ("symbol", "emoji", "dingbat", "wingding", "cmex", "cmsy", "stixnonuni",
                                    "stixsizes", "math", "lastresort", "sample")):
            continue
        try:
            f = ImageFont.truetype(p, 24)
            # шрифт должен содержать кириллицу, латиницу и цифры — иначе «квадратики»
            notdef = bytes(f.getmask("\U0010FFFD"))
            if any(bytes(f.getmask(ch)) == notdef or not any(bytes(f.getmask(ch))) for ch in "ЖяA5"):
                continue
        except Exception:
            continue
        good.append(p)
    return tuple(good)


def _font(rng: random.Random, size: int):
    files = font_files()
    if files:
        return ImageFont.truetype(rng.choice(files), size)
    return ImageFont.load_default(size=size)


def _level(rng: random.Random, lo: int = 80) -> int:
    return rng.choice([255, 255, 255, rng.randint(lo, 255)])


def draw_text(rng: random.Random) -> Image.Image:
    img = Image.new("L", (SIZE, SIZE), 0)
    if rng.random() < 0.65:
        text = "".join(rng.choice(_CHARS) for _ in range(rng.choice([1, 1, 1, 2, 2, 3])))
    else:
        text = rng.choice(_WORDS)
    size = rng.randint(28, 110) if len(text) == 1 else rng.randint(18, 70)
    font = _font(rng, size)
    layer = Image.new("L", (SIZE * 2, SIZE * 2), 0)
    d = ImageDraw.Draw(layer)
    bbox = d.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = SIZE - w // 2 - bbox[0]
    y = SIZE - h // 2 - bbox[1]
    stroke = rng.choice([0, 0, 0, 1, 2])
    d.text((x, y), text, fill=_level(rng), font=font, stroke_width=stroke, stroke_fill=_level(rng))
    if rng.random() < 0.5:
        layer = layer.rotate(rng.uniform(-35, 35), resample=Image.BILINEAR)
    # случайный сдвиг центра
    cx = SIZE + rng.randint(-SIZE // 4, SIZE // 4)
    cy = SIZE + rng.randint(-SIZE // 4, SIZE // 4)
    layer = layer.crop((cx - SIZE // 2, cy - SIZE // 2, cx + SIZE // 2, cy + SIZE // 2))
    img.paste(layer)
    return img


def _poly(rng, cx, cy, r, k, star=False):
    pts = []
    a0 = rng.uniform(0, 2 * math.pi)
    m = 2 * k if star else k
    for i in range(m):
        rr = r if (not star or i % 2 == 0) else r * rng.uniform(0.35, 0.6)
        a = a0 + 2 * math.pi * i / m
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return pts


def draw_shapes(rng: random.Random) -> Image.Image:
    img = Image.new("L", (SIZE, SIZE), 0)
    d = ImageDraw.Draw(img)
    for _ in range(rng.choice([1, 1, 2, 3, 4, 6])):
        kind = rng.choice(["ellipse", "rect", "poly", "star", "line", "ring", "arc", "cross"])
        cx, cy = rng.uniform(16, SIZE - 16), rng.uniform(16, SIZE - 16)
        r = rng.uniform(6, 50)
        fill = _level(rng)
        width = rng.randint(1, 8)
        filled = rng.random() < 0.6
        box = [cx - r, cy - r * rng.uniform(0.4, 1.0), cx + r, cy + r * rng.uniform(0.4, 1.0)]
        box = [min(box[0], box[2]), min(box[1], box[3]), max(box[0], box[2]), max(box[1], box[3])]
        if kind == "ellipse":
            d.ellipse(box, fill=fill if filled else None, outline=fill, width=width)
        elif kind == "rect":
            d.rectangle(box, fill=fill if filled else None, outline=fill, width=width)
        elif kind in ("poly", "star"):
            pts = _poly(rng, cx, cy, r, rng.randint(3, 8), star=(kind == "star"))
            if filled:
                d.polygon(pts, fill=fill)
            else:
                d.line(pts + [pts[0]], fill=fill, width=width)
        elif kind == "line":
            d.line([(cx, cy), (rng.uniform(0, SIZE), rng.uniform(0, SIZE))], fill=fill, width=width)
        elif kind == "ring":
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=fill, width=max(2, width))
        elif kind == "arc":
            a = rng.uniform(0, 360)
            d.arc([cx - r, cy - r, cx + r, cy + r], a, a + rng.uniform(60, 300), fill=fill, width=width)
        else:
            d.line([(cx - r, cy), (cx + r, cy)], fill=fill, width=width)
            d.line([(cx, cy - r), (cx, cy + r)], fill=fill, width=width)
    return img


def draw_points(rng: random.Random) -> Image.Image:
    img = Image.new("L", (SIZE, SIZE), 0)
    d = ImageDraw.Draw(img)
    if rng.random() < 0.5:
        # регулярная решётка точек
        k = rng.randint(2, 8)
        step = rng.uniform(8, 110 / max(k - 1, 1))
        x0 = SIZE / 2 - step * (k - 1) / 2
        rad = rng.uniform(0.5, min(step / 3, 6))
        for i in range(k):
            for j in range(k):
                x, y = x0 + i * step, x0 + j * step
                d.ellipse([x - rad, y - rad, x + rad, y + rad], fill=_level(rng, 150))
    elif rng.random() < 0.5:
        for _ in range(rng.randint(3, 40)):
            x, y = rng.uniform(4, SIZE - 4), rng.uniform(4, SIZE - 4)
            rad = rng.uniform(0.5, 4)
            d.ellipse([x - rad, y - rad, x + rad, y + rad], fill=_level(rng))
    else:
        # решётка полос
        period = rng.uniform(4, 24)
        duty = rng.uniform(0.2, 0.6)
        ang = rng.uniform(0, math.pi)
        yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
        t = (xx * math.cos(ang) + yy * math.sin(ang)) / period
        arr = ((t - np.floor(t)) < duty).astype(np.float32) * _level(rng)
        mask = np.zeros_like(arr)
        r = rng.uniform(20, 64)
        mask[(xx - SIZE / 2) ** 2 + (yy - SIZE / 2) ** 2 < r * r] = 1
        if rng.random() < 0.5:
            arr *= mask
        return Image.fromarray(arr.astype(np.uint8), "L")
    return img


def draw_texture(rng: random.Random, nrng: np.random.Generator) -> Image.Image:
    kind = rng.choice(["fnoise", "blobs", "gradient"])
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32) / SIZE
    if kind == "fnoise":
        # 1/f^β шум — статистика, похожая на природные изображения
        beta = rng.uniform(1.5, 3.0)
        fx = np.fft.fftfreq(SIZE)[:, None]
        fy = np.fft.fftfreq(SIZE)[None, :]
        f = np.sqrt(fx ** 2 + fy ** 2)
        f[0, 0] = 1.0
        spec = (nrng.normal(size=(SIZE, SIZE)) + 1j * nrng.normal(size=(SIZE, SIZE))) / f ** (beta / 2)
        arr = np.real(np.fft.ifft2(spec))
    elif kind == "blobs":
        arr = np.zeros((SIZE, SIZE), np.float32)
        for _ in range(rng.randint(2, 8)):
            cx, cy, s = rng.random(), rng.random(), rng.uniform(0.03, 0.2)
            arr += rng.uniform(0.3, 1) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))
    else:
        a = rng.uniform(0, 2 * math.pi)
        arr = np.cos(a) * xx + np.sin(a) * yy
        if rng.random() < 0.5:
            arr = np.sin(2 * math.pi * rng.uniform(0.5, 4) * arr)
    arr = arr - arr.min()
    arr = arr / max(arr.max(), 1e-9)
    if rng.random() < 0.3:
        arr = (arr > rng.uniform(0.4, 0.7)).astype(np.float32)
    arr = arr ** rng.uniform(0.6, 1.8)
    return Image.fromarray(np.uint8(np.clip(arr, 0, 1) * 255), "L")


def random_image(rng: random.Random, nrng: np.random.Generator) -> np.ndarray:
    """Одно случайное 8-битное изображение 128×128."""
    p = rng.random()
    if p < 0.38:
        img = draw_text(rng)
    elif p < 0.68:
        img = draw_shapes(rng)
    elif p < 0.80:
        img = draw_points(rng)
    elif p < 0.90:
        img = draw_texture(rng, nrng)
    else:
        a = np.asarray(draw_text(rng), np.float32)
        b = np.asarray(draw_shapes(rng) if rng.random() < 0.7 else draw_texture(rng, nrng), np.float32)
        img = Image.fromarray(np.uint8(np.maximum(a, b * rng.uniform(0.3, 1.0))), "L")
    if rng.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 2.0)))
    arr = np.asarray(img, np.uint8)
    if rng.random() < 0.12:
        arr = 255 - arr  # негатив: тёмный объект на светлом фоне
    if rng.random() < 0.5:
        arr = np.ascontiguousarray(np.rot90(arr, rng.randint(0, 3)))
    if arr.max() == arr.min():  # пустое изображение — подменяем
        return random_image(rng, nrng)
    return arr


TEXTURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "textures")


def load_textures(split: str = "train", root: str = TEXTURE_DIR) -> np.ndarray:
    """Текстуры из data/textures (набор SBS Noise Texture Pack, CC0) → (K,128,128) uint8.

    split: "train" | "test" | "all" — по файлу data/textures/split.json.
    """
    import json

    meta = os.path.join(root, "split.json")
    if not os.path.exists(meta):  # нет PNG (например, на Kaggle) — берём упакованную копию из data/cache
        if split == "all":
            return np.concatenate([load_cache("textures128", "train"), load_cache("textures128", "test")])
        arr = load_cache("textures128", split)
        return arr if arr.ndim == 3 and arr.shape[1] == SIZE else np.zeros((0, SIZE, SIZE), np.uint8)
    sp = json.load(open(meta))
    names = sp["train"] + sp["test"] if split == "all" else sp[split]
    return np.stack([np.asarray(Image.open(os.path.join(root, n)).convert("L"), np.uint8) for n in names], 0)


def texture_image(rng: random.Random, nrng: np.random.Generator, textures: np.ndarray) -> np.ndarray:
    """Случайная текстура с аугментацией: масштаб/кроп, повороты, гамма, контраст, негатив, маска."""
    t = textures[rng.randrange(len(textures))]
    img = Image.fromarray(t, "L")
    zoom = rng.choice([1.0, 1.0, rng.uniform(1.0, 2.5), rng.uniform(0.5, 1.0)])
    if zoom > 1.0:  # увеличение: кроп части текстуры
        c = int(round(SIZE / zoom))
        x0, y0 = rng.randint(0, SIZE - c), rng.randint(0, SIZE - c)
        img = img.crop((x0, y0, x0 + c, y0 + c)).resize((SIZE, SIZE), Image.BICUBIC)
    elif zoom < 1.0:  # уменьшение: мозаика 2×2 (текстуры бесшовные) и сжатие
        big = Image.new("L", (2 * SIZE, 2 * SIZE))
        for dx in (0, SIZE):
            for dy in (0, SIZE):
                big.paste(img, (dx, dy))
        c = int(round(SIZE / zoom))
        x0, y0 = rng.randint(0, 2 * SIZE - c), rng.randint(0, 2 * SIZE - c)
        img = big.crop((x0, y0, x0 + c, y0 + c)).resize((SIZE, SIZE), Image.LANCZOS)
    arr = np.asarray(img, np.float32) / 255.0
    if rng.random() < 0.5:
        arr = np.rot90(arr, rng.randint(1, 3))
    if rng.random() < 0.5:
        arr = arr[:, ::-1]
    if rng.random() < 0.25:
        arr = 1.0 - arr
    lo, hi = np.percentile(arr, [rng.uniform(0, 5), rng.uniform(95, 100)])
    if rng.random() < 0.3 and hi > lo:
        arr = np.clip((arr - lo) / (hi - lo), 0, 1)  # растяжение контраста
    arr = np.clip(arr, 0, 1) ** float(np.exp(rng.uniform(-0.5, 0.5)))
    if rng.random() < 0.15:  # текстура внутри фигуры/символа на тёмном фоне
        mask = np.asarray(draw_text(rng) if rng.random() < 0.5 else draw_shapes(rng), np.float32) / 255.0
        arr = arr * (mask > 0.2)
    out = np.ascontiguousarray(np.uint8(np.round(arr * 255)))
    if out.max() == out.min():
        return texture_image(rng, nrng, textures)
    return out


CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache")


def load_cache(name: str, split: str = "train", root: str = CACHE_DIR) -> np.ndarray:
    """Дополнительные данные из data/cache (см. scripts/prepare_data.py): textures512 | cifar_gray."""
    path = os.path.join(root, f"{name}.npz")
    if not os.path.exists(path):
        return np.zeros((0, 1, 1), np.uint8)
    with np.load(path, allow_pickle=False) as z:
        return z[split]


def texture512_image(rng: random.Random, tex: np.ndarray) -> np.ndarray:
    """Фрагмент бесшовной текстуры 512×512 в случайном масштабе (1×, 2× или 4× к версии 128)."""
    t = tex[rng.randrange(len(tex))]
    crop = rng.choice([128, 128, 192, 256, 384, 512])
    y, x = rng.randint(0, 512 - crop), rng.randint(0, 512 - crop)
    patch = t[y:y + crop, x:x + crop]
    img = Image.fromarray(np.ascontiguousarray(patch), "L")
    if crop != SIZE:
        img = img.resize((SIZE, SIZE), Image.LANCZOS if crop > SIZE else Image.BICUBIC)
    return _photometric(rng, np.asarray(img, np.float32) / 255.0)


def cifar_image(rng: random.Random, cifar: np.ndarray) -> np.ndarray:
    """Изображение CIFAR-10 (32×32, оттенки серого), увеличенное до 128×128 или вставленное в кадр."""
    t = Image.fromarray(cifar[rng.randrange(len(cifar))], "L")
    if rng.random() < 0.7:
        img = t.resize((SIZE, SIZE), Image.BICUBIC)
    else:
        s = rng.randint(48, 112)
        img = Image.new("L", (SIZE, SIZE), 0)
        img.paste(t.resize((s, s), Image.BICUBIC), (rng.randint(0, SIZE - s), rng.randint(0, SIZE - s)))
    return _photometric(rng, np.asarray(img, np.float32) / 255.0, invert_p=0.1)


def _photometric(rng: random.Random, arr: np.ndarray, invert_p: float = 0.25) -> np.ndarray:
    if rng.random() < 0.5:
        arr = np.rot90(arr, rng.randint(1, 3))
    if rng.random() < 0.5:
        arr = arr[:, ::-1]
    if rng.random() < invert_p:
        arr = 1.0 - arr
    if rng.random() < 0.3:
        lo, hi = np.percentile(arr, [rng.uniform(0, 5), rng.uniform(95, 100)])
        if hi > lo:
            arr = (arr - lo) / (hi - lo)
    arr = np.clip(arr, 0, 1) ** float(np.exp(rng.uniform(-0.4, 0.4)))
    out = np.ascontiguousarray(np.uint8(np.round(arr * 255)))
    if out.max() == out.min():
        out = np.ascontiguousarray(np.uint8(np.round(np.clip(arr + rng.random() * 0.01, 0, 1) * 255)))
        out[0, 0] = 255 - out[0, 0]
    return out


# Доли источников по умолчанию: синтетика / текстуры 128 / текстуры 512 / CIFAR-10
DEFAULT_MIX = (0.35, 0.15, 0.30, 0.20)

# Реальные изображения из внешних датасетов (DTD, Imagenette, BSDS500, …): список массивов uint8
# (оттенки серого, короткая сторона ~192 px). Заполняется до создания пула процессов (fork наследует).
_PHOTOS: list = []
_REALTEX: list = []


def collect_images(root: str, include: tuple[str, ...], exclude: tuple[str, ...] = (), limit: int = 0,
                   seed: int = 0) -> list[str]:
    """Пути к jpg/png под root, в полном пути которых есть одна из подстрок include и нет exclude."""
    exts = (".jpg", ".jpeg", ".png")
    out = []
    for d, dirs, files in os.walk(root, followlinks=True):
        # большие деревья символов (kkanji, CASIA) не обходим, если их не просили явно
        dirs[:] = [x for x in dirs if not any(t in x.lower() for t in _GLYPH_TREES)
                   or any(t in x.lower() for t in include)]
        low = d.lower()
        if not any(t in low for t in include) or any(t in low for t in exclude):
            continue
        out += [os.path.join(d, f) for f in files if f.lower().endswith(exts)]
    out.sort()
    if limit and len(out) > limit:
        random.Random(seed).shuffle(out)
        out = out[:limit]
    return out


def _load_small(path: str, short: int = 192) -> np.ndarray | None:
    try:
        im = Image.open(path).convert("L")
    except Exception:
        return None
    w, h = im.size
    k = short / min(w, h)
    if k < 1:
        im = im.resize((max(SIZE, int(w * k)), max(SIZE, int(h * k))), Image.LANCZOS)
    a = np.asarray(im, np.uint8)
    return a if min(a.shape) >= SIZE else None


def load_real_images(photo_paths: list[str], tex_paths: list[str], workers: int = 1) -> tuple[int, int]:
    """Загружает реальные фото и текстуры в _PHOTOS/_REALTEX (до создания банка)."""
    global _PHOTOS, _REALTEX
    if workers > 1:
        from multiprocessing import Pool

        with Pool(workers) as pool:
            ph = pool.map(_load_small, photo_paths, chunksize=64)
            tx = pool.map(_load_small, tex_paths, chunksize=64)
    else:
        ph = [_load_small(p) for p in photo_paths]
        tx = [_load_small(p) for p in tex_paths]
    _PHOTOS = [a for a in ph if a is not None]
    _REALTEX = [a for a in tx if a is not None]
    return len(_PHOTOS), len(_REALTEX)


def real_image(rng: random.Random, pool: list) -> np.ndarray:
    """Случайный фрагмент реального изображения (масштаб 1…1,5), с аугментациями."""
    a = pool[rng.randrange(len(pool))]
    h, w = a.shape
    c = rng.randint(SIZE, min(h, w, int(SIZE * 1.5)))
    y, x = rng.randint(0, h - c), rng.randint(0, w - c)
    img = Image.fromarray(np.ascontiguousarray(a[y:y + c, x:x + c]), "L")
    if c != SIZE:
        img = img.resize((SIZE, SIZE), Image.LANCZOS)
    return _photometric(rng, np.asarray(img, np.float32) / 255.0, invert_p=0.08)


# Символы и силуэты (EMNIST, Fashion-MNIST, Kuzushiji-49, Kuzushiji-Kanji, CASIA-HWDB): яркий объект на тёмном
# фоне, мелкие массивы uint8. Заполняются до создания пула процессов, как и _PHOTOS.
_GLYPHS: list = []
_GLYPH_TREES = ("kkanji", "casia-hwdb", "hwdb")
_SKIP_TREES = ("lfw", "imagenette", "dtd", "bsds", "bsr")


def _read_idx(path: str) -> np.ndarray | None:
    """Изображения в формате IDX (MNIST): 16 байт заголовка, затем N×H×W uint8."""
    with open(path, "rb") as f:
        head = f.read(16)
        if len(head) < 16 or head[2] != 8 or head[3] != 3:
            return None
        n, h, w = (int.from_bytes(head[4 + 4 * i: 8 + 4 * i], "big") for i in range(3))
        a = np.frombuffer(f.read(n * h * w), np.uint8)
    a = a[: (a.size // (h * w)) * h * w].reshape(-1, h, w)
    return a.transpose(0, 2, 1) if "emnist" in os.path.basename(path).lower() else a


def find_glyph_sources(root: str) -> tuple[list[str], list[str]]:
    """Файлы IDX/NPZ символьных датасетов и корни PNG-деревьев (kkanji, CASIA) под root."""
    files, trees = [], []
    for d, dirs, names in os.walk(root, followlinks=True):
        if any(t in os.path.basename(d).lower() for t in _GLYPH_TREES):
            trees.append(d)
            dirs[:] = []
            continue
        dirs[:] = [x for x in dirs if not any(t in x.lower() for t in _SKIP_TREES)]
        for f in names:
            fl = f.lower()
            if fl.endswith("images-idx3-ubyte") and ("emnist" not in fl or "emnist-balanced" in fl):
                files.append(os.path.join(d, f))
            elif fl.endswith("-imgs.npz"):
                files.append(os.path.join(d, f))
    return sorted(set(files)), sorted(set(trees))


def _is_test(path: str) -> bool:
    name = os.path.basename(path).lower()
    return "test" in name or "t10k" in name


def _sample_tree(root: str, limit: int, rng: random.Random, per_dir: int = 3) -> list[str]:
    """До limit PNG из дерева «класс/файл»: обход в случайном порядке, не больше per_dir из папки."""
    out = []
    for d, dirs, names in os.walk(root, followlinks=True):
        rng.shuffle(dirs)
        pngs = [n for n in names if n.lower().endswith(".png")]
        rng.shuffle(pngs)
        out += [os.path.join(d, n) for n in pngs[:per_dir]]
        if len(out) >= limit:
            break
    return out


def _bright_on_dark(a: np.ndarray) -> np.ndarray | None:
    if a is None or a.size == 0 or a.max() == a.min():
        return None
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]).astype(np.float32).mean()
    return np.ascontiguousarray(255 - a if border > 127 else a)


def _load_glyph_png(path: str) -> np.ndarray | None:
    try:
        a = np.asarray(Image.open(path).convert("L"), np.uint8)
    except Exception:
        return None
    return _bright_on_dark(a)


def load_glyphs(roots: list[str], per_source: int = 40000, seed: int = 0,
                workers: int = 1) -> tuple[int, list]:
    """Заполняет _GLYPHS обучающими символами; возвращает (их число, отложенные символы для контроля).

    Обучающая часть — train-файлы IDX/NPZ и 95 % PNG; контроль — test-файлы и остальные 5 % PNG.
    """
    global _GLYPHS
    rng = random.Random(seed)
    train, test = [], []
    for r in roots:
        files, trees = find_glyph_sources(r)
        for f in files:
            try:
                a = _read_idx(f) if "idx3" in f.lower() else np.load(f, allow_pickle=False)["arr_0"]
            except Exception:
                continue
            if a is None or a.ndim != 3:
                continue
            idx = rng.sample(range(len(a)), min(len(a), per_source))
            items = [np.ascontiguousarray(a[i]) for i in idx]
            (test if _is_test(f) else train).extend(g for g in items if g.max() > g.min())
            print(f"символы: {os.path.basename(f)} → {len(items)}", flush=True)
        for t in trees:
            paths = _sample_tree(t, per_source // 2, rng)
            if workers > 1:
                from multiprocessing import Pool

                with Pool(workers) as pool:
                    arrs = pool.map(_load_glyph_png, paths, chunksize=64)
            else:
                arrs = [_load_glyph_png(p) for p in paths]
            for i, a in enumerate(arrs):
                if a is not None:
                    (test if i % 20 == 0 else train).append(a)
            print(f"символы: {t} → {len(paths)}", flush=True)
    _GLYPHS = train
    return len(train), test


def _glyph_patch(rng: random.Random, g: np.ndarray, size: int) -> np.ndarray:
    """Символ в квадрате size×size с сохранением пропорций."""
    h, w = g.shape
    k = size / max(h, w)
    nh, nw = max(1, int(round(h * k))), max(1, int(round(w * k)))
    im = Image.fromarray(g, "L").resize((nw, nh), Image.BICUBIC if k > 1 else Image.LANCZOS)
    out = np.zeros((size, size), np.float32)
    y, x = (size - nh) // 2, (size - nw) // 2
    out[y:y + nh, x:x + nw] = np.asarray(im, np.float32) / 255.0
    return out


def glyph_image(rng: random.Random, pool: list) -> np.ndarray:
    """Сцена 128×128 из символов: один крупный, несколько разбросанных или строки «текста»."""
    canvas = np.zeros((SIZE, SIZE), np.float32)
    r = rng.random()
    if r < 0.4:
        s = rng.randint(64, 124)
        p = _glyph_patch(rng, pool[rng.randrange(len(pool))], s)
        y, x = rng.randint(0, SIZE - s), rng.randint(0, SIZE - s)
        canvas[y:y + s, x:x + s] = p
    elif r < 0.75:
        for _ in range(rng.randint(2, 7)):
            s = rng.randint(24, 72)
            p = _glyph_patch(rng, pool[rng.randrange(len(pool))], s)
            y, x = rng.randint(0, SIZE - s), rng.randint(0, SIZE - s)
            canvas[y:y + s, x:x + s] = np.maximum(canvas[y:y + s, x:x + s], p)
    else:
        rows = rng.randint(1, 3)
        s = rng.randint(22, min(44, SIZE // rows))
        cols = rng.randint(2, max(2, SIZE // s))
        y0 = rng.randint(0, SIZE - rows * s)
        for i in range(rows):
            x0 = rng.randint(0, SIZE - cols * s)
            for j in range(cols):
                p = _glyph_patch(rng, pool[rng.randrange(len(pool))], s)
                y, x = y0 + i * s, x0 + j * s
                canvas[y:y + s, x:x + s] = np.maximum(canvas[y:y + s, x:x + s], p)
    if rng.random() < 0.3:  # резкие края вместо размытого увеличения 28 → 128
        canvas = np.clip((canvas - 0.3) / 0.4, 0, 1)
    if canvas.max() <= 0:
        canvas[rng.randrange(SIZE), rng.randrange(SIZE)] = 1.0
    return _photometric(rng, canvas, invert_p=0.1)


def build_bank(count: int, seed: int = 0, workers: int = 0, texture_frac: float = 0.0,
               texture_split: str = "train", mix: tuple[float, ...] | None = None) -> np.ndarray:
    """Банк изображений (count,128,128) uint8; из него собираются четвёрки.

    mix — доли источников (синтетика, текстуры 128, фрагменты текстур 512, CIFAR-10);
    если не задан, используется texture_frac (только синтетика + текстуры 128).
    Отсутствующие источники (нет data/cache) автоматически заменяются синтетикой.
    """
    if mix is None:
        mix = (1.0 - texture_frac, texture_frac, 0.0, 0.0)
    if workers and workers > 1:
        from multiprocessing import Pool

        chunks = [(c, seed * 1000003 + i, tuple(mix), texture_split)
                  for i, c in enumerate(_split(count, workers * 4))]
        with Pool(workers) as pool:
            parts = pool.starmap(_bank_chunk, chunks)
        return np.concatenate(parts, 0)
    return _bank_chunk(count, seed, tuple(mix), texture_split)


def _split(total: int, parts: int) -> list[int]:
    base, rem = divmod(total, parts)
    return [base + (1 if i < rem else 0) for i in range(parts) if base + (1 if i < rem else 0) > 0]


def _bank_chunk(count: int, seed: int, mix: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0),
                texture_split: str = "train") -> np.ndarray:
    """mix: (синтетика, текстуры 128, текстуры 512, CIFAR-10[, реальные фото, реальные текстуры, символы])."""
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    mix = tuple(mix) + (0.0,) * (7 - len(mix))
    tex = load_textures(texture_split) if mix[1] > 0 else np.zeros((0,))
    tex512 = load_cache("textures512", texture_split) if mix[2] > 0 else np.zeros((0,))
    cifar = load_cache("cifar_gray", texture_split) if mix[3] > 0 else np.zeros((0,))
    avail = [mix[0], mix[1] if len(tex) else 0.0, mix[2] if len(tex512) else 0.0, mix[3] if len(cifar) else 0.0,
             mix[4] if _PHOTOS else 0.0, mix[5] if _REALTEX else 0.0, mix[6] if _GLYPHS else 0.0]
    avail[0] += sum(mix) - sum(avail)  # недоступные источники → синтетика
    cum = np.cumsum(avail) / max(sum(avail), 1e-9)
    out = []
    for _ in range(count):
        r = rng.random()
        if r < cum[0]:
            out.append(random_image(rng, nrng))
        elif r < cum[1]:
            out.append(texture_image(rng, nrng, tex))
        elif r < cum[2]:
            out.append(texture512_image(rng, tex512))
        elif r < cum[3]:
            out.append(cifar_image(rng, cifar))
        elif r < cum[4]:
            out.append(real_image(rng, _PHOTOS))
        elif r < cum[5]:
            out.append(real_image(rng, _REALTEX))
        else:
            out.append(glyph_image(rng, _GLYPHS))
    return np.stack(out, 0)
