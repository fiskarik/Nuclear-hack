"""Подготовка дополнительных обучающих данных (кэш в data/cache/, в git не хранится).

Источники:
  1. SBS Noise Texture Pack (Screaming Brain Studios, CC0), https://opengameart.org/content/700-noise-textures
     — 262 бесшовные текстуры в разрешениях 128/256/512. Версия 128×128 лежит в
     репозитории (data/textures); здесь из версии 512×512 берутся фрагменты
     разного масштаба. Разбиение train/test то же, что в data/textures/split.json
     (номера 3 и 7 каждой категории — только для контроля).
  2. CIFAR-10 (Krizhevsky, 2009) в оттенках серого — как в
     github.com/Jemuna/Image-Classification-Using-Deep-Learning-in-TensorFlow.
     Берётся с Kaggle (датасет fedesoriano/cifar10-python-in-csv, 10 000 изображений)
     через Kaggle API в виде CSV — без pickle.

    python scripts/prepare_data.py --tex-zip sbs_-_noise_texture_pack_-_512x512.zip \
        --cifar-csv test.csv.zip
Без аргументов скачивает всё сам (нужен доступ к opengameart.org и Kaggle API).
"""
from __future__ import annotations

import argparse
import io
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
TEX_URL = "https://opengameart.org/sites/default/files/sbs_-_noise_texture_pack_-_512x512.zip"
HELD_OUT = (3, 7)


def textures_from_zip(path: Path, size: int = 512):
    """→ (train array, test array, train names, test names); имена вида 'cracks_01'."""
    data = {"train": [], "test": []}
    names = {"train": [], "test": []}
    with zipfile.ZipFile(path) as z:
        files = [n for n in z.namelist() if n.lower().endswith(".png")]
        for n in sorted(files):
            m = re.search(r"/([^/]+?) (\d+) - \d+x\d+\.png$", n)
            if not m:
                continue
            cat, k = m.group(1).lower().replace(" ", "_"), int(m.group(2))
            im = Image.open(io.BytesIO(z.read(n))).convert("RGB").convert("L")
            if im.size != (size, size):
                im = im.resize((size, size), Image.LANCZOS)
            part = "test" if k in HELD_OUT else "train"
            data[part].append(np.asarray(im, np.uint8))
            names[part].append(f"{cat}_{k:02d}")
    return np.stack(data["train"]), np.stack(data["test"]), names["train"], names["test"]


def cifar_gray_csv(csv_zip: Path, held_out: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """CIFAR-10 из CSV (pixel_0…pixel_3071 в порядке R-, G-, B-плоскостей, + label).

    CSV разбирается как числа — без pickle, поэтому сторонний файл не может
    выполнить код. Перевод в оттенки серого: 0.299R + 0.587G + 0.114B.
    Последние held_out изображений — контрольные.
    """
    with zipfile.ZipFile(csv_zip) as z:
        raw = z.read(z.namelist()[0]).decode()
    lines = raw.splitlines()
    header = lines[0].split(",")
    pix = [i for i, h in enumerate(header) if h.startswith("pixel_")]
    assert len(pix) == 3072, f"ожидалось 3072 пикселя, найдено {len(pix)}"
    arr = np.loadtxt(io.StringIO("\n".join(lines[1:])), delimiter=",", dtype=np.float32)
    x = arr[:, pix].reshape(-1, 3, 32, 32)
    g = np.clip(np.round(0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]), 0, 255).astype(np.uint8)
    return g[:-held_out], g[-held_out:]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tex-zip", help="zip с текстурами 512×512 (по умолчанию скачать)")
    ap.add_argument("--cifar-csv", help="test.csv.zip из Kaggle-датасета fedesoriano/cifar10-python-in-csv "
                                        "(по умолчанию скачать через Kaggle API)")
    ap.add_argument("--out", default=str(CACHE))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp())

    # упакованная копия data/textures (128×128) — для запуска на Kaggle без PNG-файлов
    import json

    sp = json.load(open(ROOT / "data" / "textures" / "split.json"))
    t128 = {k: np.stack([np.asarray(Image.open(ROOT / "data" / "textures" / n).convert("L")) for n in sp[k]])
            for k in ("train", "test")}
    np.savez_compressed(out / "textures128.npz", **t128)
    print(f"textures128: train {t128['train'].shape}, test {t128['test'].shape}")

    tex_zip = Path(a.tex_zip) if a.tex_zip else tmp / "tex512.zip"
    if not a.tex_zip:
        print("скачиваю", TEX_URL)
        req = urllib.request.Request(TEX_URL, headers={"User-Agent": "Mozilla/5.0"})
        tex_zip.write_bytes(urllib.request.urlopen(req, timeout=600).read())
    tr, te, ntr, nte = textures_from_zip(tex_zip)
    np.savez_compressed(out / "textures512.npz", train=tr, test=te, train_names=ntr, test_names=nte)
    print(f"textures512: train {tr.shape}, test {te.shape}")

    if a.cifar_csv:
        csv_zip = Path(a.cifar_csv)
    else:
        cli = os.environ.get("KAGGLE_CLI", "kaggle").split()
        subprocess.run(cli + ["datasets", "download", "fedesoriano/cifar10-python-in-csv", "-f", "test.csv",
                              "-p", str(tmp)], check=True)
        csv_zip = tmp / "test.csv.zip"
    ctr, cte = cifar_gray_csv(csv_zip)
    np.savez_compressed(out / "cifar_gray.npz", train=ctr, test=cte)
    print(f"cifar_gray: train {ctr.shape}, test {cte.shape}")


if __name__ == "__main__":
    sys.exit(main())
