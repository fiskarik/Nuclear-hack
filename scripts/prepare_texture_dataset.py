"""Build family-disjoint texture quartets from the supplied 128x128 SBS ZIP."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
from PIL import Image

RESERVED = {"Cracks", "Gabor", "Perlin", "Voronoi"}
VALIDATION = {"Super Perlin", "Turbulence"}
TEST = {"Streak", "Vein"}


def prepare(archive, output, train=256, validation=32, test=32, seed=4187):
    archive, output = Path(archive), Path(output)
    if min(train, validation, test) < 1:
        raise ValueError("all splits must be nonempty")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    pools = {split: [] for split in ("train", "validation", "test", "reserved")}
    catalog = []
    with ZipFile(archive) as zipped:
        (output / "License.txt").write_bytes(zipped.read("License.txt"))
        for name in sorted(zipped.namelist()):
            if not name.lower().endswith(".png"):
                continue
            member = PurePosixPath(name)
            if member.is_absolute() or ".." in member.parts or len(member.parts) != 3:
                raise ValueError(f"unexpected archive path: {name}")
            family = member.parts[1]
            split = "reserved" if family in RESERVED else "validation" if family in VALIDATION else "test" if family in TEST else "train"
            raw = zipped.read(name)
            with Image.open(io.BytesIO(raw)) as im:
                if im.format != "PNG" or im.size != (128, 128):
                    raise ValueError(f"expected 128x128 PNG: {name}")
                rgba = im.convert("RGBA")
                grayscale = Image.alpha_composite(Image.new("RGBA", im.size, (0, 0, 0, 255)), rgba).convert("L")
            a = np.array(grayscale)
            if a.max() == 0 or a.std() < 1e-6:
                raise ValueError(f"blank or constant texture: {name}")
            pixel_hash = hashlib.sha256(a.tobytes()).hexdigest()
            relative = Path("pool") / family.replace(" ", "_") / member.name
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            grayscale.save(target)
            record = {"member": name, "family": family, "split": split, "path": str(relative),
                      "source_sha256": hashlib.sha256(raw).hexdigest(), "pixel_sha256": pixel_hash,
                      "png_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
            catalog.append(record)
            pools[split].append((a, record))
    # Exact duplicates may share a split, but cannot leak across split boundaries.
    assignment = {}
    for item in catalog:
        previous = assignment.setdefault(item["pixel_sha256"], item["split"])
        if previous != item["split"]:
            raise ValueError(f"duplicate pixels cross splits: {item['member']}")
    for split in ("train", "validation", "test"):
        if len(pools[split]) < 4:
            raise ValueError(f"need >=4 textures in {split}")
    manifest = {"source": "user-provided SBS noise textures; not the inaccessible full OpenGameArt download",
                "source_archive": archive.name, "source_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "seed": seed, "size": 128, "mode": "L", "image_counts": {k: len(v) for k, v in pools.items()},
                "families": {k: sorted({r[1]["family"] for r in v}) for k, v in pools.items()},
                "split_policy": "disjoint texture families, plus cross-split exact grayscale-pixel duplicate check; reserved families never used for model selection",
                "conversion": "alpha composite on black, then Pillow L; no rescaling or contrast stretch",
                "train_augmentation": "random 90-degree rotations, horizontal reflection and toroidal translations; all transforms recorded",
                "splits": {}}
    for split_id, (split, count) in enumerate((("train", train), ("validation", validation), ("test", test))):
        rng = np.random.default_rng(seed + split_id * 100_000)
        samples = []
        for index in range(count):
            folder = output / split / f"sample_{index:04d}"
            folder.mkdir(parents=True)
            images, hashes = [], []
            for plane, source_index in enumerate(rng.choice(len(pools[split]), size=4, replace=False), start=1):
                array, record = pools[split][int(source_index)]
                rotations = int(rng.integers(4)) if split == "train" else 0
                reflection = bool(rng.integers(2)) if split == "train" else False
                shift = tuple(int(x) for x in rng.integers(0, 128, 2)) if split == "train" else (0, 0)
                augmented = np.rot90(array, rotations)
                if reflection:
                    augmented = augmented[:, ::-1]
                augmented = np.roll(augmented, shift, axis=(0, 1))
                path = folder / f"plane_{plane}.png"
                Image.fromarray(augmented.copy()).save(path)
                images.append({"member": record["member"], "family": record["family"],
                               "source_pixel_sha256": record["pixel_sha256"],
                               "rotation_90": rotations, "horizontal_reflection": reflection, "roll_yx": shift})
                hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
            samples.append({"path": str(folder.relative_to(output)), "images": images, "sha256": hashes})
        manifest["splits"][split] = samples
    (output / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"image_counts": manifest["image_counts"], "families": manifest["families"],
                      "quartets": {k: len(v) for k, v in manifest["splits"].items()}}, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train", type=int, default=256)
    parser.add_argument("--validation", type=int, default=32)
    parser.add_argument("--test", type=int, default=32)
    parser.add_argument("--seed", type=int, default=4187)
    prepare(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
