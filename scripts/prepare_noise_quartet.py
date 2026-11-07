"""Prepare four preselected texture images as strict hackathon PNG inputs."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image


# First numbered variant in four distinct texture families. Selected before synthesis.
SELECTION = (
    "128x128/Cracks/Cracks 1 - 128x128.png",
    "128x128/Gabor/Gabor 1 - 128x128.png",
    "128x128/Perlin/Perlin 1 - 128x128.png",
    "128x128/Voronoi/Voronoi 1 - 128x128.png",
)
DISTANCES_MM = (24.1, 25.3, 26.5, 27.7)


def prepare(archive, output):
    archive, output = Path(archive), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "originals").mkdir()
    records = []
    with ZipFile(archive) as zipped:
        license_bytes = zipped.read("License.txt")
        (output / "License.txt").write_bytes(license_bytes)
        for i, (name, distance) in enumerate(zip(SELECTION, DISTANCES_MM), start=1):
            raw = zipped.read(name)
            with Image.open(io.BytesIO(raw)) as source:
                if source.format != "PNG" or source.size != (128, 128):
                    raise ValueError(f"Invalid source image: {name}")
                original_mode = source.mode
                rgba = source.convert("RGBA")
                alpha = np.asarray(rgba)[..., 3]
                background = Image.new("RGBA", source.size, (0, 0, 0, 255))
                grayscale = Image.alpha_composite(background, rgba).convert("L")
            filename = f"plane_{i}.png"
            (output / "originals" / filename).write_bytes(raw)
            grayscale.save(output / filename)
            array = np.asarray(grayscale)
            records.append({"plane": i, "distance_mm": distance, "archive_member": name,
                            "source_mode": original_mode, "source_alpha_min": int(alpha.min()),
                            "source_alpha_max": int(alpha.max()), "output": filename,
                            "size": list(grayscale.size), "mode": grayscale.mode,
                            "pixel_min": int(array.min()), "pixel_max": int(array.max()),
                            "pixel_std": float(array.std()),
                            "source_sha256": hashlib.sha256(raw).hexdigest(),
                            "output_sha256": hashlib.sha256((output / filename).read_bytes()).hexdigest()})
    manifest = {
        "source_archive": archive.name, "source_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "selection": "first numbered variant from Cracks, Gabor, Perlin, Voronoi; chosen before inference, not selected by achieved metrics",
        "conversion": "Pillow alpha composite on black, then convert('L'); no resize, contrast stretch, denoising or histogram adjustment",
        "phase_plane_order": "increasing distance: 24.1, 25.3, 26.5, 27.7 mm",
        "license": "License.txt supplied in archive states CC0/Public Domain, Screaming Brain Studios",
        "images": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare(args.archive, args.output)


if __name__ == "__main__":
    main()
