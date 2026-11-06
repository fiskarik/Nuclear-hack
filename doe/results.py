"""Quantized deliverables and independently computable image quality metrics."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
import torch

from .data import load_quartet


def encode_phase(phase):
    phase = np.asarray(phase, dtype=np.float64)
    if not np.isfinite(phase).all() or (np.abs(phase) > math.pi + 1e-6).any():
        raise ValueError("phase must be finite and in [-pi, pi]")
    return np.rint((np.clip(phase, -math.pi, math.pi) + math.pi) * (255 / (2 * math.pi))).astype(np.uint8)


def decode_phase(image):
    image = np.asarray(image)
    if image.dtype != np.uint8:
        raise ValueError("phase PNG samples must be uint8")
    return image.astype(np.float64) * (2 * math.pi / 255) - math.pi


def amplitude_png(amplitude):
    amplitude = np.asarray(amplitude, dtype=np.float64)
    if not np.isfinite(amplitude).all() or (amplitude < 0).any():
        raise ValueError("amplitude must be finite and nonnegative")
    peak = float(amplitude.max())
    return np.rint(255 * amplitude / max(peak, 1e-30)).astype(np.uint8), peak


def metrics(target, reconstructed):
    """Metrics of actual 8-bit images. No fitting, denoising or target-dependent gain."""
    a = np.asarray(target, dtype=np.float64) / 255
    b = np.asarray(reconstructed, dtype=np.float64) / 255
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("metrics require two equally-sized grayscale images")
    if min(a.shape) < 7:
        raise ValueError("SSIM requires dimensions >= 7")
    ac, bc = a - a.mean(), b - b.mean()
    denominator = float(np.linalg.norm(ac) * np.linalg.norm(bc))
    correlation = float(np.clip(np.sum(ac * bc) / denominator, -1, 1)) if denominator > 1e-14 else None
    return {"pearson": correlation, "ssim": float(structural_similarity(a, b, data_range=1.0, win_size=7))}


def phase_metrics(optics, phase, targets):
    """Evaluate after phase quantization and amplitude PNG quantization."""
    encoded = encode_phase(phase.detach().cpu().numpy())
    decoded = torch.as_tensor(decode_phase(encoded), device=phase.device, dtype=phase.dtype)
    with torch.no_grad():
        amplitudes = np.abs(optics.physical(decoded).cpu().numpy())
    source = np.rint(targets.detach().cpu().numpy() * 255).astype(np.uint8)
    records = []
    for b in range(len(source)):
        planes = [metrics(source[b, i], amplitude_png(amplitudes[b, i])[0]) for i in range(4)]
        correlations = [v["pearson"] for v in planes if v["pearson"] is not None]
        records.append({"pearson": float(np.mean(correlations)) if correlations else None,
                        "ssim": float(np.mean([v["ssim"] for v in planes])), "planes": planes})
    return records


def export_result(output, optics, phase, image_paths, metadata=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing result: {output}")
    targets = load_quartet(image_paths)
    if phase.ndim != 3 or phase.shape[0] != 1:
        raise ValueError("export expects one phase map [1,128,128]")
    output.mkdir(parents=True, exist_ok=True)
    encoded = encode_phase(phase[0].detach().cpu().numpy())
    phase_path = output / "doe_phase.png"
    Image.fromarray(encoded).save(phase_path)
    # Intentionally read the saved file back: all reported reconstructions use its quantized phase.
    with Image.open(phase_path) as saved:
        decoded = decode_phase(np.array(saved))
    decoded_tensor = torch.as_tensor(decoded, device=phase.device, dtype=phase.dtype)[None]
    with torch.no_grad():
        physical = optics.physical(decoded_tensor)[0].cpu().numpy()
    amplitudes = np.abs(physical)
    np.save(output / "phase_rad.npy", decoded)
    np.save(output / "reconstructed_amplitudes.npy", amplitudes)
    np.save(output / "reconstructed_complex_fields.npy", physical)
    input_arrays = np.rint(targets[0].numpy() * 255).astype(np.uint8)
    rows = []
    images = []
    for i, z in enumerate(optics.config.distances_m):
        image, peak = amplitude_png(amplitudes[i])
        images.append(image)
        Image.fromarray(input_arrays[i]).save(output / f"target_{i + 1}.png")
        Image.fromarray(image).save(output / f"reconstruction_{i + 1}.png")
        rows.append({"plane": i + 1, "distance_mm": z * 1000,
                     "output_pixel_pitch_um": optics.config.output_pixel_pitches_m[i] * 1e6,
                     "amplitude_peak_before_display_scaling": peak,
                     **metrics(input_arrays[i], image)})
    report = {
        "optics": optics.config.to_dict(), "metadata": metadata or {}, "planes": rows,
        "mean_pearson": float(np.mean([row["pearson"] for row in rows if row["pearson"] is not None]))
                        if any(row["pearson"] is not None for row in rows) else None,
        "mean_ssim": float(np.mean([row["ssim"] for row in rows])),
        "phase_encoding": "uint8 0 -> -pi; 255 -> +pi; round to nearest",
        "reconstruction": "single-FFT Fresnel, reconstructed from the saved quantized DOE PNG",
        "png_scaling": "amplitude (not intensity); each plane divided by its own maximum; no background subtraction or fitted gain",
        "ssim": {"implementation": "skimage.metrics.structural_similarity", "win_size": 7, "data_range": 1.0},
        "pearson_constant_image": "null (undefined), never a fabricated zero or one",
        "doe_sha256": hashlib.sha256(phase_path.read_bytes()).hexdigest(),
        "inputs": [{"name": Path(path).name, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()} for path in image_paths],
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_result(output / "comparison.png", input_arrays, np.stack(images), encoded, rows)
    return report


def plot_result(path, targets, reconstructions, phase, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 3, figsize=(10, 11), constrained_layout=True)
    for i, row in enumerate(rows):
        axes[i, 0].imshow(targets[i], cmap="gray", vmin=0, vmax=255)
        axes[i, 0].set_title(f"Target {i+1}, z={row['distance_mm']:.1f} mm")
        axes[i, 1].imshow(reconstructions[i], cmap="gray", vmin=0, vmax=255)
        corr = row["pearson"]
        axes[i, 1].set_title(f"Reconstruction |U|; r={corr:.3f}" if corr is not None else "Reconstruction; r undefined")
        axes[i, 2].imshow(np.abs(targets[i].astype(float) - reconstructions[i]), cmap="magma", vmin=0, vmax=255)
        axes[i, 2].set_title(f"Absolute difference; SSIM={row['ssim']:.3f}")
        for ax in axes[i]:
            ax.axis("off")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
    display = ax.imshow(decode_phase(phase), cmap="twilight", vmin=-math.pi, vmax=math.pi)
    ax.set_title("Quantized DOE phase (radians)")
    fig.colorbar(display, ax=ax)
    fig.savefig(Path(path).with_name("phase_preview.png"), dpi=140)
    plt.close(fig)
