import json
import math

import numpy as np
from PIL import Image
import pytest
import torch

from doe.data import generate_dataset, load_quartet, QuartetDataset
from doe.model import FresnelNet
from doe.optics import Fresnel
from doe.results import amplitude_png, decode_phase, encode_phase, export_result, metrics


def test_phase_endpoints_and_quantization_error():
    phi = np.linspace(-math.pi, math.pi, 1001)
    encoded = encode_phase(phi)
    assert encoded.dtype == np.uint8
    assert encoded[0] == 0 and encoded[-1] == 255
    np.testing.assert_allclose(decode_phase(np.array([0, 255], dtype=np.uint8)), [-math.pi, math.pi])
    assert np.max(np.abs(decode_phase(encoded) - phi)) <= math.pi / 255 + 1e-12
    with pytest.raises(ValueError):
        encode_phase(np.array([math.nan]))


def test_pearson_ssim_known_cases():
    a = np.arange(128 * 128).reshape(128, 128).astype(np.uint8)
    assert metrics(a, a) == {"pearson": 1.0, "ssim": 1.0}
    assert metrics(a, 255 - a)["pearson"] == pytest.approx(-1.0)
    assert metrics(np.zeros_like(a), a)["pearson"] is None
    assert metrics(np.zeros_like(a), np.zeros_like(a))["ssim"] == 1.0


def test_export_reconstructs_saved_quantized_doe_and_metrics(tmp_path):
    generate_dataset(tmp_path / "data", train=1, validation=1, test=1)
    inputs = [tmp_path / "data/test/sample_0000" / f"plane_{i}.png" for i in range(1, 5)]
    optics = Fresnel()
    torch.manual_seed(11)
    phase = torch.rand(1, 128, 128) * 2 * math.pi - math.pi
    out = tmp_path / "result"
    report = export_result(out, optics, phase, inputs)
    with Image.open(out / "doe_phase.png") as image:
        assert image.mode == "L" and image.size == (128, 128)
        decoded = decode_phase(np.array(image))
    expected = np.abs(optics.physical(torch.tensor(decoded, dtype=torch.float32)[None])[0].numpy())
    np.testing.assert_allclose(np.load(out / "reconstructed_amplitudes.npy"), expected)
    for i in range(4):
        with Image.open(out / f"reconstruction_{i + 1}.png") as image:
            assert image.mode == "L" and image.size == (128, 128)
            png = np.array(image)
        np.testing.assert_array_equal(png, amplitude_png(expected[i])[0])
        target = np.array(Image.open(inputs[i]))
        assert report["planes"][i]["ssim"] == metrics(target, png)["ssim"]
    assert len(report["planes"]) == 4
    json.loads((out / "metrics.json").read_text())
    with pytest.raises(FileExistsError):
        export_result(out, optics, phase, inputs)


def test_strict_inputs_and_separate_splits(tmp_path):
    manifest = generate_dataset(tmp_path / "data", train=2, validation=2, test=2)
    seeds = [s for split in manifest["splits"].values() for row in split for s in row["image_seeds"]]
    assert len(seeds) == len(set(seeds))
    assert QuartetDataset(tmp_path / "data", "train")[0].shape == (4, 128, 128)
    paths = [tmp_path / "data/test/sample_0000" / f"plane_{i}.png" for i in range(1, 5)]
    Image.new("RGB", (128, 128)).save(tmp_path / "rgb.png")
    with pytest.raises(ValueError, match="grayscale"):
        load_quartet([tmp_path / "rgb.png", *paths[1:]])
    with pytest.raises(ValueError, match="exactly four"):
        load_quartet(paths[:3])
    with pytest.raises(FileExistsError):
        generate_dataset(tmp_path / "data", train=1, validation=1, test=1)
