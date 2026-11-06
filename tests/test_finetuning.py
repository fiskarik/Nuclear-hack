"""Prevent silent training from scratch and source leakage during fine-tuning."""
import hashlib
import io
import json
from pathlib import Path
import runpy
from zipfile import ZipFile

import numpy as np
from PIL import Image
import pytest
import torch

from doe.data import generate_dataset
from doe.engine import TrainConfig, train
from doe.model import FresnelNet
from doe.optics import OpticsConfig


def test_finetuning_starts_from_the_supplied_weights(tmp_path):
    torch.manual_seed(13)
    model = FresnelNet(stages=1, width=4)
    with torch.no_grad():
        model.relax_logits.fill_(0.71)
        model.corrector[-1].weight.fill_(0.013)
    checkpoint = tmp_path / "parent.pt"
    torch.save({"format_version": 1, "model": model.state_dict(), "epoch": 7,
                "optics": model.optics.config.to_dict(), "architecture": {"stages": 1, "width": 4}}, checkpoint)
    generate_dataset(tmp_path / "data", train=1, validation=1, test=1)
    report = train(tmp_path / "data", tmp_path / "run", TrainConfig(
        epochs=1, batch_size=1, stages=1, width=4, learning_rate=0.0001,
        initial_checkpoint=str(checkpoint), device="cpu"))
    initial = torch.load(tmp_path / "run/initial.pt", weights_only=True)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(initial["model"][name], tensor, rtol=0, atol=0)
    assert report["parent_checkpoint"]["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert report["parent_checkpoint"]["epoch"] == 7
    assert report["parameter_l2_change_last_vs_initial"] > 0


def test_texture_source_families_do_not_cross_splits(tmp_path):
    archive = tmp_path / "textures.zip"
    rng = np.random.default_rng(15)
    with ZipFile(archive, "w") as zipped:
        zipped.writestr("License.txt", "Test fixture, generated in test")
        for family in ("Craters", "Super Perlin", "Streak", "Cracks"):
            for i in range(4):
                array = rng.integers(0, 256, (128, 128), dtype=np.uint8)
                stream = io.BytesIO()
                Image.fromarray(array).convert("RGB").save(stream, format="PNG")
                zipped.writestr(f"128x128/{family}/{family} {i}.png", stream.getvalue())
    scope = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/prepare_texture_dataset.py"))
    manifest = scope["prepare"](archive, tmp_path / "data", train=2, validation=1, test=1)
    families = {k: set(v) for k, v in manifest["families"].items()}
    for a in families:
        for b in families:
            if a != b:
                assert families[a].isdisjoint(families[b])
    assert manifest["image_counts"]["reserved"] == 4
    all_samples = [row for split in manifest["splits"].values() for row in split]
    assert not any(im["family"] == "Cracks" for row in all_samples for im in row["images"])


def test_chirp_sampling_diagnostics_distinguish_input_and_output():
    report = OpticsConfig().sampling_report()
    assert report["critical_distance_m"] == pytest.approx(0.02406015037594)
    assert all(plane["input_chirp_passes"] for plane in report["planes"])
    assert all(plane["output_chirp_nyquist_ratio"] > 1 for plane in report["planes"])
    for plane in report["planes"]:
        assert plane["input_chirp_nyquist_ratio"] * plane["output_chirp_nyquist_ratio"] == pytest.approx(1)
