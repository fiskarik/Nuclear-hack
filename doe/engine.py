"""Reproducible training, held-out evaluation and inference."""
from dataclasses import asdict, dataclass
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import QuartetDataset, load_quartet
from .model import FresnelNet, build_model, gerchberg_saxton, physics_loss
from .optics import OpticsConfig
from .quality import perceptual_loss_per_item
from .refinement import refine_phase
from .results import export_result, phase_metrics


@dataclass
class TrainConfig:
    epochs: int = 24
    batch_size: int = 4
    learning_rate: float = 0.005
    stages: int = 6
    width: int = 12
    seed: int = 42
    device: str = "auto"
    threads: int = 2
    initial_checkpoint: str | None = None
    objective: str = "physics"
    architecture: str = "fixed"
    initializer_width: int = 8
    initializer_depth: int = 3


def setup(seed=42, device="auto", threads=2):
    if threads < 1:
        raise ValueError("threads must be >= 1")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; select --device cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return torch.device(device)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_model(checkpoint, device):
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if state.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    optics = OpticsConfig(**state["optics"])
    model = build_model(optics, state["architecture"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


@torch.no_grad()
def validation_loss(model, loader, device, objective="physics"):
    model.eval()
    total, count = 0.0, 0
    for targets in loader:
        targets = targets.to(device)
        phase=model(targets)
        loss = (perceptual_loss_per_item(model.optics,phase,targets).mean() if objective=="perceptual"
                else physics_loss(model.optics,phase,targets))
        total += float(loss) * len(targets)
        count += len(targets)
    if not count:
        raise ValueError("validation set is empty")
    return total / count


def train(data, output, config=TrainConfig()):
    if config.epochs < 1 or config.batch_size < 1 or config.learning_rate <= 0:
        raise ValueError("epochs, batch size and learning rate must be positive")
    if config.objective not in ("physics","perceptual"):
        raise ValueError("objective must be physics or perceptual")
    if config.architecture not in ('fixed','wavefront','adaptive'):
        raise ValueError('architecture must be fixed, wavefront or adaptive')
    device = setup(config.seed, config.device, config.threads)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite training run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    train_data, val_data = QuartetDataset(data, "train"), QuartetDataset(data, "validation")
    loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(config.seed), num_workers=0)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, num_workers=0)
    parent = None
    requested={'kind':config.architecture,'stages':config.stages,'width':config.width}
    if config.architecture!='fixed':
        requested.update(initializer_width=config.initializer_width,initializer_depth=config.initializer_depth)
    if config.initial_checkpoint:
        model, previous = load_model(config.initial_checkpoint, device)
        old_arch=model.architecture();old_stages=model.stages
        if model.width!=config.width or config.stages<old_stages:
            raise ValueError('width must match and stage count cannot decrease during fine-tuning')
        if config.architecture=='fixed' and (old_arch['kind']!='fixed' or config.stages!=old_stages):
            raise ValueError('requested fixed architecture differs from checkpoint')
        if old_arch['kind']=='adaptive' and config.architecture!='adaptive':
            raise ValueError('cannot discard adaptive projection layers during fine-tuning')
        if old_arch['kind']!='fixed' and (config.initializer_width!=old_arch['initializer_width'] or config.initializer_depth<old_arch['initializer_depth']):
            raise ValueError('initializer width must match and depth cannot decrease')
        if old_arch!=requested:
            expanded=build_model(model.optics.config,requested).to(device)
            transferred=dict(model.state_dict())
            if config.stages!=old_stages:
                indices=torch.arange(config.stages,device=device)%old_stages
                transferred['plane_logits']=transferred['plane_logits'][indices].clone()
                transferred['relax_logits']=transferred['relax_logits'][indices].clone()
            missing,unexpected=expanded.load_state_dict(transferred,strict=False)
            allowed=lambda k:k.startswith(('phase_initializer.','amplitude_modulators.')) or k in ('reference_phase','coordinate_grid','deep_scale')
            assert not unexpected and all(allowed(k) for k in missing)
            model=expanded
        parent = {"filename": Path(config.initial_checkpoint).name,
                  "sha256": hashlib.sha256(Path(config.initial_checkpoint).read_bytes()).hexdigest(),
                  "epoch": previous["epoch"],"architecture":old_arch}
    else:
        model = build_model(OpticsConfig(),requested).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    initial = validation_loss(model, val_loader, device, config.objective)
    initial_parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
    best_loss, best_epoch = initial, 0
    history = [{"epoch": 0, "train_loss": None, "validation_loss": initial}]
    manifest_hash = hashlib.sha256((Path(data) / "manifest.json").read_bytes()).hexdigest()

    def save(path, epoch, val_loss):
        torch.save({"format_version": 1, "model": model.state_dict(),
                    "architecture": model.architecture(),
                    "optics": model.optics.config.to_dict(), "training": asdict(config),
                    "epoch": epoch, "validation_loss": val_loss,
                    "dataset_manifest_sha256": manifest_hash, "parent_checkpoint": parent}, path)

    save(output / "initial.pt", 0, initial)
    save(output / "best.pt", 0, initial)
    start = time.perf_counter()
    print(f"device={device}; train={len(train_data)} validation={len(val_data)}; initial_val={initial:.6f}", flush=True)
    for epoch in range(1, config.epochs + 1):
        model.train()
        total = 0.0
        for targets in loader:
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            phase = model(targets)
            loss = (perceptual_loss_per_item(model.optics,phase,targets).mean() if config.objective=="perceptual"
                    else physics_loss(model.optics,phase,targets))
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(targets)
        val_loss = validation_loss(model, val_loader, device, config.objective)
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            save(output / "best.pt", epoch, val_loss)
        record = {"epoch": epoch, "train_loss": total / len(train_data), "validation_loss": val_loss,
                  "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(record)
        scheduler.step()
        print(f"epoch={epoch:03d} train={record['train_loss']:.6f} val={val_loss:.6f} best={best_loss:.6f}", flush=True)
    sync(device)
    save(output / "last.pt", config.epochs, history[-1]["validation_loss"])
    drift = sum(float((p.detach() - initial_parameters[name]).square().sum()) for name, p in model.named_parameters())**0.5
    report = {"config": asdict(config), "device": str(device), "torch": torch.__version__,
              "train_count": len(train_data), "validation_count": len(val_data),
              "selection": f"minimum validation {config.objective} loss; test split never used for selection",
              "initial_validation_loss": initial, "best_validation_loss": best_loss, "best_epoch": best_epoch,
              "parameter_l2_change_last_vs_initial": drift,
              "elapsed_seconds": time.perf_counter() - start, "history": history,
              "dataset_manifest_sha256": manifest_hash, "parent_checkpoint": parent,
              "optimizer": "fresh Adam state; fine-tuning weights, not resuming optimizer state"}
    (output / "training.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


@torch.no_grad()
def evaluate(data, checkpoint, output, device="auto", batch_size=4, gs_steps=80, threads=2, refine_steps=0, refine_lr=0.08):
    device = setup(device=device, threads=threads)
    model, state = load_model(checkpoint, device)
    initial = build_model(model.optics.config, model.architecture()).to(device).eval()
    # Identical analytical initialization; CNN starts as exact zero correction.
    test = QuartetDataset(data, "test")
    loader = DataLoader(test, batch_size=batch_size, num_workers=0)
    methods = {"neural": lambda x: model(x), "untrained_network": lambda x: initial(x),
               f"gs_{model.stages}": lambda x: gerchberg_saxton(model.optics, x, model.stages),
               f"gs_{gs_steps}": lambda x: gerchberg_saxton(model.optics, x, gs_steps)}
    if refine_steps:
        methods['neural_refined']=lambda x: refine_phase(model.optics,model(x),x,steps=refine_steps,lr=refine_lr)[0]
    records = {name: [] for name in methods}
    seconds = {name: 0.0 for name in methods}
    # Exclude one warmup batch per method from performance measurements.
    warmup = next(iter(loader)).to(device)
    for name,operation in methods.items():
        # Warm kernels without repeating a potentially long optimization run.
        if name=='neural_refined':
            refine_phase(model.optics,model(warmup),warmup,steps=min(2,refine_steps),lr=refine_lr)
        else:
            operation(warmup)
    for targets in loader:
        targets = targets.to(device)
        for name, operation in methods.items():
            sync(device)
            start = time.perf_counter()
            phase = operation(targets)
            sync(device)
            seconds[name] += time.perf_counter() - start
            records[name].extend(phase_metrics(model.optics, phase, targets))
    summary = {}
    for name, values in records.items():
        correlations = [v["pearson"] for v in values if v["pearson"] is not None]
        summary[name] = {"count": len(values), "mean_pearson": float(np.mean(correlations)) if correlations else None,
                         "mean_ssim": float(np.mean([v["ssim"] for v in values])),
                         "seconds_per_quartet_batched": seconds[name] / len(values)}
    report = {"split": "test", "sample_count": len(test), "device": str(device), "batch_size": batch_size,
              "architecture":model.architecture(),
              "checkpoint_epoch": state["epoch"], "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
              "quantization": "8-bit phase and 8-bit reconstructed amplitudes, same for all methods",
              "timing": "one warmed pass per test batch; includes phase synthesis, excludes image IO and metric computation; batched throughput, not single-sample latency",
              "summary": summary, "per_sample": records}
    if refine_steps:
        report['refinement']={'steps':refine_steps,'lr':refine_lr,'method':'neural initializer + per-input phase Adam; not a pure network'}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {output}")
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return report


@torch.no_grad()
def predict(checkpoint, images, output, device="auto", threads=2, source="user PNGs",refine_steps=0,refine_lr=0.08):
    device = setup(device=device, threads=threads)
    model, state = load_model(checkpoint, device)
    targets = load_quartet(images).to(device)
    model(targets)  # warmup
    sync(device)
    start = time.perf_counter()
    phase = model(targets)
    sync(device)
    elapsed = time.perf_counter() - start
    refinement=None
    if refine_steps:
        phase,refinement=refine_phase(model.optics,phase,targets,steps=refine_steps,lr=refine_lr)
    metadata = {"method": "FresnelNet", "network_stages": model.stages, "cnn_width": model.width,
                "architecture":model.architecture(),
                "source": source, "device": str(device), "checkpoint_epoch": state["epoch"],
                "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                "network_forward_seconds": elapsed, "test_time_optimizer_steps": refine_steps}
    if refinement:
        metadata['method']='FresnelNet + per-input phase Adam'
        metadata['refinement']=refinement
    return export_result(output, model.optics, phase, images, metadata)
