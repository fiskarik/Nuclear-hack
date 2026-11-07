"""Обучение нейросети синтеза ДОЭ (без учителя, через модель Френеля).

Пример (GPU):
    python train.py --out runs/main --hours 3 --batch 24
Быстрая проверка на CPU:
    python train.py --out runs/smoke --steps 50 --batch 4 --bank 2000 --base 16 --stage-base 8
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from doe.data import build_bank, load_cache, load_textures
from doe.imageio import load_gray
from doe.losses import minmax_term, pearson_t, ssim_map, tradeoff_loss
from doe.model import DOENet, count_params
from doe.physics import FresnelSystem


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/main")
    ap.add_argument("--steps", type=int, default=0, help="число шагов (0 — определить по --hours)")
    ap.add_argument("--hours", type=float, default=1.0, help="бюджет времени, если --steps=0")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--bank", type=int, default=60000, help="размер банка синтетических изображений")
    ap.add_argument("--mix", default="0.35,0.15,0.30,0.20",
                    help="доли источников банка: синтетика, текстуры 128, фрагменты текстур 512, CIFAR-10 "
                         "(текстуры 512 и CIFAR берутся из data/cache, см. scripts/prepare_data.py)")
    ap.add_argument("--ssim-norm", choices=["fit", "max"], default="max",
                    help="нормировка восстановления в ИСС-части потерь (max — как при сохранении PNG)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--stages", type=int, default=4)
    ap.add_argument("--stage-base", type=int, default=24)
    ap.add_argument("--stage-depth", type=int, default=3)
    ap.add_argument("--grad-feat", action="store_true", help="подавать стадиям градиент потерь ∂L/∂|F_i|")
    ap.add_argument("--deep-sup", type=float, default=0.2, help="вес потерь промежуточных стадий")
    ap.add_argument("--w-mse", type=float, default=0.1)
    ap.add_argument("--w-minmax", type=float, default=0.0,
                    help="вес члена 1 − softmin(КК, ИСС): тянет вверх худшую из двух метрик")
    ap.add_argument("--select", choices=["minmax", "sum"], default="minmax",
                    help="критерий выбора чекпойнта: minmax — среднее по группам max_α min(КК, ИСС); "
                         "sum — среднее КК+ИСС при α=0,5")
    ap.add_argument("--profile", action="store_true", help="профилировать шаги 15–20 и вывести таблицу")
    ap.add_argument("--amp", action="store_true", help="fp16 для свёрток (GPU)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--examples", default="examples", help="папка с тестовыми наборами (для контроля)")
    ap.add_argument("--resume", default="")
    ap.add_argument("--init", default="", help="веса для дообучения (fine-tune)")
    return ap.parse_args()


def sample_alpha(b: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.uniform(0, 1, size=b)
    r = rng.uniform(size=b)
    a[r < 0.25] = 0.5          # чаще всего используемое значение
    a[(r >= 0.25) & (r < 0.33)] = 0.0
    a[(r >= 0.33) & (r < 0.41)] = 1.0
    return a.astype(np.float32)


def make_batch(bank: torch.Tensor, b: int, p: int, device: str) -> torch.Tensor:
    """bank — uint8 (K,N,N) на том же устройстве, что и обучение (на GPU выборка без копирования с CPU)."""
    idx = torch.randint(0, bank.shape[0], (b, p), device=bank.device)
    x = bank[idx].to(device).float().div_(255.0)
    fh = torch.rand(b, p, 1, 1, device=device) < 0.5
    fv = torch.rand(b, p, 1, 1, device=device) < 0.5
    x = torch.where(fh, x.flip(-1), x)
    x = torch.where(fv, x.flip(-2), x)
    gamma = torch.exp(torch.empty(b, p, 1, 1, device=device).uniform_(-0.35, 0.35))
    return x.clamp_min(0).pow(gamma)


def load_examples(root: str) -> dict[str, torch.Tensor]:
    out = {}
    if not root or not Path(root).is_dir():
        return out
    for d in sorted(Path(root).iterdir()):
        files = [d / f"object_{i}.png" for i in range(1, 5)]
        if all(f.exists() for f in files):
            out[d.name] = torch.tensor(np.stack([load_gray(f) for f in files])[None] / 255.0, dtype=torch.float32)
    return out


@torch.no_grad()
def evaluate(model, system, targets: torch.Tensor, alpha: float, bs: int = 32) -> dict[str, float]:
    """Метрики как при сохранении PNG: нормировка на максимум, квантование 0…255."""
    model.eval()
    pccs, ssims = [], []
    for i in range(0, len(targets), bs):
        t = targets[i:i + bs]
        psi = torch.angle(model(t, system, alpha)[-1])
        phi = system.to_physical(psi)
        phi = torch.round((phi + math.pi) / (2 * math.pi) * 255) / 255 * 2 * math.pi - math.pi
        rec = system.reconstruct(system.to_dechirped(phi))
        rec = torch.round(rec / rec.amax(dim=(-2, -1), keepdim=True) * 255) / 255
        tq = torch.round(t * 255) / 255
        pccs.append(pearson_t(rec, tq).mean(1))
        ssims.append(ssim_map(rec, tq).mean(dim=(-3, -2, -1)))
    model.train()
    return {"pcc": torch.cat(pccs).mean().item(), "ssim": torch.cat(ssims).mean().item()}


def main():
    args = parse()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = args.device
    if dev == "cuda":
        torch.backends.cudnn.benchmark = True

    t0 = time.time()
    mix = tuple(float(v) for v in args.mix.split(","))
    bank = build_bank(args.bank, seed=args.seed + 1, workers=args.workers, mix=mix)
    val_bank = build_bank(256, seed=987654)
    vrng = np.random.default_rng(4242)
    val = torch.tensor(val_bank[vrng.integers(0, len(val_bank), size=(64, 4))] / 255.0, dtype=torch.float32)
    val_sets = {"synth": val}
    tex_test = load_textures("test")
    if len(tex_test):
        val_sets["tex"] = torch.tensor(tex_test[vrng.integers(0, len(tex_test), size=(64, 4))] / 255.0,
                                       dtype=torch.float32)
    cifar_test = load_cache("cifar_gray", "test")
    if len(cifar_test):
        up = np.stack([np.asarray(Image.fromarray(c).resize((128, 128), Image.BICUBIC)) for c in cifar_test[:256]])
        val_sets["cifar"] = torch.tensor(up[vrng.integers(0, len(up), size=(64, 4))] / 255.0, dtype=torch.float32)
    print(f"bank {bank.shape} за {time.time() - t0:.1f} c", flush=True)

    bank_t = torch.from_numpy(bank).to(dev)
    del bank
    system = FresnelSystem().to(dev)
    model = DOENet(base=args.base, depth=args.depth, stages=args.stages,
                   stage_base=args.stage_base, stage_depth=args.stage_depth, grad_feat=args.grad_feat).to(dev)
    model.use_amp = args.amp
    if args.init:
        ck = torch.load(args.init, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"инициализация весами {args.init} (шаг {ck.get('step')})", flush=True)
    print(f"параметров: {count_params(model):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and dev == "cuda")
    step, best = 0, -1e9
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, best = ck["step"], ck.get("best", best)
        print("resume from step", step, flush=True)

    val_sets = {k: v.to(dev) for k, v in val_sets.items()}
    examples = {k: v.to(dev) for k, v in load_examples(args.examples).items()}
    total = args.steps
    warm = 300
    log = open(out / "train_log.jsonl", "a")
    t_start = time.time()
    t_hist = []
    while True:
        if total and step >= total:
            break
        if not total and len(t_hist) == 40:
            per = np.median(t_hist[10:])
            eval_overhead = 1.08
            total = step + int((args.hours * 3600 - (time.time() - t0)) / (per * eval_overhead))
            print(f"≈{per * 1000:.0f} мс/шаг → всего шагов: {total}", flush=True)
        ts = time.time()
        lr_total = total or 10 ** 9
        lr = args.lr * min(1.0, (step + 1) / warm) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(step / lr_total, 1.0))))
        for g in opt.param_groups:
            g["lr"] = lr
        if args.profile and step == 15:
            prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU] + (
                [torch.profiler.ProfilerActivity.CUDA] if dev == "cuda" else []))
            prof.__enter__()
        target = make_batch(bank_t, args.batch, 4, dev)
        alpha = torch.from_numpy(sample_alpha(args.batch, rng)).to(dev)
        hist = model(target, system, alpha)
        recs = [(s.unsqueeze(1) * system.rel_kernel) for s in hist]
        recs = [torch.fft.fft2(r, norm="ortho").abs() for r in recs]
        recs = [torch.fft.fftshift(r, dim=(-2, -1)) for r in recs]
        final = tradeoff_loss(recs[-1], target, alpha, w_mse=args.w_mse, ssim_norm=args.ssim_norm)
        loss = final["loss"]
        if args.w_minmax > 0:
            loss = loss + args.w_minmax * minmax_term(recs[-1], target)
        if args.deep_sup > 0 and len(recs) > 1:
            inter = sum(tradeoff_loss(r, target, alpha, w_mse=args.w_mse, ssim_norm=args.ssim_norm)["loss"] for r in recs[:-1]) / (len(recs) - 1)
            loss = loss + args.deep_sup * inter
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if scaler.is_enabled():
            scaler.step(opt)  # GradScaler сам пропускает шаг при inf/NaN — без синхронизации с CPU
        elif torch.isfinite(gn):
            opt.step()
        scaler.update()
        step += 1
        if len(t_hist) < 40 and dev == "cuda":
            torch.cuda.synchronize()
        t_hist.append(time.time() - ts)
        if args.profile and step == 20:
            if dev == "cuda":
                torch.cuda.synchronize()
            prof.__exit__(None, None, None)
            key = "self_cuda_time_total" if dev == "cuda" else "self_cpu_time_total"
            print(prof.key_averages().table(sort_by=key, row_limit=25), flush=True)
        if step % args.log_every == 0:
            rec = {"step": step, "lr": lr, "loss": loss.item(), "pcc": final["pcc"].item(),
                   "ssim": final["ssim"].item(), "mse": final["mse"].item(), "gn": float(gn),
                   "t": round(time.time() - t_start, 1)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if step % args.eval_every == 0 or step == total:
            ev = {"step": step, "val": {}}
            for vname, vt in val_sets.items():
                for a in (0.0, 0.5, 1.0):
                    ev["val"][f"{vname}_{a}"] = evaluate(model, system, vt, a)
            for name, t in examples.items():
                ev[name] = evaluate(model, system, t, 0.5)
            score_sum = float(np.mean([ev["val"][f"{v}_0.5"]["pcc"] + ev["val"][f"{v}_0.5"]["ssim"]
                                       for v in val_sets]))
            score_mm = float(np.mean([max(min(ev["val"][f"{v}_{a}"]["pcc"], ev["val"][f"{v}_{a}"]["ssim"])
                                          for a in (0.0, 0.5, 1.0)) for v in val_sets]))
            ev["score_sum"], ev["score_minmax"] = score_sum, score_mm
            score = score_mm if args.select == "minmax" else score_sum
            ev["score"] = score
            print("EVAL", json.dumps(ev), flush=True)
            log.write(json.dumps({"eval": ev}) + "\n")
            log.flush()
            ck = {"model": model.state_dict(), "config": model.config, "step": step, "best": max(best, score),
                  "eval": ev}
            if score > best:
                best = score
                torch.save({"model": model.state_dict(), "config": model.config, "step": step, "eval": ev},
                           out / "doe_net.pt")
            ck["opt"] = opt.state_dict()
            torch.save(ck, out / "ckpt_last.pt")
    print(f"готово: {step} шагов за {(time.time() - t_start) / 60:.1f} мин, лучший score={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
