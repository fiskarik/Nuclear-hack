"""Запуск обучения на GPU Kaggle через Kaggle API.

Код проекта упаковывается (tar.gz → base64) внутрь скрипта ноутбука,
поэтому не нужно ни загружать датасеты, ни включать интернет в ноутбуке.
На машине «GPU T4 ×2» каждое задание (--job) запускается на своей GPU.

Пример:
    python scripts/kaggle_run.py push --slug doe-nn-train \
        --job "main:--hours 2.5 --batch 24 --amp" \
        --job "deep:--hours 2.5 --batch 16 --stages 8 --amp"
    python scripts/kaggle_run.py wait --slug doe-nn-train
    python scripts/kaggle_run.py pull --slug doe-nn-train --dest runs/kaggle

Требуется настроенный Kaggle CLI (токен в ~/.kaggle/access_token или
KAGGLE_API_TOKEN). Команду CLI можно переопределить переменной KAGGLE_CLI.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = shlex.split(os.environ.get("KAGGLE_CLI", "kaggle"))
INCLUDE = ["doe", "train.py", "synthesize.py", "benchmark.py", "examples", "data"]

KERNEL_TEMPLATE = r'''# Автоматически сгенерировано scripts/kaggle_run.py — обучение нейросети синтеза ДОЭ
import base64, io, os, subprocess, sys, tarfile, time, json, shlex
PAYLOAD = "__PAYLOAD__"
JOBS = json.loads(__JOBS__)
SRC = "/kaggle/working/src"
os.makedirs(SRC, exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)), mode="r:gz").extractall(SRC)
os.chdir(SRC)
# подключённые Kaggle-датасеты с кэшами (textures512.npz, cifar_gray.npz) → data/cache
os.makedirs("data/cache", exist_ok=True)
for root, _, files in os.walk("/kaggle/input"):
    for f in files:
        if f.endswith(".npz") and not os.path.exists(os.path.join("data/cache", f)):
            os.symlink(os.path.join(root, f), os.path.join("data/cache", f))
print("cache:", os.listdir("data/cache"), flush=True)
# подключённые веса (*.pt) → weights/ (для дообучения: --init weights/<имя>.pt)
os.makedirs("weights", exist_ok=True)
for root, _, files in os.walk("/kaggle/input"):
    for f in files:
        if f.endswith(".pt") and not os.path.exists(os.path.join("weights", f)):
            os.symlink(os.path.join(root, f), os.path.join("weights", f))
print("weights:", os.listdir("weights"), flush=True)
# архивы в подключённых датасетах (например, imagenette-160.tgz) → /kaggle/tmp/extracted
for root, _, files in os.walk("/kaggle/input"):
    for f in files:
        if f.endswith((".tgz", ".tar.gz", ".tar")):
            dst = os.path.join("/kaggle/tmp/extracted", f.split(".")[0])
            os.makedirs(dst, exist_ok=True)
            try:
                with tarfile.open(os.path.join(root, f)) as tf:
                    tf.extractall(dst, filter="data")
                print("extracted", f, "->", dst, flush=True)
            except Exception as e:
                print("extract failed", f, e, flush=True)
subprocess.run("nvidia-smi; python -c 'import torch,sys;print(sys.version, torch.__version__, torch.cuda.device_count())'", shell=True)
try:
    import torch
    ngpu = torch.cuda.device_count()
except Exception:
    ngpu = 0
if ngpu == 0 and os.environ.get("REQUIRE_GPU", "__REQUIRE_GPU__") == "1":
    raise SystemExit("GPU недоступен (проверьте подтверждение телефона в настройках Kaggle) — обучение не запущено")
ngpu = max(ngpu, 1)
procs = []
cpus = os.cpu_count() or 2
for i, (name, args) in enumerate(JOBS):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % ngpu), PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS=str(max(1, cpus // max(len(JOBS), 1))))
    cmd = [sys.executable, "train.py", "--out", f"/kaggle/working/runs/{name}",
           "--workers", str(max(1, cpus // max(len(JOBS), 1)))] + shlex.split(args)
    os.makedirs(f"/kaggle/working/runs/{name}", exist_ok=True)
    log = open(f"/kaggle/working/runs/{name}/stdout.log", "w")
    print("START", name, " ".join(cmd), flush=True)
    procs.append((name, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)))
last = time.time()
while any(p.poll() is None for _, p in procs):
    time.sleep(30)
    if time.time() - last > 600:
        last = time.time()
        for name, _ in procs:
            try:
                tail = open(f"/kaggle/working/runs/{name}/stdout.log").read().splitlines()[-2:]
                print(name, "|", " || ".join(tail)[-600:], flush=True)
            except Exception:
                pass
for name, p in procs:
    print("DONE", name, "exit", p.returncode, flush=True)
    print(open(f"/kaggle/working/runs/{name}/stdout.log").read()[-4000:], flush=True)
'''


def build_payload() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for item in INCLUDE:
            p = ROOT / item
            if p.exists():
                tar.add(p, arcname=item,
                        filter=lambda ti: None if ("__pycache__" in ti.name or ti.name.startswith(("data/cache", "data/textures/"))
                                                   or ti.name.endswith(".png") and ti.name.startswith("data/")) else ti)
    return base64.b64encode(buf.getvalue()).decode()


def username() -> str:
    return os.environ.get("KAGGLE_USERNAME", "fiskarik")


def cmd_push(a):
    jobs = []
    for j in a.job:
        name, _, args = j.partition(":")
        jobs.append([name.strip(), args.strip()])
    kdir = Path(a.workdir) / a.slug
    kdir.mkdir(parents=True, exist_ok=True)
    script = (KERNEL_TEMPLATE.replace("__PAYLOAD__", build_payload()).replace("__JOBS__", repr(json.dumps(jobs)))
              .replace("__REQUIRE_GPU__", "1" if a.require_gpu else "0"))
    (kdir / "run_kernel.py").write_text(script)
    meta = {
        "id": f"{username()}/{a.slug}", "title": a.slug, "code_file": "run_kernel.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": a.dataset or [], "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"kernel: {kdir} ({len(script) / 1024:.0f} КБ)")
    cmd = CLI + ["kernels", "push", "-p", str(kdir), "--accelerator", a.accelerator]
    if a.timeout:
        cmd += ["-t", str(a.timeout)]
    subprocess.run(cmd, check=True)


def status(slug: str) -> str:
    r = subprocess.run(CLI + ["kernels", "status", f"{username()}/{slug}"], capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def cmd_wait(a):
    t0 = time.time()
    while True:
        s = status(a.slug)
        print(f"[{(time.time() - t0) / 60:6.1f} мин] {s}", flush=True)
        low = s.lower()
        if any(k in low for k in ("complete", "error", "cancel")):
            break
        time.sleep(a.interval)


def cmd_pull(a):
    Path(a.dest).mkdir(parents=True, exist_ok=True)
    subprocess.run(CLI + ["kernels", "output", f"{username()}/{a.slug}", "-p", a.dest, "-o"], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push")
    p.add_argument("--slug", required=True)
    p.add_argument("--job", action="append", required=True, help='"имя:аргументы train.py"')
    p.add_argument("--accelerator", default="NvidiaTeslaT4")
    p.add_argument("--dataset", action="append", help="подключить Kaggle-датасет (owner/slug), напр. fiskarik/doe-nn-data")
    p.add_argument("--require-gpu", action="store_true", help="не обучать, если Kaggle выдал машину без GPU")
    p.add_argument("--timeout", type=int, default=0, help="жёсткий лимит времени работы ноутбука, с")
    p.add_argument("--workdir", default=str(ROOT / "kaggle_kernels"))
    p.set_defaults(fn=cmd_push)
    w = sub.add_parser("wait")
    w.add_argument("--slug", required=True)
    w.add_argument("--interval", type=int, default=60)
    w.set_defaults(fn=cmd_wait)
    s = sub.add_parser("status")
    s.add_argument("--slug", required=True)
    s.set_defaults(fn=lambda a: print(status(a.slug)))
    d = sub.add_parser("pull")
    d.add_argument("--slug", required=True)
    d.add_argument("--dest", default="runs/kaggle")
    d.set_defaults(fn=cmd_pull)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
