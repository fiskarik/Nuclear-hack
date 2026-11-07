"""Потолок качества в зависимости от числа плоскостей (прямая оптимизация фазы). Аргументы: _ цифры_через_запятую буквы_через_запятую."""
import sys, time, math, torch, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from doe.physics import FresnelSystem, OpticalParams, phase_to_uint8, uint8_to_phase
from doe.imageio import load_gray, amplitude_to_uint8
from doe.metrics import pearson, ssim, ssim_7x7
from doe.losses import pearson_t, ssim_map
from doe.iterative import gerchberg_saxton
torch.set_num_threads(2)
Z = (24.1e-3, 25.3e-3, 26.5e-3, 27.7e-3)

def metrics(s, psi, imgs):
    rec = s.reconstruct_physical(uint8_to_phase(phase_to_uint8(s.to_physical(psi))))[0].numpy()
    r8 = [amplitude_to_uint8(rec[i], None, "max") for i in range(len(imgs))]
    return (np.mean([pearson(a, b) for a, b in zip(imgs, r8)]), np.mean([ssim(a, b) for a, b in zip(imgs, r8)]),
            np.mean([ssim_7x7(a, b) for a, b in zip(imgs, r8)]))

def optimize(s, t, iters, seed, gs_init=True):
    g = torch.Generator().manual_seed(seed)
    psi = ((torch.rand(1, 128, 128, generator=g) * 2 - 1) * math.pi)
    if gs_init:
        psi = gerchberg_saxton(s, t, iters=100, psi0=psi)
    psi = psi.clone().requires_grad_(True)
    opt = torch.optim.Adam([psi], lr=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=0.002)
    for it in range(iters):
        opt.zero_grad()
        r = s.reconstruct(psi)
        r = r / r.amax(dim=(-2, -1), keepdim=True)
        pcc = pearson_t(r, t).mean()
        ss = ssim_map(r, t).mean()
        # максимизируем худшую из двух метрик (мягкий минимум)
        loss = -(-torch.logsumexp(torch.stack([-pcc, -ss]) * 20, 0) / 20)
        loss.backward(); opt.step(); sched.step()
    return psi.detach()

for setname, path in [("digits", sys.argv[2]), ("letters", sys.argv[3])]:
    imgs_all = [load_gray(p) for p in path.split(",")]
    for P in (1, 2, 4):
        s = FresnelSystem(OpticalParams(distances=Z[:P]))
        imgs = imgs_all[:P]
        t = torch.tensor(np.stack(imgs)[None] / 255.0, dtype=torch.float32)
        best = None
        for seed in range(2 if P < 4 else 3):
            t0 = time.time(); psi = optimize(s, t, 3000, seed); m = metrics(s, psi, imgs)
            if best is None or min(m[0], m[1]) > min(best[0], best[1]): best = m
            print(f"{setname:8s} P={P} seed={seed}: КК={m[0]:.3f} SSIM={m[1]:.3f} SSIM7={m[2]:.3f} ({time.time()-t0:.0f} c)", flush=True)
        print(f"{setname:8s} P={P} ЛУЧШЕЕ: КК={best[0]:.3f} SSIM={best[1]:.3f} SSIM7={best[2]:.3f}", flush=True)
