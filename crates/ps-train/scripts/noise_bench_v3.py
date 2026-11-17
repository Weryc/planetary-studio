#!/usr/bin/env python3
"""Noise reductions compared on the simulated stacks with half stacks (dataset_giove_v3).

usage: .venv/bin/python noise_bench_v3.py CONFRONTO_DIR PREDS.json HALVES.json --out bench.json

CONFRONTO_DIR from `noise_bench_export` (raw, truth moved onto it, the app's learned reduction,
the subject's pixels); PREDS from `noise_train_halves.py --preds` (each stack's noise estimated
by networks that never saw its original); HALVES from `noise_halves` (the true random noise).

Score of a reduction, for a range of detail: 1 − error after / error before against the
noise-free stack (the truth through the raw's own blur): 1 = noise gone and detail untouched,
0 = nothing changed, below 0 = worse. Ranges: fine grain (< 5 px), medium grain (5–14 px),
colour (fine and medium). Versions: the app's learned reduction, a Gaussian blur σ 0.5 (the best
standard algorithm in the first comparison), and the network-guided reduction (each band of
brightness and colour keeps the share that is not noise) with the estimate of the network in
the app now, of the new network, and of the true noise (the ceiling).
"""
import json, math, os, sys
from multiprocessing import Pool
import numpy as np
from scipy import ndimage

LUMA = np.array([0.2126, 0.7152, 0.0722])
F_LO, F_HI, BINS = 0.012, 0.5, 22
root, preds_path, halves_path = sys.argv[1:4]
out = sys.argv[sys.argv.index("--out") + 1]
PREDS = json.load(open(preds_path))
HALVES = {r["stack"]: r for r in json.load(open(halves_path))}


def centre(k):
    return math.exp(math.log(F_LO) + (k - 0.5) / BINS * (math.log(F_HI) - math.log(F_LO))) if 1 <= k <= BINS else (0.006 if k == 0 else 0.6)


def band_index(side):
    f = np.fft.fftfreq(side)
    r = np.hypot(f[None, :], f[:, None])
    k = np.zeros_like(r, dtype=int)
    inside = (r >= F_LO) & (r <= F_HI)
    k[inside] = 1 + np.clip(((np.log(r[inside]) - math.log(F_LO)) / (math.log(F_HI) - math.log(F_LO)) * BINS).astype(int), 0, BINS - 1)
    k[r > F_HI] = BINS + 1
    return k


def bands(plane, idx):
    spec = np.fft.fft2(plane)
    return [np.real(np.fft.ifft2(np.where(idx == b, spec, 0))) for b in range(BINS + 2)]


def planes(img):
    if img.shape[0] == 1:
        return img[0], None
    y = np.tensordot(LUMA, img[:3], 1)
    return y, (img[0] - y, img[2] - y)


def from_planes(y, chroma):
    if chroma is None:
        return y[None]
    cr, cb = chroma
    r, b = y + cr, y + cb
    return np.stack([r, (y - LUMA[0] * r - LUMA[2] * b) / LUMA[1], b])


FINE = [k for k in range(2, 24) if centre(k) >= 0.2]
MEDIUM = [k for k in range(2, 24) if 0.07 <= centre(k) < 0.2]


def guided(img, idx, mask, mean, noise, skip_large):
    y, chroma = planes(img)
    def gains(bs, nz):
        g = np.ones(BINS + 2)
        for j, k in enumerate(range(2, 24)):
            if skip_large and centre(k) < 0.07:
                continue
            p = np.mean(sum(b[k][mask] ** 2 for b in bs)) / mean ** 2
            n = 10 ** nz[j] if nz[j] is not None and np.isfinite(nz[j]) else 0.0
            g[k] = max(0.0, 1 - n / max(p, 1e-30))
        return g
    yb = bands(y, idx)
    gy = gains([yb], noise[:22])
    y2 = sum(gy[k] * yb[k] for k in range(BINS + 2))
    if chroma is None:
        return y2[None]
    cbs = [bands(c, idx) for c in chroma]
    gc = gains(cbs, noise[22:])
    return from_planes(y2, tuple(sum(gc[k] * c[k] for k in range(BINS + 2)) for c in cbs))


def score_all(raw, truth, mask, idx, versions):
    ry, rc = planes(raw)
    ty, tc = planes(truth)
    def target(rp, tp):
        rb, tb = bands(rp, idx), bands(tp, idx)
        return [((r[mask] * t[mask]).sum() / max((t[mask] ** 2).sum(), 1e-30)) * t for r, t in zip(rb, tb)]
    sy = target(ry, ty)
    sc = [target(r, t) for r, t in zip(rc, tc)] if rc is not None else None
    def errors(img):
        y, c = planes(img)
        yb = bands(y, idx)
        ey = [np.mean((yb[k][mask] - sy[k][mask]) ** 2) for k in range(BINS + 2)]
        ec = None
        if sc is not None:
            cbs = [bands(p, idx) for p in c]
            ec = [sum(np.mean((cbs[j][k][mask] - sc[j][k][mask]) ** 2) for j in range(2)) for k in range(BINS + 2)]
        return ey, ec
    e0y, e0c = errors(raw)
    res = {}
    for name, img in versions.items():
        ey, ec = errors(img)
        s = {"fine": float(1 - sum(ey[k] for k in FINE) / sum(e0y[k] for k in FINE)), "media": float(1 - sum(ey[k] for k in MEDIUM) / sum(e0y[k] for k in MEDIUM))}
        if ec is not None:
            s["colore"] = float(1 - sum(ec[k] for k in FINE + MEDIUM) / sum(e0c[k] for k in FINE + MEDIUM))
        res[name] = s
    return res


def one(name):
    d = os.path.join(root, name)
    if name not in PREDS["solo simulazioni"] or name not in HALVES:
        return None
    raw, truth = np.load(f"{d}/grezza.npy"), np.load(f"{d}/verita.npy")
    mask = np.load(f"{d}/dentro.npy") > 0.5
    idx = band_index(raw.shape[-1])
    mean = float(planes(raw)[0][mask].mean())
    h = HALVES[name]
    true = list(np.array(h["rumore_grana"])[2:24]) + (list(np.array(h["rumore_colore"])[2:24]) if h["rumore_colore"] else [None] * 22)
    versions = {
        "Appresa (app)": np.load(f"{d}/app_appresa.npy"),
        "Gaussiana σ 0,5": np.stack([ndimage.gaussian_filter(c, 0.5) for c in raw]),
        "Rete attuale": guided(raw, idx, mask, mean, PREDS["app attuale"][name], False),
        "Rete nuova": guided(raw, idx, mask, mean, PREDS["solo simulazioni"][name], True),
        "Rumore vero (tetto)": guided(raw, idx, mask, mean, true, True),
    }
    return {"nome": name, "colore": raw.shape[0] > 1, "punteggi": score_all(raw, truth, mask, idx, versions)}


if __name__ == "__main__":
    names = sorted(os.listdir(root))
    with Pool(int(os.environ.get("PROCS", "6"))) as p:
        rows = [r for r in p.map(one, names, chunksize=2) if r]
    json.dump(rows, open(out, "w"))
    print(f"{len(rows)} stack")
    for m in rows[0]["punteggi"]:
        line = f"  {m:22}"
        for k in ("fine", "media", "colore"):
            v = [r["punteggi"][m][k] for r in rows if k in r["punteggi"][m]]
            line += f"  {k} {np.median(v):+.2f}"
        print(line)
