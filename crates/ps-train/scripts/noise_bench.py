#!/usr/bin/env python3
"""Noise reduction compared on the simulated Jupiter stacks, with a score anyone can read.

usage: .venv/bin/python noise_bench.py PIANO CONFRONTO_DIR RISULTATI.json --out bench.json

For each stack (from `noise_bench_export`: raw, truth moved onto it, the app's reductions) the
target is the noise-free stack: the truth seen through the raw's own blur, band by band (the
part of each band of the raw that moves with the truth). The score of a noise reduction, for
a range of detail sizes, is

    1 − error after / error before

over the subject: 1 = all the noise removed and the detail untouched, 0 = no change, below 0 =
worse than doing nothing (detail removed, or noise added). Three ranges: fine grain (detail
under 5 px), medium grain (5–14 px) on the brightness, and the colour noise (R − Y and B − Y,
all sizes).

Methods: the app's two (planetary wavelets, learned network; colour reduction 2 levels),
standard ones with their usual automatic settings (scikit-image / SciPy: Gaussian blur,
median, bilateral, wavelet BayesShrink, total variation, non-local means; noise level from
`estimate_sigma`), and a reduction guided by the noise network: each band kept in the share
that is not noise, per the network's estimate (made by networks that never saw this stack's
original, from `noise_train.py --out`); the same guided by the true noise is the ceiling such
a reduction can reach.
"""
import json, math, os, sys
from multiprocessing import Pool
import numpy as np
from scipy import ndimage
from skimage.restoration import denoise_bilateral, denoise_nl_means, denoise_tv_chambolle, denoise_wavelet, estimate_sigma

LUMA = np.array([0.2126, 0.7152, 0.0722])
F_LO, F_HI, BINS = 0.012, 0.5, 22
args = sys.argv[1:]
plan_path, root, res_path = args[0], args[1], args[2]
out = args[args.index("--out") + 1]
plan = {}
for line in open(plan_path).read().splitlines()[1:]:
    f = line.split("\t")
    plan[f[0]] = {"split": f[1], "gruppo": f[2], "telescopio": f[4], "camera": f[5], "d_r0": float(f[8]), "gain": f[9]}
RES = json.load(open(res_path))["righe"]
EST = {e["nome"]: e for e in RES if e["forza_riduzione"] == 0 and e["fase"] == "ridotta"}


def band_index(side):
    f = np.fft.fftfreq(side)
    r = np.hypot(f[None, :], f[:, None])
    k = np.zeros_like(r, dtype=int)
    inside = (r >= F_LO) & (r <= F_HI)
    k[inside] = 1 + np.clip(((np.log(r[inside]) - math.log(F_LO)) / (math.log(F_HI) - math.log(F_LO)) * BINS).astype(int), 0, BINS - 1)
    k[r > F_HI] = BINS + 1
    return k


def centre(k):
    lo, hi = math.log(F_LO), math.log(F_HI)
    return math.exp(lo + (k - 0.5) / BINS * (hi - lo))


def bands(plane, idx):
    spec = np.fft.fft2(plane)
    return [np.real(np.fft.ifft2(np.where(idx == b, spec, 0))) for b in range(BINS + 2)]


def planes(img):
    """Luminance and (for colour) the two colour differences."""
    if img.shape[0] == 1:
        return img[0], None
    y = np.tensordot(LUMA, img[:3], 1)
    return y, (img[0] - y, img[2] - y)


def from_planes(y, chroma, mono):
    if mono:
        return y[None]
    cr, cb = chroma
    r, b = y + cr, y + cb
    g = (y - LUMA[0] * r - LUMA[2] * b) / LUMA[1]
    return np.stack([r, g, b])


FINE = [k for k in range(2, BINS + 2) if (centre(k) if k <= BINS else 0.6) >= 0.2]
MEDIUM = [k for k in range(2, BINS + 2) if 0.07 <= (centre(k) if k <= BINS else 0.6) < 0.2]
ALL = list(range(2, BINS + 2))


def guided(img, idx, mask, mean, noise_lum, noise_col):
    """Each band of brightness and colour kept in the share that is not noise (a Wiener gain
    from the band's measured power and the estimated noise power, both log10 over mean²)."""
    y, chroma = planes(img)
    yb = bands(y, idx)
    def gains(bs, noise):
        g = np.ones(BINS + 2)
        for j, k in enumerate(range(2, BINS + 2)):
            p = np.mean(sum(b[k][mask] ** 2 for b in bs)) / mean ** 2
            n = 10 ** noise[j] if noise[j] is not None else 0.0
            g[k] = max(0.0, 1 - n / max(p, 1e-30))
        return g
    gy = gains([yb], noise_lum)
    y2 = sum(gy[k] * yb[k] for k in range(BINS + 2))
    if chroma is None:
        return y2[None]
    cb_ = [bands(c, idx) for c in chroma]
    gc = gains(cb_, noise_col)
    ch = tuple(sum(gc[k] * c[k] for k in range(BINS + 2)) for c in cb_)
    return from_planes(y2, ch, False)


def standard(img):
    """Standard reductions, each with its usual automatic setting."""
    mono = img.shape[0] == 1
    hwc = np.moveaxis(img, 0, -1)
    top = max(float(np.percentile(hwc, 99.9)), 1e-6)
    x = np.clip(hwc / top, 0, None)
    ax = None if mono else -1
    xx = x[..., 0] if mono else x
    sigma = float(estimate_sigma(xx, channel_axis=ax, average_sigmas=True))
    sigma = max(sigma, 1e-4)
    # Each at several strengths (multiples of its usual automatic setting; the Gaussian by its
    # σ in px): the comparison keeps, for each algorithm, the one strength that does best over
    # all the stacks, the same for every image.
    out = {}
    for g in (0.5, 0.7, 1.0, 1.5):
        out[f"gaussiana|σ {g}"] = np.stack([ndimage.gaussian_filter(x[..., c], g) for c in range(x.shape[-1])], -1)
    for m in (3, 5):
        out[f"mediana|{m}×{m}"] = np.stack([ndimage.median_filter(x[..., c], m) for c in range(x.shape[-1])], -1)
    for k in (1, 4, 16, 64):
        out[f"bilaterale|×{k}"] = denoise_bilateral(xx, sigma_color=2 * k * sigma, sigma_spatial=1.5, channel_axis=ax)
        out[f"wavelet|×{k}"] = denoise_wavelet(xx, sigma=k * sigma, method="BayesShrink", mode="soft", channel_axis=ax, convert2ycbcr=not mono, rescale_sigma=True)
        out[f"variazione totale|×{k}"] = denoise_tv_chambolle(xx, weight=k * sigma, channel_axis=ax)
        out[f"non-local means|×{k}"] = denoise_nl_means(xx, h=0.8 * k * sigma, sigma=k * sigma, fast_mode=True, patch_size=5, patch_distance=6, channel_axis=ax)
    res = {}
    for k, v in out.items():
        v = v[..., None] if v.ndim == 2 else v
        res[k] = np.moveaxis(v, -1, 0) * top
    return res, sigma * top


def score_all(img_raw, truth, mask, idx, versions):
    """Score of each version (fine grain, medium grain, colour) against the noise-free stack."""
    ry, rc = planes(img_raw)
    ty, tc = planes(truth)
    def target_bands(rp, tp):
        rb, tb = bands(rp, idx), bands(tp, idx)
        return [((r[mask] * t[mask]).sum() / max((t[mask] ** 2).sum(), 1e-30)) * t for r, t in zip(rb, tb)]
    sy = target_bands(ry, ty)
    sc = [target_bands(r, t) for r, t in zip(rc, tc)] if rc is not None and tc is not None else None
    def errors(img):
        y, c = planes(img)
        yb = bands(y, idx)
        ey = [np.mean((yb[k][mask] - sy[k][mask]) ** 2) for k in range(BINS + 2)]
        ec = None
        if sc is not None and c is not None:
            cbs = [bands(p, idx) for p in c]
            ec = [sum(np.mean((cbs[j][k][mask] - sc[j][k][mask]) ** 2) for j in range(2)) for k in range(BINS + 2)]
        return ey, ec
    e0y, e0c = errors(img_raw)
    out = {}
    for name, img in versions.items():
        ey, ec = errors(img)
        s = {"fine": float(1 - sum(ey[k] for k in FINE) / sum(e0y[k] for k in FINE)),
             "media": float(1 - sum(ey[k] for k in MEDIUM) / sum(e0y[k] for k in MEDIUM))}
        if ec is not None:
            s["colore"] = float(1 - sum(ec[k] for k in ALL) / sum(e0c[k] for k in ALL))
        out[name] = s
    return out


def one(name):
    d = os.path.join(root, name)
    raw, truth = np.load(f"{d}/grezza.npy"), np.load(f"{d}/verita.npy")
    mask = np.load(f"{d}/dentro.npy") > 0.5
    side = raw.shape[-1]
    idx = band_index(side)
    y, _ = planes(raw)
    mean = float(y[mask].mean())
    versions = {"app (pianeti)": np.load(f"{d}/app_pianeti.npy"), "app (rete appresa)": np.load(f"{d}/app_appresa.npy")}
    std, sigma = standard(raw)
    versions.update(std)
    e = EST.get(name)
    if e is not None:
        versions["guidata dalla rete del rumore"] = guided(raw, idx, mask, mean, e["grana_rete"], e["colore_rete"])
        versions["guidata dal rumore vero (tetto)"] = guided(raw, idx, mask, mean, e["grana_vero"], e["colore_vero"])
    return {"nome": name, **plan[name[:4]], "colore": raw.shape[0] > 1, "punteggi": score_all(raw, truth, mask, idx, versions)}


if __name__ == "__main__":
    names = sorted(os.listdir(root))
    with Pool(int(os.environ.get("PROCS", "6"))) as p:
        rows = p.map(one, names, chunksize=2)
    def overall(sc):
        return np.mean([v for v in sc.values()])
    # The best strength of each standard algorithm (same for every stack).
    best = {}
    for m in rows[0]["punteggi"]:
        if "|" in m:
            alg = m.split("|")[0]
            med = np.median([overall(r["punteggi"][m]) for r in rows])
            if alg not in best or med > best[alg][1]:
                best[alg] = (m, med)
    print("forza scelta per ogni algoritmo standard:", {a: v[0].split("|")[1] for a, v in best.items()})
    keep = [m for m in rows[0]["punteggi"] if "|" not in m] + [v[0] for v in best.values()]
    for r in rows:
        r["punteggi"] = {(m.split("|")[0] if "|" in m else m): r["punteggi"][m] for m in keep}
        r["punteggi_automatici"] = None
    json.dump({"righe": rows, "forze": {a: v[0].split("|")[1] for a, v in best.items()}}, open(out, "w"))
    print(f"{len(rows)} stack")
    for m in rows[0]["punteggi"]:
        line = f"{m:34}"
        for k in ("fine", "media", "colore"):
            v = [r["punteggi"][m][k] for r in rows if k in r["punteggi"][m]]
            line += f"  {k} {np.median(v):+.2f}"
        print(line)
