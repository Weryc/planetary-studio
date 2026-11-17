#!/usr/bin/env python3
"""Train the noise estimator on the simulated stacks (dataset from `noise_data`).

usage: .venv/bin/python noise_train.py PIANO dataset.json --out results.json --model model.json

For each stage of each stack (noise reduction 0/0.5/1/1.5, with and without the smart
sharpening at 100 %) the dataset holds what the estimator sees (ps_core::noise_net::features)
and the noise there really is inside the subject, band by band (log10 of its power over the
subject's mean squared), for the grain (luminance) and the colour (R − Y plus B − Y).

Model: an average of 5 small networks (one tanh hidden layer of 32) giving both spectra, bands
2‥23 (band 0 is the mean, band 1 holds almost no frequencies at these sizes), trained on the
squared error in log10 with weight decay; missing targets (mono images) are left out.
Scored leave-one-original-out (12 originals), against two simple estimates:
- the sky: the noise measured around the disk taken as the subject's (bands 8‥, needs a sky);
- white noise: the app's single noise σ spread evenly over the frequencies.
"""
import json, math, sys
import numpy as np

args = sys.argv[1:]
plan_path, data_path = args[0], args[1]
opt = lambda k: args[args.index(k) + 1] if k in args else None
plan = {}
for line in open(plan_path).read().splitlines()[1:]:
    f = line.split("\t")
    plan[f[0]] = f[2]

rows = json.load(open(data_path))
B0, B1 = 2, 24  # bands used
NB = B1 - B0
F_LO, F_HI, BINS = 0.012, 0.5, 22


def centre(k):
    lo, hi = math.log(F_LO), math.log(F_HI)
    return 0.5 * F_LO if k == 0 else 0.6 if k > BINS else math.exp(lo + (k - 0.5) / BINS * (hi - lo))


def edges(k):
    lo, hi = math.log(F_LO), math.log(F_HI)
    if k > BINS:
        return F_HI, 0.7071
    return math.exp(lo + (k - 1) / BINS * (hi - lo)), math.exp(lo + k / BINS * (hi - lo))


def nan(v):
    return np.array([np.nan if x is None else x for x in v], float)


X = np.array([r["x"] for r in rows], float)
YL = np.array([nan(r["rumore_grana"])[B0:B1] for r in rows])
YC = np.array([nan(r["rumore_colore"])[B0:B1] if r["rumore_colore"] else [np.nan] * NB for r in rows])
Y = np.concatenate([YL, YC], 1)
Y[Y < -13] = np.nan  # empty bands
group = np.array([plan[r["nome"][:4]] for r in rows])
print(f"{len(rows)} fasi da {len(set(r['nome'] for r in rows))} stack, {len(set(group))} originali; ingressi {X.shape[1]}")

# White-noise estimate from the app's σ: power in each band ∝ its area in the frequency square.
white = np.array([[math.log10(max(r["rumore_app"] ** 2 / r["media"] ** 2 * max(math.pi * (edges(k)[1] ** 2 - edges(k)[0] ** 2), 1e-9), 1e-14)) for k in range(B0, B1)] for r in rows])
# Sky estimate (luminance, bands 8‥23): its powers sit in the inputs at 24.. (bands 8‥23).
sky_l = np.full((len(rows), NB), np.nan)
for i, r in enumerate(rows):
    if r["cielo"]:
        sky_l[i, 8 - B0:] = X[i, 24:40]


class Scaler:
    def __init__(self, A):
        self.mu, self.sd = A.mean(0), A.std(0)
        self.sd[self.sd < 1e-6] = 1.0
        Z = (A - self.mu) / self.sd
        self.zmin, self.zmax = Z.min(0), Z.max(0)

    def __call__(self, x):
        return np.clip((x - self.mu) / self.sd, self.zmin - 0.5, self.zmax + 0.5)


def fit(tr, hidden=32, decay=1e-3, epochs=2500, seed=0):
    rng = np.random.default_rng(seed)
    sc = Scaler(X[tr])
    Z = sc(X[tr])
    T = Y[tr]
    mask = np.isfinite(T)
    T0 = np.where(mask, T, 0.0)
    ym = np.nanmean(T, 0)
    p = Z.shape[1]
    W1 = rng.normal(0, 1 / math.sqrt(p), (p, hidden)); b1 = np.zeros(hidden)
    W2 = rng.normal(0, 0.01, (hidden, Y.shape[1])); b2 = ym.copy()
    params = [W1, b1, W2, b2]
    mom = [np.zeros_like(q) for q in params]; vel = [np.zeros_like(q) for q in params]
    n = mask.sum()
    for it in range(1, epochs + 1):
        H = np.tanh(Z @ W1 + b1)
        P = H @ W2 + b2
        dP = 2 * (P - T0) * mask / n
        gW2 = H.T @ dP + decay * W2; gb2 = dP.sum(0)
        dH = (dP @ W2.T) * (1 - H * H)
        gW1 = Z.T @ dH + decay * W1; gb1 = dH.sum(0)
        for q, g, m_, v_ in zip(params, [gW1, gb1, gW2, gb2], mom, vel):
            m_ *= 0.9; m_ += 0.1 * g
            v_ *= 0.999; v_ += 0.001 * g * g
            q -= 0.01 * (m_ / (1 - 0.9 ** it)) / (np.sqrt(v_ / (1 - 0.999 ** it)) + 1e-8)
    f = lambda x: np.tanh(sc(x) @ W1 + b1) @ W2 + b2
    f.parts = (sc, W1, b1, W2, b2)
    return f


def ensemble(tr, n=5):
    nets = [fit(tr, seed=s) for s in range(n)]
    f = lambda x: np.mean([m(x) for m in nets], 0)
    f.nets = nets
    return f


groups = sorted(set(group))
pred = np.zeros_like(Y)
for g in groups:
    tr = np.where(group != g)[0]
    te = np.where(group == g)[0]
    pred[te] = ensemble(tr)(X[te])
    print(f"  lasciato fuori {g}", flush=True)


def scale_of(k):
    f = centre(k)
    return "fine" if f >= 0.2 else "media" if f >= 0.07 else "grande"


SCALES = ["fine", "media", "grande"]
band_scale = [scale_of(k) for k in range(B0, B1)]


def total(logp, scale):
    """RMS noise (% of the subject's mean) over the bands of a scale."""
    idx = [j for j, s in enumerate(band_scale) if s == scale]
    v = logp[..., idx]
    return 100 * np.sqrt(np.nansum(10 ** v, -1)) if np.ndim(v) > 1 else 100 * math.sqrt(np.nansum(10 ** v))


def score(P, T, cols):
    d = (P[:, cols] - T[:, cols])
    return np.sqrt(np.nanmean(d ** 2))


L, C = slice(0, NB), slice(NB, 2 * NB)
print("\nerrore medio in log10 per banda (0,30 = un fattore 2 in potenza, 1,4 in ampiezza):")
for name, P in (("rete", pred), ("cielo", np.concatenate([sky_l, np.full_like(sky_l, np.nan)], 1)), ("rumore bianco dell'app", np.concatenate([white, np.full_like(white, np.nan)], 1))):
    sl = [score(P[:, L], Y[:, L], [j for j in range(NB) if band_scale[j] == s]) for s in SCALES]
    sc_ = [score(P[:, C], Y[:, C], [j for j in range(NB) if band_scale[j] == s]) for s in SCALES]
    print(f"  {name:24} grana fine {sl[0]:.2f} media {sl[1]:.2f} grande {sl[2]:.2f} | colore fine {sc_[0]:.2f} media {sc_[1]:.2f} grande {sc_[2]:.2f}")

res = []
for i, r in enumerate(rows):
    e = {k: r[k] for k in ("nome", "forza_riduzione", "colore_riduzione", "fase", "errore", "cielo")}
    e["gruppo"] = str(group[i])
    for lab, sl in (("grana", L), ("colore", C)):
        for s in SCALES:
            e[f"{lab}_{s}_vero"] = float(total(Y[i, sl], s))
            e[f"{lab}_{s}_rete"] = float(total(pred[i, sl], s))
        e[f"{lab}_vero"] = [None if not np.isfinite(v) else float(v) for v in Y[i, sl]]
        e[f"{lab}_rete"] = [float(v) for v in pred[i, sl]]
    e["grana_cielo"] = [None if not np.isfinite(v) else float(v) for v in sky_l[i]]
    e["grana_bianco"] = [float(v) for v in white[i]]
    res.append(e)

# Totals: how well the network gets the RMS noise of each scale (ratio predicted / true).
print("\nrapporto rumore previsto / vero (mediana, e quota entro ±25 %):")
for lab in ("grana", "colore"):
    for s in SCALES:
        q = np.array([e[f"{lab}_{s}_rete"] / e[f"{lab}_{s}_vero"] for e in res if e[f"{lab}_{s}_vero"] > 0 and np.isfinite(e[f"{lab}_{s}_vero"])])
        print(f"  {lab:6} {s:6} mediana {np.median(q):.2f}  entro ±25 % {np.mean(np.abs(q - 1) < 0.25) * 100:.0f} %  (n={len(q)})")

if opt("--out"):
    json.dump({"bande": [centre(k) for k in range(B0, B1)], "scale_bande": band_scale, "righe": res}, open(opt("--out"), "w"))

if opt("--model"):
    final = ensemble(np.arange(len(rows)))
    sc = final.nets[0].parts[0]
    json.dump({"mu": sc.mu.tolist(), "sd": sc.sd.tolist(), "zmin": sc.zmin.tolist(), "zmax": sc.zmax.tolist(),
               "nets": [[q.tolist() for q in m.parts[1:]] for m in final.nets], "bande": [centre(k) for k in range(B0, B1)], "scale_bande": band_scale},
              open(opt("--model"), "w"))
    print("modello in", opt("--model"))
