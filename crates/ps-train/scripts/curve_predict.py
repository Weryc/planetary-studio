#!/usr/bin/env python3
"""Prototype: predict the best sharpening curve (gain per frequency band) from what the app can
measure on the raw alone, trained on the simulated dataset.

usage: curve_predict.py PIANO METRICS.json… --out gains.tsv [--setup]

- features: log power of each band over the raw's subject interior (sim_eval "potenza_bande"),
  the analysis (noise, SNR, cutoff, edge sigma), the disk size; with --setup also the sampling
  (λ/D in px) and the bit depth, which the app knows once the user names telescope and camera;
- target: log of the best gains (sim_eval "guadagni_bande", relative to band 0);
- model: ridge regression on standardised features, λ chosen by leave-one-group-out on train.
Training rows predict themselves leave-one-group-out (the image they come from never seen),
validation rows are predicted by the model fitted on all of train.
Writes "name<TAB>g0 … g23" for sim_eval --gains, and prints the errors in log-gain.
"""
import json, math, sys

args = sys.argv[1:]
out = args[args.index("--out") + 1]
setup = "--setup" in args
files = [a for a in args[1:] if a.endswith(".json")]
plan = {}
for line in open(args[0]).read().splitlines()[1:]:
    f = line.split("\t")
    plan[f[0]] = {"split": f[1], "group": f[2]}

rows = []
for path in files:
    for v in json.load(open(path))["versioni"]:
        gid = v["nome"].split("_")[0]
        m, a = v["meta"], v["analisi"]
        x = list(v["potenza_bande"])
        sig = a["sigma_bordo"]
        sig_ok = sig is not None and sig == sig
        # The app's analysis: cutoff and edge sigma may be missing (0 / NaN): flagged, not guessed.
        x += [math.log10(max(a["rumore"], 1e-7)), math.log10(max(a["snr"], 1.0)), a["taglio"], float(a["taglio"] > 0),
              sig if sig_ok else 5.0, float(sig_ok), math.log10(v["disco_px"])]
        if setup:
            x += [m["lambda_d_arcsec"] / m["scala_arcsec_px"], m["camera"]["bit"] / 12.0]
        g = v["guadagni_bande"]
        y = [math.log(min(max(gi if gi is not None else 1.0, 0.1), 100.0)) for gi in g]
        rows.append({"name": v["nome"], "id": gid, **plan[gid], "x": x, "y": y, "g": g})


def solve(m, r):
    n = len(r)
    m = [row[:] + [r[i]] for i, row in enumerate(m)]
    for c in range(n):
        p = max(range(c, n), key=lambda q: abs(m[q][c]))
        m[c], m[p] = m[p], m[c]
        for q in range(c + 1, n):
            f = m[q][c] / m[c][c]
            for j in range(c, n + 1):
                m[q][j] -= f * m[c][j]
    x = [0.0] * n
    for c in reversed(range(n)):
        x[c] = (m[c][n] - sum(m[c][j] * x[j] for j in range(c + 1, n))) / m[c][c]
    return x


def fit(train, lam):
    p = len(train[0]["x"])
    mu = [sum(r["x"][j] for r in train) / len(train) for j in range(p)]
    sd = [max(1e-6, math.sqrt(sum((r["x"][j] - mu[j]) ** 2 for r in train) / len(train))) for j in range(p)]
    ym = [sum(r["y"][k] for r in train) / len(train) for k in range(len(train[0]["y"]))]
    Z = [[(r["x"][j] - mu[j]) / sd[j] for j in range(p)] for r in train]
    A = [[sum(z[a] * z[b] for z in Z) + (lam if a == b else 0.0) for b in range(p)] for a in range(p)]
    W = []
    for k in range(len(ym)):
        rhs = [sum(z[a] * (r["y"][k] - ym[k]) for z, r in zip(Z, train)) for a in range(p)]
        W.append(solve(A, rhs))
    def predict(x):
        z = [(x[j] - mu[j]) / sd[j] for j in range(p)]
        return [ym[k] + sum(wk[j] * z[j] for j in range(p)) for k, wk in enumerate(W)]
    return predict


def err(pred, y):
    # RMS log-gain error over bands 1‥22 (band 0 is 1 by construction, 23 the corners).
    return math.sqrt(sum((pred[k] - y[k]) ** 2 for k in range(1, 23)) / 22)


train = [r for r in rows if r["split"] == "train"]
val = [r for r in rows if r["split"] == "val"]
groups = sorted({r["group"] for r in train})


def logo(lam):
    preds = {}
    for g in groups:
        model = fit([r for r in train if r["group"] != g], lam)
        for r in train:
            if r["group"] == g:
                preds[r["name"]] = model(r["x"])
    return preds


best = None
for lam in [0.3, 1, 3, 10, 30, 100]:
    preds = logo(lam)
    e = sum(err(preds[r["name"]], r["y"]) for r in train) / len(train)
    print(f"λ {lam:>5}: errore log-guadagno leave-one-group-out {e:.3f}")
    if best is None or e < best[0]:
        best = (e, lam, preds)
_, lam, preds = best
model = fit(train, lam)
for r in val:
    preds[r["name"]] = model(r["x"])
# Baseline: the mean curve of train, the same for every raw.
mean = [sum(r["y"][k] for r in train) / len(train) for k in range(24)]
for split, rs in (("train (LOGO)", train), ("val", val)):
    e = sum(err(preds[r["name"]], r["y"]) for r in rs) / len(rs)
    e0 = sum(err(mean, r["y"]) for r in rs) / len(rs)
    print(f"{split}: errore log-guadagno {e:.3f} (curva media {e0:.3f}), λ {lam}")
with open(out, "w") as f:
    for r in rows:
        g = [math.exp(v) for v in preds[r["name"]]]
        g[0] = 1.0
        f.write(r["name"] + "\t" + "\t".join(f"{v:.5f}" for v in g) + "\n")
