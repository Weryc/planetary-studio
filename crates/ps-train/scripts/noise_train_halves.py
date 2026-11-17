#!/usr/bin/env python3
"""Train the noise estimator on the true random noise of stacks: half the difference of their
even- and odd-frame half stacks (noise_halves), on simulated stacks and real recordings.

usage: .venv/bin/python noise_train_halves.py SIM.json PIANO REAL.json OLD_MODEL.json OLD_DATA.rs
                                              [--export crates/ps-core/src/noise_net_data.rs] [--out results.json]

Targets: log10 noise power over the disk's mean squared, bands 2‥23, grain and colour. Only
bands of fine and medium detail (centre ≥ 0.07 cycles/px) count: at larger scales the half
stacks differ mostly by the seeing's leftover deformation, which is not noise to remove.
Inputs: ps_core::noise_net::features on the full stack, and the same with its sky clipped to
black (both used for training).

Models compared:
- the one in the app now (old network + fixed calibration on the real stacks);
- new, trained on the simulations only;
- new, trained on the simulations and the real stacks (these weighted ×5: few, but real).
Scores: simulations leave-one-original-out; real stacks leave-one-recording-out.
"""
import json, math, re, sys
import numpy as np

args = sys.argv[1:]
sim_path, plan_path, real_path, old_model_path, old_rs_path = args[:5]
opt = lambda k: args[args.index(k) + 1] if k in args else None
plan = {l.split("\t")[0]: l.split("\t") for l in open(plan_path).read().splitlines()[1:]}
F_LO, F_HI, BINS = 0.012, 0.5, 22
centre = lambda k: math.exp(math.log(F_LO) + (k - 0.5) / BINS * (math.log(F_HI) - math.log(F_LO))) if k <= BINS else 0.6
BANDS = list(range(2, 24))
NB = len(BANDS)
USED = np.array([centre(k) >= 0.07 for k in BANDS])
SCALE = ["fine" if centre(k) >= 0.2 else "media" if centre(k) >= 0.07 else "grande" for k in BANDS]


def load(path, real):
    rows = []
    for r in json.load(open(path)):
        y = np.full(2 * NB, np.nan)
        y[:NB] = np.array(r["rumore_grana"])[2:24]
        if r["rumore_colore"]:
            y[NB:] = np.array(r["rumore_colore"])[2:24]
        y[np.concatenate([~USED, ~USED])] = np.nan
        y[y < -13] = np.nan
        group = r["stack"].split("_")[0] if real else plan[r["stack"][:4]][2]
        rows.append({"nome": r["stack"], "real": real, "group": group, "x": np.array(r["x"]), "xb": np.array(r["x_nero"]) if r.get("x_nero") else np.array(r["x"]), "y": y})
    return rows


SIM, REAL = load(sim_path, False), load(real_path, True)
ALL = SIM + REAL
print(f"{len(SIM)} stack simulati ({len(set(r['group'] for r in SIM))} originali), {len(REAL)} veri ({len(set(r['group'] for r in REAL))} riprese)")


class Scaler:
    def __init__(self, A):
        self.mu, self.sd = A.mean(0), A.std(0)
        self.sd[self.sd < 1e-6] = 1.0
        Z = (A - self.mu) / self.sd
        self.zmin, self.zmax = Z.min(0), Z.max(0)

    def __call__(self, x):
        return np.clip((x - self.mu) / self.sd, self.zmin - 0.5, self.zmax + 0.5)


def fit(rows, hidden=32, decay=1e-3, epochs=3000, seed=0, real_weight=5.0):
    X = np.array([r["x"] for r in rows] + [r["xb"] for r in rows])
    Y = np.array([r["y"] for r in rows] * 2)
    wt = np.array([real_weight if r["real"] else 1.0 for r in rows] * 2)
    rng = np.random.default_rng(seed)
    sc = Scaler(X)
    Z = sc(X)
    mask = np.isfinite(Y)
    T = np.where(mask, Y, 0.0)
    ym = np.nanmean(np.where(mask, Y, np.nan), 0)
    ym = np.where(np.isfinite(ym), ym, -9.0)
    W = mask * wt[:, None]
    p = Z.shape[1]
    W1 = rng.normal(0, 1 / math.sqrt(p), (p, hidden)); b1 = np.zeros(hidden)
    W2 = rng.normal(0, 0.01, (hidden, Y.shape[1])); b2 = ym.copy()
    params = [W1, b1, W2, b2]
    mom = [np.zeros_like(q) for q in params]; vel = [np.zeros_like(q) for q in params]
    n = W.sum()
    for it in range(1, epochs + 1):
        H = np.tanh(Z @ W1 + b1)
        P = H @ W2 + b2
        dP = 2 * (P - T) * W / n
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


def ensemble(rows, n=5, **kw):
    nets = [fit(rows, seed=s, **kw) for s in range(n)]
    f = lambda x: np.mean([m(x) for m in nets], 0)
    f.nets = nets
    return f


# The model in the app now.
OM = json.load(open(old_model_path))
cal = np.array([float(v) for v in re.search(r"calibration: &\[(.*?)\]", open(old_rs_path).read(), re.S).group(1).split(",")])
omu, osd, ozmin, ozmax = (np.array(OM[k]) for k in ("mu", "sd", "zmin", "zmax"))


def old(x):
    z = np.clip((x - omu) / osd, ozmin - 0.5, ozmax + 0.5)
    with np.errstate(all="ignore"):
        return np.mean([np.tanh(z @ np.array(a) + np.array(b)) @ np.array(c) + np.array(d) for a, b, c, d in OM["nets"]], 0) + cal


def total(v, sel):
    return 100 * math.sqrt(sum(10 ** v[j] for j in sel))


KEYS = [("grana fine", 0, "fine"), ("grana media", 0, "media"), ("colore fine", NB, "fine"), ("colore medio", NB, "media")]


def ratios(pred, rows):
    out = {}
    for lab, off, s in KEYS:
        sel = [j for j in range(NB) if SCALE[j] == s]
        q = [total(pred[r["nome"]][off:off + NB], sel) / total(r["y"][off:off + NB], sel) for r in rows if np.all(np.isfinite(r["y"][off:off + NB][sel]))]
        out[lab] = q
    return out


def report(name, pred, rows):
    rr = ratios(pred, rows)
    line = f"  {name:34}"
    for lab, q in rr.items():
        if q:
            line += f" {lab}: {np.median(q):.2f} ({100*np.mean(np.abs(np.log(q)) < math.log(1.25)):.0f}%)"
    print(line)
    return {lab: {"mediana": float(np.median(q)), "entro25": float(np.mean(np.abs(np.log(q)) < math.log(1.25))), "rapporti": [float(v) for v in q]} for lab, q in rr.items() if q}


results = {}
P_old = {r["nome"]: old(r["x"]) for r in ALL}
P_sim, P_mix = {}, {}
for g in sorted(set(r["group"] for r in SIM)):
    tr_sim = [r for r in SIM if r["group"] != g]
    a, b = ensemble(tr_sim), ensemble(tr_sim + REAL)
    for r in SIM:
        if r["group"] == g:
            P_sim[r["nome"]], P_mix[r["nome"]] = a(r["x"]), b(r["x"])
    print(f"  originale {g} lasciato fuori", flush=True)
a = ensemble(SIM)
for r in REAL:
    P_sim[r["nome"]] = a(r["x"])
for g in sorted(set(r["group"] for r in REAL)):
    b = ensemble(SIM + [r for r in REAL if r["group"] != g])
    for r in REAL:
        if r["group"] == g:
            P_mix[r["nome"]] = b(r["x"])
    print(f"  ripresa {g} lasciata fuori", flush=True)
print("stima / rumore vero (mediana) e quota entro ±25 %:")
for where, rows in (("simulazioni", SIM), ("riprese vere", REAL)):
    print(where)
    results[where] = {"app attuale": report("rete attuale nell'app", P_old, rows), "solo simulazioni": report("nuova, solo simulazioni", P_sim, rows), "simulazioni + vere": report("nuova, simulazioni + riprese vere", P_mix, rows)}
if opt("--out"):
    json.dump(results, open(opt("--out"), "w"))
if opt("--preds"):
    json.dump({k: {n: v.tolist() for n, v in d.items()} for k, d in (("app attuale", P_old), ("solo simulazioni", P_sim), ("simulazioni + vere", P_mix))}, open(opt("--preds"), "w"))

if opt("--export"):
    # Trained on the simulations only: with the real stacks as well (weighted or not) the
    # network did worse on the real recording it hadn't seen; the real stacks are the check.
    final = ensemble(SIM)
    sc = final.nets[0].parts[0]
    fl = lambda a_: "&[" + ", ".join(f"{float(v):.6}" for v in np.ravel(a_)) + "]"
    hidden = final.nets[0].parts[1].shape[1]
    nets = "\n".join(f"        Net {{ w1: {fl(m.parts[1])}, b1: {fl(m.parts[2])}, w2: {fl(m.parts[3])}, b2: {fl(m.parts[4])} }}," for m in final.nets)
    src = f"""//! Weights of the noise estimator, written by `ps-train/scripts/noise_train_halves.py` (do not edit).
//! 5 networks (hidden {hidden}) trained on the true random noise (half the difference of even- and odd-frame half
//! stacks) of {len(SIM)} simulated Jupiter stacks, fine and medium detail only; checked on {len(REAL)} real stacks from
//! {len(set(r['group'] for r in REAL))} recordings it never saw (no calibration needed).
#![allow(clippy::excessive_precision)]
use crate::curve_net::Net;
use crate::noise_net::NoiseModel;

pub(crate) static JUPITER: NoiseModel = NoiseModel {{
    inputs: {len(sc.mu)},
    hidden: {hidden},
    outputs: {2 * NB},
    mean: {fl(sc.mu)},
    scale: {fl(sc.sd)},
    z_min: {fl(sc.zmin)},
    z_max: {fl(sc.zmax)},
    nets: &[
{nets}
    ],
    calibration: {fl(np.zeros(2 * NB))},
}};
"""
    open(opt("--export"), "w").write(src)
    print("esportata in", opt("--export"))
