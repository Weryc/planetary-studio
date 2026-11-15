#!/usr/bin/env python3
"""Train the per-subject sharpening network (ps_core::curve_net) on the simulated stacks.

usage: .venv/bin/python curve_train.py PIANO EVAL_DIR [--features caratteristiche.json]
                                       [--export crates/ps-core/src/curve_net_data.rs]
                                       [--logo-gains gains.tsv] [--out results.json]

Inputs: what `curve_features` computed with ps_core::curve_net::features, exactly as the app
does, on each stack as it is and with its sky clipped to black (both are used for training, so
the network does without the sky when an earlier step removed its noise).

Target: the true image error of a curve g over the subject, the quadratic form saved by
sim_eval (E(g) = gᵀMg − 2rᵀg + tt); with the best overall scale E*(g) = tt − (rᵀg)²/gᵀMg, as
sqrt(E*/tvar) comparable to sim_eval's "errore_regione" [raw, smart, best].

Model: an average of 5 small networks (one tanh hidden layer of 24, output the log gains of
bands 1‥23, band 0 = 1), trained on E* directly with weight decay, plus a ridge regression of
the best curves as a linear reference.

All originals are used (the validation ones too, now that the comparison is done). Scores are
leave-one-original-out over the 12 originals: each stack is predicted by networks that never
saw its original. Then the final networks are trained on everything and, with --export,
written as Rust for ps-core.
"""
import glob, json, math, os, sys
import numpy as np

args = sys.argv[1:]
plan_path, eval_dir = args[0], args[1]
opt = lambda k: args[args.index(k) + 1] if k in args else None
feat_path = opt("--features") or os.path.join(eval_dir, "caratteristiche.json")
plan = {}
for line in open(plan_path).read().splitlines()[1:]:
    f = line.split("\t")
    plan[f[0]] = {"split": f[1], "group": f[2], "night": f[6]}

FEAT = {c["nome"]: c for c in json.load(open(feat_path))}
rows = []
for path in sorted(glob.glob(os.path.join(eval_dir, "*", "metrics.json"))):
    for v in json.load(open(path))["versioni"]:
        q = os.path.join(os.path.dirname(path), "quad", v["nome"] + ".json")
        if os.path.exists(q) and FEAT.get(v["nome"], {}).get("normale", {}).get("x") is not None:
            rows.append((v, json.load(open(q))))
names = [v["nome"] for v, _ in rows]
ids = [n[:4] for n in names]
print(f"{len(rows)} grezze con misure e forma quadratica")
X = np.array([FEAT[n]["normale"]["x"] for n in names], float)
XB = np.array([(FEAT[n]["nero"]["x"] or FEAT[n]["normale"]["x"]) for n in names], float)
M = np.array([np.array(q["m"]).reshape(24, 24) for _, q in rows])
R = np.array([np.array(q["r"]) for _, q in rows])
TT = np.array([q["tt"] for _, q in rows])
TV = np.array([q["tvar"] for _, q in rows])
group = np.array([plan[i]["group"] for i in ids])
split = np.array([plan[i]["split"] for i in ids])
night = np.array([plan[i]["night"] for i in ids])
reg = np.array([v["errore_regione"] for v, _ in rows])  # raw, smart, best
kind = np.array(["standard" if v["analisi"]["taglio"] == 0 else ("cap" if v["nitidezza"]["forza"] >= 1.19 else "learned") for v, _ in rows])
dr0 = np.array([v["meta"]["atmosfera"]["d_r0"] for v, _ in rows])
best = []
for m, r in zip(M, R):
    lam = 1e-6 * np.trace(m) / 24
    g = np.linalg.solve(m + lam * np.eye(24), r + lam)
    best.append(g / g[0])
best = np.array(best)
NB = 23


def err(g, k):
    rg, gmg = R[k] @ g, g @ M[k] @ g
    return math.sqrt(max(TT[k] - rg * rg / max(gmg, 1e-30), 0.0) / TV[k])


class Scaler:
    def __init__(self, Xtr):
        self.mu, self.sd = Xtr.mean(0), Xtr.std(0)
        self.sd[self.sd < 1e-6] = 1.0
        Z = (Xtr - self.mu) / self.sd
        self.zmin, self.zmax = Z.min(0), Z.max(0)

    def __call__(self, x):
        return np.clip((x - self.mu) / self.sd, self.zmin - 0.5, self.zmax + 0.5)


def train_set(tr):
    """Rows `tr`, each as it is and with its sky clipped to black (same target)."""
    return np.concatenate([X[tr], XB[tr]]), np.concatenate([tr, tr])


def fit_ridge(tr, lam=10.0):
    Xtr, k = train_set(tr)
    sc = Scaler(Xtr)
    Z = sc(Xtr)
    Y = np.log(np.clip(best[k][:, 1:], 0.1, 100))
    ym = Y.mean(0)
    W = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (Y - ym))
    return lambda x: np.concatenate([[1.0], np.exp(np.clip(ym + sc(x) @ W, -3, 5))])


def fit_mlp(tr, hidden=24, decay=3e-3, epochs=1500, seed=0):
    rng = np.random.default_rng(seed)
    Xtr, k = train_set(tr)
    sc = Scaler(Xtr)
    Z = sc(Xtr)
    y0 = np.log(np.clip(best[k][:, 1:], 0.1, 100)).mean(0)
    p = Z.shape[1]
    W1 = rng.normal(0, 1 / math.sqrt(p), (p, hidden)); b1 = np.zeros(hidden)
    W2 = rng.normal(0, 0.01, (hidden, NB)); b2 = y0.copy()
    params = [W1, b1, W2, b2]
    mom = [np.zeros_like(q) for q in params]; vel = [np.zeros_like(q) for q in params]
    Mt, Rt, TVt = M[k], R[k], TV[k]
    n = len(k)
    lr, beta1, beta2 = 0.01, 0.9, 0.999
    for it in range(1, epochs + 1):
        H = np.tanh(Z @ W1 + b1)
        L = H @ W2 + b2
        G = np.concatenate([np.ones((n, 1)), np.exp(np.clip(L, -3, 5))], 1)
        rg = np.einsum("nk,nk->n", Rt, G)
        MG = np.einsum("nab,nb->na", Mt, G)
        gmg = np.einsum("nk,nk->n", G, MG)
        dG = (-2 * rg / gmg)[:, None] * Rt + (rg * rg / gmg ** 2)[:, None] * 2 * MG
        dG /= TVt[:, None] * n
        dL = dG[:, 1:] * G[:, 1:] * ((L > -3) & (L < 5))
        gW2 = H.T @ dL + decay * W2; gb2 = dL.sum(0)
        dH = (dL @ W2.T) * (1 - H * H)
        gW1 = Z.T @ dH + decay * W1; gb1 = dH.sum(0)
        for q, g, m_, v_ in zip(params, [gW1, gb1, gW2, gb2], mom, vel):
            m_ *= beta1; m_ += (1 - beta1) * g
            v_ *= beta2; v_ += (1 - beta2) * g * g
            q -= lr * (m_ / (1 - beta1 ** it)) / (np.sqrt(v_ / (1 - beta2 ** it)) + 1e-8)

    def predict(x):
        h = np.tanh(sc(x) @ W1 + b1)
        return np.concatenate([[1.0], np.exp(np.clip(h @ W2 + b2, -3, 5))])
    predict.parts = (sc, W1, b1, W2, b2)
    return predict


def ensemble(tr, n=5):
    nets = [fit_mlp(tr, seed=s) for s in range(n)]
    f = lambda x: np.exp(np.mean([np.log(m(x)) for m in nets], 0))
    f.nets = nets
    return f


groups = sorted(set(group))
everything = np.arange(len(rows))
P = {"ridge": {}, "mlp": {}, "mlp_nero": {}}
for g in groups:
    tr = np.array([k for k in everything if group[k] != g])
    fr, fm = fit_ridge(tr), ensemble(tr)
    for k in everything:
        if group[k] == g:
            P["ridge"][k], P["mlp"][k], P["mlp_nero"][k] = fr(X[k]), fm(X[k]), fm(XB[k])
    print(f"  lasciato fuori {g}: fatto", flush=True)

res = []
for k in everything:
    res.append({"nome": names[k], "id": ids[k], "split": str(split[k]), "group": str(group[k]), "night": str(night[k]), "kind": str(kind[k]), "dr0": float(dr0[k]),
                "grezza": float(reg[k][0]), "nitida": float(reg[k][1]), "ottima": err(best[k], k),
                "ridge": err(P["ridge"][k], k), "mlp": err(P["mlp"][k], k), "mlp_nero": err(P["mlp_nero"][k], k),
                "curva_ottima": best[k].tolist(), "curva_mlp": P["mlp"][k].tolist(), "curva_mlp_nero": P["mlp_nero"][k].tolist()})


def gap(e, key):
    return (e["grezza"] - e[key]) / max(e["grezza"] - e["ottima"], 1e-9)


def summary(sel, label):
    if not sel:
        return
    line = f"{label:26} n={len(sel):3}"
    for key in ("grezza", "nitida", "ridge", "mlp", "mlp_nero", "ottima"):
        line += f"  {key} {np.mean([e[key] for e in sel]):.4f}"
    line += f"  divario app {np.median([gap(e, 'nitida') for e in sel]):.2f} rete {np.median([gap(e, 'mlp') for e in sel]):.2f} rete/cielo nero {np.median([gap(e, 'mlp_nero') for e in sel]):.2f}"
    line += f"  rete<app {sum(e['mlp'] < e['nitida'] for e in sel)}  peggio>20% {sum(e['mlp'] > 1.2 * e['nitida'] for e in sel)}"
    print(line)


summary(res, "tutte (un originale fuori)")
for s in ("train", "val"):
    summary([e for e in res if e["split"] == s], "ex " + s)
for kd in ("standard", "learned", "cap"):
    summary([e for e in res if e["kind"] == kd], "curva app " + kd)
for lo, hi in ((0, 2), (2, 4), (4, 20)):
    summary([e for e in res if lo <= e["dr0"] < hi], f"D/r0 {lo}-{hi}")
print("casi peggiori per la rete:")
for e in sorted(res, key=lambda e: e["nitida"] - e["mlp"])[:5]:
    print(f"  {e['id']} {e['group']} D/r0 {e['dr0']:.2f} app {e['nitida']:.3f} rete {e['mlp']:.3f} ottima {e['ottima']:.3f}")

if opt("--logo-gains"):
    with open(opt("--logo-gains"), "w") as f:
        for e in res:
            f.write(e["nome"] + "\t" + "\t".join(f"{g:.5f}" for g in e["curva_mlp"]) + "\n")
if opt("--out"):
    json.dump(res, open(opt("--out"), "w"))

if opt("--export"):
    final = ensemble(everything)
    sc = final.nets[0].parts[0]
    fl = lambda a: "&[" + ", ".join(f"{float(v):.6}" for v in np.ravel(a)) + "]"
    nets = []
    for m in final.nets:
        _, W1, b1, W2, b2 = m.parts
        nets.append(f"        Net {{ w1: {fl(W1)}, b1: {fl(b1)}, w2: {fl(W2)}, b2: {fl(b2)} }},")
    note = (f"5 reti (24 neuroni) su {len(rows)} stack simulati di Giove da {len(groups)} originali, ognuno anche con il cielo tagliato a nero; "
            f"errore medio con un originale escluso {np.mean([e['mlp'] for e in res]):.3f} (app {np.mean([e['nitida'] for e in res]):.3f}, ottima {np.mean([e['ottima'] for e in res]):.3f})")
    src = f"""//! Weights of the per-subject sharpening networks, written by `ps-train/scripts/curve_train.py` (do not edit).
//! {note}.
#![allow(clippy::excessive_precision)]
use crate::curve_net::{{Model, Net}};

pub(crate) static JUPITER: Model = Model {{
    inputs: {X.shape[1]},
    hidden: {final.nets[0].parts[1].shape[1]},
    mean: {fl(sc.mu)},
    scale: {fl(sc.sd)},
    z_min: {fl(sc.zmin)},
    z_max: {fl(sc.zmax)},
    nets: &[
{chr(10).join(nets)}
    ],
    note: "{note}",
}};
"""
    open(opt("--export"), "w").write(src)
    # Inputs and outputs of the exported networks, to check the Rust side against.
    check = [{"x": X[k].tolist(), "guadagni": final(X[k]).tolist()} for k in everything[:5]]
    json.dump(check, open(os.path.join(eval_dir, "rete_controllo.json"), "w"))
    print("esportata in", opt("--export"))
