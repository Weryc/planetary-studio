#!/usr/bin/env python3
"""Saturn's sharpening curves and noise estimators: two networks (globe, rings) against one for
both, trained on the simulated stacks measured by `saturn_eval`.

usage: .venv/bin/python saturn_train.py PIANO EVAL_DIR [--preds preds.json] [--out risultati.json]
                                        [--export crates/ps-core/src/saturn_data.rs]

Sharpening. Target: the true image error of a band curve g over each zone, from the quadratic
forms saved by saturn_eval: err(g) = sqrt((tt − (rᵀg)²/gᵀMg) / tvar) (best scale per zone).
- one network for both zones: trained on err_globe² + err_rings²;
- two networks: each trained on its own zone's error only;
both see the same inputs (ps_core::saturn::features), averages of 5 small networks (tanh 24),
each stack also with its sky clipped to black. Ceilings: the best single curve for both zones
and the best curve of each zone.

Noise. Target: the true random noise of each zone (half the difference of the even and odd
half stacks), log10 power over the globe's mean squared, bands 2‥23 of brightness and colour,
only fine and medium detail (centre ≥ 0.07 cycles/px). One network trained on both zones
together ("insieme") against one network per zone; inputs ps_core::saturn::noise_features.

Scores are leave-one-original-out: each stack is predicted by networks that never saw its
original. --preds writes those predictions for `saturn_eval --apply` (the in-app check);
--export trains the two zone networks of each kind on everything and writes them for ps-core.
"""
import glob, json, math, os, sys
import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*encountered in matmul")  # spurious on macOS Accelerate

args = sys.argv[1:]
plan_path, eval_dir = args[0], args[1]
opt = lambda k: args[args.index(k) + 1] if k in args else None
# Several plans can be given, comma-separated (piano.tsv,piano_nuovi.tsv).
plan = {l.split("\t")[0]: l.split("\t") for pp in plan_path.split(",") for l in open(pp).read().splitlines()[1:]}

D = []
for p in sorted(glob.glob(os.path.join(eval_dir, "[gn][0-9][0-9][0-9]_*.json"))):
    if os.path.basename(p)[:4] not in plan:
        continue
    d = json.load(open(p))
    d["id"] = d["nome"][:4]
    d["group"] = plan[d["id"]][2]
    D.append(d)
N = len(D)
groups = sorted(set(d["group"] for d in D))
print(f"{N} stack di Saturno da {len(groups)} originali")

ZONES = ("pianeta", "anelli")
Q = {z: (np.array([np.array(d[f"quad_{z}"]["m"]).reshape(24, 24) for d in D]), np.array([d[f"quad_{z}"]["r"] for d in D]),
         np.array([d[f"quad_{z}"]["tt"] for d in D]), np.array([d[f"quad_{z}"]["tvar"] for d in D])) for z in ZONES}
X = np.array([d["x"] for d in D], float)
XB = np.array([d["x_nero"] for d in D], float)
XN = np.array([d["xn"] for d in D], float)
XNB = np.array([d["xn_nero"] for d in D], float)
NB = 23


def err(g, k, z):
    M, R, TT, TV = Q[z]
    rg, gmg = R[k] @ g, g @ M[k] @ g
    return math.sqrt(max(TT[k] - rg * rg / max(gmg, 1e-30), 0.0) / TV[k])


def solve(m, r):
    lam = 1e-6 * np.trace(m) / 24
    g = np.linalg.solve(m + lam * np.eye(24), r + lam * 1.0)
    return g / g[0]


def best_zone(k, z):
    M, R, _, _ = Q[z]
    return solve(M[k], R[k])


def best_both(k):
    # One curve for both zones, minimising the sum of the two normalised errors: start from the
    # weighted normal equations, then a few Gauss–Newton-free refinements by direct search.
    Mp, Rp, _, TVp = Q["pianeta"]
    Mr, Rr, _, TVr = Q["anelli"]
    g = solve(Mp[k] / TVp[k] + Mr[k] / TVr[k], Rp[k] / TVp[k] + Rr[k] / TVr[k])
    from_g = lambda g: err(g, k, "pianeta") ** 2 + err(g, k, "anelli") ** 2
    lg = np.log(np.clip(g[1:], 0.05, 200))
    best = from_g(g)
    step = 0.2
    for _ in range(60):
        improved = False
        for j in range(NB):
            for s in (step, -step):
                t = lg.copy(); t[j] += s
                c = from_g(np.concatenate([[1.0], np.exp(t)]))
                if c < best:
                    best, lg, improved = c, t, True
        if not improved:
            step /= 2
            if step < 0.01:
                break
    return np.concatenate([[1.0], np.exp(lg)])


class Scaler:
    def __init__(self, A):
        self.mu, self.sd = A.mean(0), A.std(0)
        self.sd[self.sd < 1e-6] = 1.0
        Z = (A - self.mu) / self.sd
        self.zmin, self.zmax = Z.min(0), Z.max(0)

    def __call__(self, x):
        return np.clip((x - self.mu) / self.sd, self.zmin - 0.5, self.zmax + 0.5)


def adam(params, grads, state, it, lr=0.01):
    for q, g, m_, v_ in zip(params, grads, state[0], state[1]):
        m_ *= 0.9; m_ += 0.1 * g
        v_ *= 0.999; v_ += 0.001 * g * g
        q -= lr * (m_ / (1 - 0.9 ** it)) / (np.sqrt(v_ / (1 - 0.999 ** it)) + 1e-8)


def fit_curve(tr, zones, hidden=24, decay=3e-3, epochs=1500, seed=0):
    """A network of band log-gains trained on the summed normalised squared error of `zones`."""
    rng = np.random.default_rng(seed)
    k = np.concatenate([tr, tr])
    Xtr = np.concatenate([X[tr], XB[tr]])
    sc = Scaler(Xtr)
    Z = sc(Xtr)
    y0 = np.mean([np.log(np.clip(best_zone(i, zones[0])[1:], 0.1, 100)) for i in tr], 0)
    p = Z.shape[1]
    W1 = rng.normal(0, 1 / math.sqrt(p), (p, hidden)); b1 = np.zeros(hidden)
    W2 = rng.normal(0, 0.01, (hidden, NB)); b2 = y0.copy()
    params = [W1, b1, W2, b2]
    state = ([np.zeros_like(q) for q in params], [np.zeros_like(q) for q in params])
    n = len(k)
    for it in range(1, epochs + 1):
        H = np.tanh(Z @ W1 + b1)
        L = H @ W2 + b2
        G = np.concatenate([np.ones((n, 1)), np.exp(np.clip(L, -3, 5))], 1)
        dG = np.zeros_like(G)
        for z in zones:
            M, R, _, TV = Q[z]
            Mt, Rt, TVt = M[k], R[k], TV[k]
            rg = np.einsum("nk,nk->n", Rt, G)
            MG = np.einsum("nab,nb->na", Mt, G)
            gmg = np.einsum("nk,nk->n", G, MG)
            dG += ((-2 * rg / gmg)[:, None] * Rt + (rg * rg / gmg ** 2)[:, None] * 2 * MG) / (TVt[:, None] * n)
        dL = dG[:, 1:] * G[:, 1:] * ((L > -3) & (L < 5))
        gW2 = H.T @ dL + decay * W2; gb2 = dL.sum(0)
        dH = (dL @ W2.T) * (1 - H * H)
        gW1 = Z.T @ dH + decay * W1; gb1 = dH.sum(0)
        adam(params, [gW1, gb1, gW2, gb2], state, it)

    def predict(x):
        h = np.tanh(sc(x) @ W1 + b1)
        return np.concatenate([[1.0], np.exp(np.clip(h @ W2 + b2, -3, 5))])
    predict.parts = (sc, W1, b1, W2, b2)
    return predict


def ensemble(fit, n=5):
    nets = [fit(seed=s) for s in range(n)]
    f = lambda x: np.exp(np.mean([np.log(m(x)) for m in nets], 0))
    f.nets = nets
    return f


# ------------------------------------------------------------------ noise
CENTRE = lambda k: math.exp(math.log(0.012) + (k - 0.5) / 22 * (math.log(0.5) - math.log(0.012))) if k <= 22 else 0.6
BANDS = list(range(2, 24))
USED = np.array([CENTRE(k) >= 0.07 for k in BANDS])
FINE = np.array([CENTRE(k) >= 0.2 for k in BANDS])
MEDIUM = USED & ~FINE


def noise_target(d, zone):
    z = d["rumore"][zone]
    y = np.full(44, np.nan)
    y[:22] = np.array(z["grana"])[2:24]
    if z["colore"]:
        y[22:] = np.array(z["colore"])[2:24]
    y[np.concatenate([~USED, ~USED])] = np.nan
    y[y < -13] = np.nan
    return y


YN = {z: np.array([noise_target(d, z) for d in D]) for z in ("pianeta", "anelli", "insieme")}


def fit_noise(tr, zone, hidden=32, decay=1e-3, epochs=3000, seed=0):
    rng = np.random.default_rng(seed)
    Xtr = np.concatenate([XN[tr], XNB[tr]])
    Y = np.concatenate([YN[zone][tr]] * 2)
    sc = Scaler(Xtr)
    Z = sc(Xtr)
    mask = np.isfinite(Y)
    T = np.where(mask, Y, 0.0)
    ym = np.nanmean(np.where(mask, Y, np.nan), 0)
    ym = np.where(np.isfinite(ym), ym, -9.0)
    p = Z.shape[1]
    W1 = rng.normal(0, 1 / math.sqrt(p), (p, hidden)); b1 = np.zeros(hidden)
    W2 = rng.normal(0, 0.01, (hidden, 44)); b2 = ym.copy()
    params = [W1, b1, W2, b2]
    state = ([np.zeros_like(q) for q in params], [np.zeros_like(q) for q in params])
    n = mask.sum()
    for it in range(1, epochs + 1):
        H = np.tanh(Z @ W1 + b1)
        P = H @ W2 + b2
        dP = 2 * (P - T) * mask / n
        gW2 = H.T @ dP + decay * W2; gb2 = dP.sum(0)
        dH = (dP @ W2.T) * (1 - H * H)
        gW1 = Z.T @ dH + decay * W1; gb1 = dH.sum(0)
        adam(params, [gW1, gb1, gW2, gb2], state, it)
    f = lambda x: np.tanh(sc(x) @ W1 + b1) @ W2 + b2
    f.parts = (sc, W1, b1, W2, b2)
    return f


def mean_ensemble(fit, n=5):
    nets = [fit(seed=s) for s in range(n)]
    f = lambda x: np.mean([m(x) for m in nets], 0)
    f.nets = nets
    return f


def level(y, sel):
    """Noise amplitude (RMS) over the bands `sel` of a log10-power vector (NaN-safe)."""
    v = y[sel]
    v = v[np.isfinite(v)]
    return math.sqrt(np.sum(10 ** v)) if len(v) else float("nan")


# ------------------------------------------------------------------ leave one original out
everything = np.arange(N)
P = {key: {} for key in ("unica", "pianeta", "anelli", "rumore_unica", "rumore_pianeta", "rumore_anelli")}
for g in groups:
    tr = np.array([k for k in everything if D[k]["group"] != g])
    te = [k for k in everything if D[k]["group"] == g]
    nets = {
        "unica": ensemble(lambda seed: fit_curve(tr, ZONES, seed=seed)),
        "pianeta": ensemble(lambda seed: fit_curve(tr, ("pianeta",), seed=seed)),
        "anelli": ensemble(lambda seed: fit_curve(tr, ("anelli",), seed=seed)),
    }
    noise = {
        "rumore_unica": mean_ensemble(lambda seed: fit_noise(tr, "insieme", seed=seed)),
        "rumore_pianeta": mean_ensemble(lambda seed: fit_noise(tr, "pianeta", seed=seed)),
        "rumore_anelli": mean_ensemble(lambda seed: fit_noise(tr, "anelli", seed=seed)),
    }
    for k in te:
        for key, f in nets.items():
            P[key][k] = f(X[k])
        for key, f in noise.items():
            P[key][k] = f(XN[k])
    print(f"  lasciato fuori {g} ({len(te)}): fatto", flush=True)

res = []
for k in everything:
    d = D[k]
    e = {"nome": d["nome"], "id": d["id"], "group": d["group"], "split": plan[d["id"]][1], "scope": plan[d["id"]][4], "dr0": float(plan[d["id"]][8])}
    bb = best_both(k)
    for i, z in enumerate(ZONES):
        e[z] = {
            "grezza": d["errore_grezza"][i], "app": d["errore_app"][i],
            "unica": err(P["unica"][k], k, z), "doppia": err(P[z][k], k, z),
            "ottima_unica": err(bb, k, z), "ottima_doppia": err(best_zone(k, z), k, z),
        }
    # Noise: predicted against true amplitude, fine grain, medium grain, colour, per zone.
    for z in ZONES:
        t = YN[z][k]
        ps, pd = P["rumore_unica"][k], P["rumore_" + z][k]
        e["rumore_" + z] = {}
        for name, sel in (("fine", np.concatenate([FINE, np.zeros(22, bool)])), ("media", np.concatenate([MEDIUM, np.zeros(22, bool)])), ("colore", np.concatenate([np.zeros(22, bool), USED]))):
            tv = level(t, sel)
            e["rumore_" + z][name] = {"vero": tv, "unica": level(np.where(np.isfinite(t), ps, np.nan), sel) / tv if tv == tv else None,
                                      "doppia": level(np.where(np.isfinite(t), pd, np.nan), sel) / tv if tv == tv else None}
    e["curve"] = {key: P[key][k].tolist() for key in ("unica", "pianeta", "anelli")}
    e["curve"]["ottima_pianeta"] = best_zone(k, "pianeta").tolist()
    e["curve"]["ottima_anelli"] = best_zone(k, "anelli").tolist()
    e["curve"]["ottima_unica"] = bb.tolist()
    res.append(e)


def score(e, z, key):
    """Share of the way from the raw to the best per-zone curve: 1 = as the best, 0 = as the raw."""
    r = e[z]
    return (r["grezza"] - r[key]) / max(r["grezza"] - r["ottima_doppia"], 1e-9)


print("\nNITIDEZZA (errore medio per zona, un originale fuori; punteggio = mediana della strada fatta verso l'ottima per zona)")
for z in ZONES:
    line = f"  {z:8}"
    for key in ("grezza", "app", "unica", "doppia", "ottima_unica", "ottima_doppia"):
        line += f"  {key} {np.mean([e[z][key] for e in res]):.4f}"
    print(line)
    print("           punteggi: " + "  ".join(f"{key} {np.median([score(e, z, key) for e in res]):.2f}" for key in ("app", "unica", "doppia", "ottima_unica")))
    print(f"           doppia meglio dell'unica in {sum(e[z]['doppia'] < e[z]['unica'] for e in res)}/{N}, peggio dell'app in {sum(e[z]['doppia'] > e[z]['app'] for e in res)}/{N}")

print("\nRUMORE (stima / vero, ampiezza; dentro ±25 %)")
for z in ZONES:
    for name in ("fine", "media", "colore"):
        vals = [e["rumore_" + z][name] for e in res if e["rumore_" + z][name]["unica"] is not None]
        if not vals:
            continue
        inside = lambda key: np.mean([abs(v[key] - 1) <= 0.25 for v in vals])
        med = lambda key: np.median([v[key] for v in vals])
        print(f"  {z:8} {name:7} n={len(vals):2}  unica ±25% {inside('unica'):.2f} (mediana ×{med('unica'):.2f})   doppia ±25% {inside('doppia'):.2f} (mediana ×{med('doppia'):.2f})")
ratio = [e["rumore_anelli"]["fine"]["vero"] / e["rumore_pianeta"]["fine"]["vero"] for e in res]
print(f"  grana fine vera anelli/globo: mediana {np.median(ratio):.2f} (da {min(ratio):.2f} a {max(ratio):.2f})")

if opt("--out"):
    json.dump(res, open(opt("--out"), "w"))
if opt("--preds"):
    out = []
    for k in everything:
        out.append({"nome": D[k]["nome"], **{key: [float(v) for v in P[key][k]] for key in P}})
    json.dump(out, open(opt("--preds"), "w"))

if opt("--export"):
    fl = lambda a: "&[" + ", ".join(f"{float(v):.6}" for v in np.ravel(a)) + "]"

    def nets_src(f):
        out = []
        for m in f.nets:
            _, W1, b1, W2, b2 = m.parts
            out.append(f"        Net {{ w1: {fl(W1)}, b1: {fl(b1)}, w2: {fl(W2)}, b2: {fl(b2)} }},")
        return "\n".join(out)

    def curve_model(name, f, note):
        sc = f.nets[0].parts[0]
        return f"""pub(crate) static {name}: Model = Model {{
    inputs: {X.shape[1]},
    hidden: {f.nets[0].parts[1].shape[1]},
    mean: {fl(sc.mu)},
    scale: {fl(sc.sd)},
    z_min: {fl(sc.zmin)},
    z_max: {fl(sc.zmax)},
    nets: &[
{nets_src(f)}
    ],
    note: "{note}",
}};
"""

    def noise_model(name, f):
        sc = f.nets[0].parts[0]
        return f"""pub(crate) static {name}: NoiseModel = NoiseModel {{
    inputs: {XN.shape[1]},
    hidden: {f.nets[0].parts[1].shape[1]},
    outputs: 44,
    mean: {fl(sc.mu)},
    scale: {fl(sc.sd)},
    z_min: {fl(sc.zmin)},
    z_max: {fl(sc.zmax)},
    nets: &[
{nets_src(f)}
    ],
    calibration: {fl(np.zeros(44))},
}};
"""
    note = f"5 reti (24 neuroni) per zona su {N} stack simulati di Saturno da {len(groups)} originali"
    pl = ensemble(lambda seed: fit_curve(everything, ("pianeta",), seed=seed))
    # Rings: no network of their own. At 55 % (102 stacks), applied as the app applies them with
    # no edge protection, the globe's curve beat on the rings both the rings-only network and
    # the one trained on both zones (0.79 against 0.70 / 0.75), so the globe's goes everywhere.
    an = None
    npl = mean_ensemble(lambda seed: fit_noise(everything, "pianeta", seed=seed))
    nan_ = mean_ensemble(lambda seed: fit_noise(everything, "anelli", seed=seed))
    src = f"""//! Weights of the Saturn networks (globe and rings), written by `ps-train/scripts/saturn_train.py` (do not edit).
//! {note}; rumore: 5 reti (32 neuroni) per zona sul rumore vero dei mezzi stack.
#![allow(clippy::excessive_precision)]
use crate::curve_net::{{Model, Net}};
use crate::noise_net::NoiseModel;

/// The globe's sharpening network, and the rings' own (none: the globe's curve goes everywhere).
pub(crate) fn sharpen_models() -> Option<(&'static Model, Option<&'static Model>)> {{
    Some((&PLANET, None))
}}

/// The globe's and the rings' noise estimators.
pub(crate) fn noise_models() -> Option<(&'static NoiseModel, &'static NoiseModel)> {{
    Some((&NOISE_PLANET, &NOISE_RINGS))
}}

{curve_model("PLANET", pl, note + " (globo)")}
{noise_model("NOISE_PLANET", npl)}
{noise_model("NOISE_RINGS", nan_)}"""
    open(opt("--export"), "w").write(src)
    check = [{"x": X[k].tolist(), "pianeta": pl(X[k]).tolist(), "xn": XN[k].tolist(), "rumore_pianeta": npl(XN[k]).tolist()} for k in everything[:3]]
    json.dump(check, open(os.path.join(eval_dir, "reti_controllo.json"), "w"))
    print("esportate in", opt("--export"))

if "--curva" in args:
    # Learning curve: the same leave-one-original-out, with the networks trained on only a share
    # of the other originals (random, 3 draws). If the error still falls from 50 % to 100 % of the
    # originals, more data should still help.
    print("\nCURVA DI APPRENDIMENTO (errore medio di nitidezza per zona; rumore: grana fine entro ±25 %)")
    rng = np.random.default_rng(7)
    for share in (0.5, 0.75, 1.0):
        acc = {("unica", "pianeta"): [], ("unica", "anelli"): [], ("doppia", "pianeta"): [], ("doppia", "anelli"): []}
        nacc = {"unica": [], "doppia": []}
        draws = 1 if share == 1.0 else 3
        for _ in range(draws):
            for g in groups:
                others = [x for x in groups if x != g]
                keep = set(rng.choice(others, max(2, int(round(share * len(others)))), replace=False))
                tr = np.array([k for k in everything if D[k]["group"] in keep])
                te = [k for k in everything if D[k]["group"] == g]
                fs = ensemble(lambda seed: fit_curve(tr, ZONES, seed=seed), n=3)
                fp = ensemble(lambda seed: fit_curve(tr, ("pianeta",), seed=seed), n=3)
                fa = ensemble(lambda seed: fit_curve(tr, ("anelli",), seed=seed), n=3)
                ns = mean_ensemble(lambda seed: fit_noise(tr, "insieme", seed=seed), n=3)
                npl = mean_ensemble(lambda seed: fit_noise(tr, "pianeta", seed=seed), n=3)
                nan_ = mean_ensemble(lambda seed: fit_noise(tr, "anelli", seed=seed), n=3)
                fine = np.concatenate([FINE, np.zeros(22, bool)])
                for k in te:
                    acc[("unica", "pianeta")].append(err(fs(X[k]), k, "pianeta"))
                    acc[("unica", "anelli")].append(err(fs(X[k]), k, "anelli"))
                    acc[("doppia", "pianeta")].append(err(fp(X[k]), k, "pianeta"))
                    acc[("doppia", "anelli")].append(err(fa(X[k]), k, "anelli"))
                    for z, nd in (("pianeta", npl), ("anelli", nan_)):
                        t = YN[z][k]
                        tv = level(t, fine)
                        for key, f in (("unica", ns), ("doppia", nd)):
                            nacc[key].append(abs(level(np.where(np.isfinite(t), f(XN[k]), np.nan), fine) / tv - 1) <= 0.25)
        n_orig = int(round(share * (len(groups) - 1)))
        print(f"  {share:.0%} degli altri originali (~{n_orig}): " + "  ".join(f"{a}/{z} {np.mean(v):.4f}" for (a, z), v in acc.items())
              + f"  | rumore ±25% unica {np.mean(nacc['unica']):.2f} doppia {np.mean(nacc['doppia']):.2f}", flush=True)


if opt("--nuovi"):
    # Do new originals add something? (A) the networks trained on the old originals only, on the
    # new ones' stacks: if they do much worse there than on the old (each judged without its own
    # original), the new ones are different. (B) on the old originals, leave one out, trained
    # with and without the new stacks.
    new = set(opt("--nuovi").split(","))
    old_k = np.array([k for k in everything if D[k]["group"] not in new])
    new_k = [k for k in everything if D[k]["group"] in new]
    fine = np.concatenate([FINE, np.zeros(22, bool)])

    def noise_hit(f, k, z):
        t = YN[z][k]
        tv = level(t, fine)
        return abs(level(np.where(np.isfinite(t), f(XN[k]), np.nan), fine) / tv - 1) <= 0.25

    def sc(k, z, e):
        r = Q[z]
        return (D[k]["errore_grezza"][ZONES.index(z)] - e) / max(D[k]["errore_grezza"][ZONES.index(z)] - err(best_zone(k, z), k, z), 1e-9)

    print("\nORIGINALI NUOVI:", ", ".join(sorted(new)), f"({len(new_k)} stack)")
    fp = ensemble(lambda seed: fit_curve(old_k, ("pianeta",), seed=seed))
    npl = mean_ensemble(lambda seed: fit_noise(old_k, "pianeta", seed=seed))
    nan_ = mean_ensemble(lambda seed: fit_noise(old_k, "anelli", seed=seed))
    a_glob = [sc(k, "pianeta", err(fp(X[k]), k, "pianeta")) for k in new_k]
    a_ring = [sc(k, "anelli", err(fp(X[k]), k, "anelli")) for k in new_k]
    a_noise = [noise_hit(npl, k, "pianeta") for k in new_k] + [noise_hit(nan_, k, "anelli") for k in new_k]
    print(f"  (A) reti dei vecchi sui nuovi: nitidezza globo {np.median(a_glob):.2f} anelli {np.median(a_ring):.2f} (curva del globo), rumore entro ±25% {np.mean(a_noise):.2f}")
    # (C) each new original by networks that saw the old ones and the other new ones: what the
    # new data add for images like them (the honest counterpart of A).
    c_glob, c_ring, c_noise = [], [], []
    for g in sorted(new):
        tr = np.array(list(old_k) + [k for k in new_k if D[k]["group"] != g])
        te = [k for k in new_k if D[k]["group"] == g]
        fp = ensemble(lambda seed: fit_curve(tr, ("pianeta",), seed=seed))
        npl = mean_ensemble(lambda seed: fit_noise(tr, "pianeta", seed=seed))
        nan_ = mean_ensemble(lambda seed: fit_noise(tr, "anelli", seed=seed))
        c_glob += [sc(k, "pianeta", err(fp(X[k]), k, "pianeta")) for k in te]
        c_ring += [sc(k, "anelli", err(fp(X[k]), k, "anelli")) for k in te]
        c_noise += [noise_hit(npl, k, "pianeta") for k in te] + [noise_hit(nan_, k, "anelli") for k in te]
    better = sum(c < a for c, a in zip(c_glob, a_glob))
    print(f"  (C) reti con gli altri nuovi    : nitidezza globo {np.median(c_glob):.2f} anelli {np.median(c_ring):.2f}, rumore entro ±25% {np.mean(c_noise):.2f}  (globo peggio di A in {better}/{len(c_glob)})", flush=True)
    for label, extra in (("senza i nuovi", []), ("con i nuovi", new_k)):
        g_s, r_s, n_h = [], [], []
        for g in sorted(set(D[k]["group"] for k in old_k)):
            tr = np.array([k for k in old_k if D[k]["group"] != g] + list(extra))
            te = [k for k in old_k if D[k]["group"] == g]
            fp = ensemble(lambda seed: fit_curve(tr, ("pianeta",), seed=seed), n=3)
            npl = mean_ensemble(lambda seed: fit_noise(tr, "pianeta", seed=seed), n=3)
            nan_ = mean_ensemble(lambda seed: fit_noise(tr, "anelli", seed=seed), n=3)
            for k in te:
                g_s.append(sc(k, "pianeta", err(fp(X[k]), k, "pianeta")))
                r_s.append(sc(k, "anelli", err(fp(X[k]), k, "anelli")))
                n_h += [noise_hit(npl, k, "pianeta"), noise_hit(nan_, k, "anelli")]
        print(f"  (B) vecchi originali, {label:13}: nitidezza globo {np.median(g_s):.3f} anelli {np.median(r_s):.3f}, rumore entro ±25% {np.mean(n_h):.3f}", flush=True)
