#!/usr/bin/env python3
"""How different are Jupiter and Saturn, for the sharpening curve and for the noise?

usage: .venv/bin/python saturn_vs_giove.py SATURNO_DIR GIOVE_DIR GIOVE_V3_DIR

Sharpening:
- Jupiter's network (in the app) on Saturn's stacks, measured with Saturn's globe as its disk,
  scored on Saturn's own image error (globe and rings) next to Saturn's networks (never trained
  on the stack's original) and the best curve;
- the best curves' shapes: log gain by band over the peak's, averaged by seeing (D/r0) for the
  two planets (Jupiter: dataset_giove valutazione2; Saturn: the globe).

Noise (fine and medium grain, colour; the same features and targets as Jupiter's estimator,
measured on Saturn's globe): trained on Jupiter only, Saturn only, or both, scored by leave one
original out on each planet (share of stacks within ±25 % of the true noise).
"""
import glob, json, math, os, sys
import warnings
import numpy as np

warnings.filterwarnings("ignore")
S, G, G3 = sys.argv[1:4]
F_LO, F_HI, BINS = 0.012, 0.5, 22
centre = lambda k: math.exp(math.log(F_LO) + (k - 0.5) / BINS * (math.log(F_HI) - math.log(F_LO))) if 1 <= k <= BINS else 0.6


def err(g, q):
    M, R, TT, TV = np.array(q["m"]).reshape(24, 24), np.array(q["r"]), q["tt"], q["tvar"]
    rg, gmg = R @ g, g @ M @ g
    return math.sqrt(max(TT - rg * rg / max(gmg, 1e-30), 0.0) / TV)


def best(q):
    M, R = np.array(q["m"]).reshape(24, 24), np.array(q["r"])
    lam = 1e-6 * np.trace(M) / 24
    g = np.linalg.solve(M + lam * np.eye(24), R + lam)
    return g / g[0]


# ---------------------------------------------------------------- sharpening
plan_s = {l.split("\t")[0]: l.split("\t") for l in open(os.path.join(S, "piano.tsv")).read().splitlines()[1:]}
jov = {r["nome"]: np.array(r["giove"]) for r in json.load(open(os.path.join(S, "confronto_giove", "giove_su_saturno.json")))}
res = {r["nome"]: r for r in json.load(open(os.path.join(S, "risultati.json")))}
rows = []
for name, g in jov.items():
    d = json.load(open(os.path.join(S, "valutazione", name + ".json")))
    r = res[name]
    rows.append({
        "globo": (d["errore_grezza"][0], err(g, d["quad_pianeta"]), r["pianeta"]["doppia"], r["pianeta"]["ottima_doppia"], d["errore_app"][0]),
        "anelli": (d["errore_grezza"][1], err(g, d["quad_anelli"]), r["anelli"]["doppia"], r["anelli"]["ottima_doppia"], d["errore_app"][1]),
    })
print(f"NITIDEZZA, {len(rows)} stack di Saturno (punteggio: 1 = curva migliore, 0 = grezza; mediana)")
for z in ("globo", "anelli"):
    sc = lambda i: np.median([(r[z][0] - r[z][i]) / max(r[z][0] - r[z][3], 1e-9) for r in rows])
    print(f"  {z:6}: app standard {sc(4):.2f}  rete di Giove {sc(1):.2f}  rete di Saturno {sc(2):.2f}   (Saturno meglio in {sum(r[z][2] < r[z][1] for r in rows)}/{len(rows)})")

# Shapes of the best curves by seeing.
def shapes(pairs):
    out = {}
    for dr0, g in pairs:
        b = "<2" if dr0 < 2 else ("2-4" if dr0 < 4 else ">4")
        lg = np.log(np.clip(g[1:23], 0.05, 200))
        out.setdefault(b, []).append(lg - lg.max())
    return {b: (np.mean(v, 0), len(v)) for b, v in out.items()}

sat_pairs = [(float(plan_s[n[:4]][8]), best(json.load(open(os.path.join(S, "valutazione", n + ".json")))["quad_pianeta"])) for n in jov]
plan_g = {l.split("\t")[0]: l.split("\t") for l in open(os.path.join(G, "piano.tsv")).read().splitlines()[1:]}
jup_pairs = []
for q in glob.glob(os.path.join(G, "valutazione2", "*", "quad", "*.json")):
    n = os.path.basename(q)[:4]
    if n in plan_g:
        jup_pairs.append((float(plan_g[n][8]), best(json.load(open(q)))))
ss, js = shapes(sat_pairs), shapes(jup_pairs)
print("\nForma della curva ottima (log guadagno sotto il picco; periodo del picco; differenza media tra i due pianeti):")
out_shapes = {}
for b in ("<2", "2-4", ">4"):
    if b in ss and b in js:
        (a, na), (c, nc) = ss[b], js[b]
        pa, pc = int(np.argmax(a)) + 1, int(np.argmax(c)) + 1
        diff = np.mean(np.abs(np.exp(a) - np.exp(c)))
        print(f"  D/r0 {b:4}: Saturno n={na:3} picco a {1 / centre(pa):.1f} px | Giove n={nc:3} picco a {1 / centre(pc):.1f} px | differenza media dei guadagni relativi {diff:.2f}")
        out_shapes[b] = {"periodi": [1 / centre(k) for k in range(1, 23)], "saturno": np.exp(a).tolist(), "giove": np.exp(c).tolist(), "n": [na, nc]}
json.dump(out_shapes, open(os.path.join(S, "confronto_giove", "forme.json"), "w"))

# ---------------------------------------------------------------- noise
BANDS = list(range(2, 24))
USED = np.array([centre(k) >= 0.07 for k in BANDS])
FINE = np.array([centre(k) >= 0.2 for k in BANDS])


def load(path, plan, planet):
    rows = []
    for r in json.load(open(path)):
        y = np.full(44, np.nan)
        y[:22] = np.array(r["rumore_grana"])[2:24]
        if r["rumore_colore"]:
            y[22:] = np.array(r["rumore_colore"])[2:24]
        y[np.concatenate([~USED, ~USED])] = np.nan
        y[y < -13] = np.nan
        rows.append({"group": planet + ":" + plan[r["stack"][:4]][2], "planet": planet, "x": np.array(r["x"]), "xb": np.array(r["x_nero"] or r["x"]), "y": y})
    return rows


plan_g3 = {}
for p in glob.glob(os.path.join(G3, "piano*.tsv")):
    plan_g3.update({l.split("\t")[0]: l.split("\t") for l in open(p).read().splitlines()[1:]})
JUP = load(os.path.join(G3, "rumore_mezzi.json"), plan_g3, "giove")
SAT = load(os.path.join(S, "confronto_giove", "rumore_mezzi_saturno.json"), plan_s, "saturno")
print(f"\nRUMORE: {len(JUP)} stack di Giove, {len(SAT)} di Saturno (globo), stessi ingressi e bersagli")


class Scaler:
    def __init__(self, A):
        self.mu, self.sd = A.mean(0), A.std(0)
        self.sd[self.sd < 1e-6] = 1.0
        Z = (A - self.mu) / self.sd
        self.zmin, self.zmax = Z.min(0), Z.max(0)

    def __call__(self, x):
        return np.clip((x - self.mu) / self.sd, self.zmin - 0.5, self.zmax + 0.5)


def fit(rows, hidden=32, decay=1e-3, epochs=2000, seed=0):
    X = np.array([r["x"] for r in rows] + [r["xb"] for r in rows])
    Y = np.array([r["y"] for r in rows] * 2)
    rng = np.random.default_rng(seed)
    sc = Scaler(X)
    Z = sc(X)
    mask = np.isfinite(Y)
    T = np.where(mask, Y, 0.0)
    ym = np.nanmean(np.where(mask, Y, np.nan), 0)
    ym = np.where(np.isfinite(ym), ym, -9.0)
    p = Z.shape[1]
    W1 = rng.normal(0, 1 / math.sqrt(p), (p, hidden)); b1 = np.zeros(hidden)
    W2 = rng.normal(0, 0.01, (hidden, 44)); b2 = ym.copy()
    params = [W1, b1, W2, b2]
    mom = [np.zeros_like(q) for q in params]; vel = [np.zeros_like(q) for q in params]
    n = mask.sum()
    for it in range(1, epochs + 1):
        H = np.tanh(Z @ W1 + b1)
        P = H @ W2 + b2
        dP = 2 * (P - T) * mask / n
        grads = [Z.T @ ((dP @ W2.T) * (1 - H * H)) + decay * W1, ((dP @ W2.T) * (1 - H * H)).sum(0), H.T @ dP + decay * W2, dP.sum(0)]
        for q, g, m_, v_ in zip(params, grads, mom, vel):
            m_ *= 0.9; m_ += 0.1 * g
            v_ *= 0.999; v_ += 0.001 * g * g
            q -= 0.01 * (m_ / (1 - 0.9 ** it)) / (np.sqrt(v_ / (1 - 0.999 ** it)) + 1e-8)
    return lambda x: np.tanh(sc(x) @ W1 + b1) @ W2 + b2


def ensemble(rows, n=3):
    nets = [fit(rows, seed=s) for s in range(n)]
    return lambda x: np.mean([m(x) for m in nets], 0)


def level(y, sel):
    v = y[sel]
    v = v[np.isfinite(v)]
    return math.sqrt(np.sum(10 ** v)) if len(v) else float("nan")


SEL = {"fine": np.concatenate([FINE, np.zeros(22, bool)]), "media": np.concatenate([USED & ~FINE, np.zeros(22, bool)]), "colore": np.concatenate([np.zeros(22, bool), USED])}


def score(train_of, test):
    """Leave one original out on `test`; `train_of(group)` gives the training rows."""
    hits = {k: [] for k in SEL}
    for g in sorted(set(r["group"] for r in test)):
        f = ensemble(train_of(g))
        for r in (r for r in test if r["group"] == g):
            p = f(r["x"])
            for k, sel in SEL.items():
                t = level(r["y"], sel)
                if t == t:
                    hits[k].append(abs(level(np.where(np.isfinite(r["y"]), p, np.nan), sel) / t - 1) <= 0.25)
    return {k: float(np.mean(v)) for k, v in hits.items()}


out = {}
for test, name in ((SAT, "Saturno"), (JUP, "Giove")):
    same = [r for r in (SAT if test is SAT else JUP)]
    other = JUP if test is SAT else SAT
    a = score(lambda g: [r for r in same if r["group"] != g], test)
    b = score(lambda g: other, test)
    c = score(lambda g: [r for r in same if r["group"] != g] + other, test)
    out[name] = {"stesso": a, "altro": b, "insieme": c}
    print(f"  su {name}: entro ±25 % (fine / media / colore)")
    for label, v in (("solo " + name, a), ("solo l'altro pianeta", b), ("tutti e due", c)):
        print(f"    {label:22} {v['fine']:.2f} / {v['media']:.2f} / {v['colore']:.2f}")
json.dump(out, open(os.path.join(S, "confronto_giove", "rumore.json"), "w"))
