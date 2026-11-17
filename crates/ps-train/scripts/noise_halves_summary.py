#!/usr/bin/env python3
"""Summary of the half-stack noise (noise_halves) of simulated stacks against real ones.

usage: .venv/bin/python noise_halves_summary.py SIM.json PIANO REAL.json MODEL.json NOISE_DATA.rs

Prints, as % of the disk's mean level: grain (fine, medium, large) and colour noise of the
simulations and of the real stacks, the noise's shape (medium/fine, large/fine, colour/grain),
how it follows the number of frames by gain, and the calibrated noise network's estimate over
the true noise.
"""
import json, math, re, sys
import numpy as np

sim_path, plan_path, real_path, model_path, rs_path = sys.argv[1:6]
plan = {l.split("\t")[0]: l.split("\t") for l in open(plan_path).read().splitlines()[1:]}
M = json.load(open(model_path))
SC = M["scale_bande"]
NB = len(SC)
mu, sd, zmin, zmax = (np.array(M[k]) for k in ("mu", "sd", "zmin", "zmax"))
cal = np.array([float(v) for v in re.search(r"calibration: &\[(.*?)\]", open(rs_path).read(), re.S).group(1).split(",")])


def pred(x):
    z = np.clip((np.array(x) - mu) / sd, zmin - 0.5, zmax + 0.5)
    with np.errstate(all="ignore"):
        return np.mean([np.tanh(z @ np.array(a) + np.array(b)) @ np.array(c) + np.array(d) for a, b, c, d in M["nets"]], 0) + cal


def tot(v, s):
    return 100 * math.sqrt(sum(10 ** v[j] for j in range(NB) if SC[j] == s))


def rows(path, real):
    out = []
    for r in json.load(open(path)):
        tl = np.array(r["rumore_grana"])[2:24]
        tc = np.array(r["rumore_colore"])[2:24] if r["rumore_colore"] else None
        p = pred(r["x"])
        e = {"nome": r["stack"]}
        for s in ("fine", "media", "grande"):
            e["g_" + s] = tot(tl, s)
            e["c_" + s] = tot(tc, s) if tc is not None else None
            e["pg_" + s] = tot(p[:NB], s)
            e["pc_" + s] = tot(p[NB:], s) if tc is not None else None
        if real:
            e.update(tenuti=r["meta"]["tenuti"], gain="vero", camera="vera")
        else:
            q = plan[r["stack"][:4]]
            e.update(tenuti=int(round(int(q[11]) * float(q[12]))), gain=q[9], camera=q[5])
        out.append(e)
    return out


sim, real = rows(sim_path, False), rows(real_path, True)
med = lambda v: float(np.median(v))
print(f"simulazioni {len(sim)}, riprese vere {len(real)}")
for k in ("g_fine", "g_media", "g_grande", "c_fine", "c_media"):
    a = [e[k] for e in sim if e[k] is not None]
    b = [e[k] for e in real if e[k] is not None]
    print(f"  {k:8} sim {med(a):.3f} ({min(a):.3f}–{max(a):.3f})   vere {med(b):.3f} ({min(b):.3f}–{max(b):.3f})")
for name, f in (("media/fine", lambda e: e["g_media"] / e["g_fine"]), ("grande/fine", lambda e: e["g_grande"] / e["g_fine"]), ("colore/grana", lambda e: e["c_fine"] / e["g_fine"] if e["c_fine"] else None)):
    a = [f(e) for e in sim if f(e) is not None]
    b = [f(e) for e in real if f(e) is not None]
    print(f"  forma {name:13} sim {med(a):.2f}   vere {med(b):.2f}")
for cam in ("imx462c", "asi678mc", "imx462m"):
    for g in ("basso", "medio", "alto"):
        sel = [e for e in sim if e["gain"] == g and e["camera"] == cam]
        if len(sel) >= 4:
            x, y = np.log([e["tenuti"] for e in sel]), np.log([e["g_fine"] for e in sel])
            print(f"  {cam:8} gain {g:5} n={len(sel):2} grana fine {med([e['g_fine'] for e in sel]):.3f}%  pendenza con i fotogrammi {np.polyfit(x, y, 1)[0]:+.2f} (attesa −0,5)")
print("  rete tarata, stima/vero:", {k: round(med([e["p" + k] / e[k] for e in sim if e[k]]), 2) for k in ("g_fine", "g_media", "c_fine", "c_media")})
