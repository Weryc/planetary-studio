#!/usr/bin/env python3
"""Write the noise estimator (from noise_train.py --model) as Rust for ps-core, with a
calibration on real stacks: for each band, the median gap (log10) between the real noise
measured from half stacks (noise_halves) and the network's estimate on those stacks.

usage: .venv/bin/python noise_export.py modello.json rumore_mezzi.json out.rs

Provisional: the network was trained on simulated targets that counted the stack's fixed
errors as noise; the calibration brings its answers down to the random noise real recordings
show, until it is retrained on simulated half stacks.
"""
import json, sys
import numpy as np

model_path, halves_path, out = sys.argv[1:4]
M = json.load(open(model_path))
H = json.load(open(halves_path))
mu, sd, zmin, zmax = (np.array(M[k]) for k in ("mu", "sd", "zmin", "zmax"))
nets = [[np.array(q) for q in n] for n in M["nets"]]
NB = len(M["bande"])


def predict(x):
    z = np.clip((np.array(x) - mu) / sd, zmin - 0.5, zmax + 0.5)
    return np.mean([np.tanh(z @ W1 + b1) @ W2 + b2 for W1, b1, W2, b2 in nets], 0)


gaps = []
for r in H:
    p = predict(r["x"])
    true = np.concatenate([np.array(r["rumore_grana"])[2:24], np.array(r["rumore_colore"])[2:24] if r["rumore_colore"] else np.full(NB, np.nan)])
    gaps.append(true - p)
gaps = np.array(gaps)
calib = np.nanmedian(gaps, 0)
calib = np.where(np.isfinite(calib), calib, 0.0)
# Only ever lower the estimate (the real noise was below the network's everywhere measured).
calib = np.minimum(calib, 0.0)
print("taratura (log10, grana poi colore):", np.round(calib, 2))
fl = lambda a: "&[" + ", ".join(f"{float(v):.6}" for v in np.ravel(a)) + "]"
hidden = nets[0][0].shape[1]
src = f"""//! Weights of the noise estimator, written by `ps-train/scripts/noise_export.py` (do not edit).
//! 5 networks (hidden {hidden}) trained on {len(H) and 'simulated Jupiter stacks'}; calibrated on {len(H)} real stacks
//! (half-stack noise of 3 recordings).
#![allow(clippy::excessive_precision)]
use crate::curve_net::Net;
use crate::noise_net::NoiseModel;

pub(crate) static JUPITER: NoiseModel = NoiseModel {{
    inputs: {len(mu)},
    hidden: {hidden},
    outputs: {2 * NB},
    mean: {fl(mu)},
    scale: {fl(sd)},
    z_min: {fl(zmin)},
    z_max: {fl(zmax)},
    nets: &[
{chr(10).join(f"        Net {{ w1: {fl(W1)}, b1: {fl(b1)}, w2: {fl(W2)}, b2: {fl(b2)} }}," for W1, b1, W2, b2 in nets)}
    ],
    calibration: {fl(calib)},
}};
"""
open(out, "w").write(src)
print("scritto", out)
