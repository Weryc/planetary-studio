#!/usr/bin/env python3
"""Plan of the simulated-raw dataset for one subject (Jupiter, or Saturn with `--subject saturno`), as a TSV that
`dataset_run.sh` executes row by row with `sim_stack`.

    python3 crates/ps-train/scripts/dataset_plan.py > Immagini_test/dataset_giove/piano.tsv
    python3 crates/ps-train/scripts/dataset_plan.py --keep OLD_PLAN N > NEW_PLAN

With --keep, the first N rows of OLD_PLAN (already simulated) are copied as they are and only
the rest is planned again: training groups get what they still miss, validation the grid cells
not yet done, ids and seeds continue after N.

Revision 2 (2026-11-15): r0 drawn over 4–22 cm (was 4–15) and D/r0 no longer clamped at 1.5
for training (floor 0.6): the first 44 raws showed big telescopes almost never below D/r0 2.2
and small ones stuck at 1.5, and a curve learned on them over-sharpened good-seeing raws.
The validation grid keeps its old rule, so the bench stays the same for all its rows.

Training groups get random captures: each original goes through all seven telescopes in
turn; the night is drawn as Fried's r0 (4–15 cm at 550 nm, log-uniform), so a small telescope
sees a smaller D/r0 than a big one on the same night, as in reality. Validation groups are
never trained on and get a fixed grid: three nights (r0 12, 7, 4 cm) × seven telescopes, the
bench every version of the network is compared on.

Rows are ordered round-robin (groups, telescopes, nights, train/validation in proportion),
so any first block of the plan is already balanced: running it in steps is fine.
The same seed always gives the same plan.
"""
import math
import random

SEED = 20261115
# Frames generated and share kept (training rows); `--frames LO HI --share LO HI --seed N` change
# them (`--share`, since `--keep` keeps rows of an old plan). Revision 3 (2026-11-16): 1000–6000 frames, 25–75 % kept, like the real recordings the
# noise is calibrated on (the first plans had 300–3000 and 10–30 %: far noisier stacks).
import sys as _sys
def _opt(k, default):
    a = _sys.argv
    return (float(a[a.index(k) + 1]), float(a[a.index(k) + 2])) if k in a else default
FRAMES = _opt("--frames", (300, 3000))
KEEP = _opt("--share", (0.10, 0.30))
# The validation grid takes the middle of those ranges (1500 frames, 20 % before).
VAL_FRAMES = int(round(math.sqrt(FRAMES[0] * FRAMES[1]), -2)) if "--frames" in _sys.argv else 1500
VAL_KEEP = round(0.5 * (KEEP[0] + KEEP[1]), 2) if "--share" in _sys.argv else 0.2
if "--seed" in _sys.argv:
    SEED = int(_sys.argv[_sys.argv.index("--seed") + 1])
SCOPES = {  # name: aperture (mm)
    "apo102": 102, "mak127": 127, "newton200": 200, "c8": 203, "sc235": 235, "sc356": 356, "edgehd11": 279.4,
}
SCOPE_ORDER = ["apo102", "mak127", "newton200", "c8", "sc235", "sc356", "edgehd11"]
CAMERAS = ["imx462c", "imx462m", "asi678mc"]

# Jupiter: (group, [files]); multi-globe groups share their simulations between the globes.
TRAIN = [
    ("heic1410a", ["heic1410a.jpg"]), ("heic1708a", ["heic1708a.jpg"]), ("heic2017a", ["heic2017a.jpg"]),
    ("heic2113b", ["heic2113b.jpg"]), ("heic2303d", ["heic2303d.jpg"]), ("heic2404b", ["heic2404b.jpg"]),
    ("heic2404c", ["heic2404c.jpg"]),
    ("heic1010b", [f"heic1010b_{i}.png" for i in (1, 2)]),
    ("heic2412b", [f"heic2412b_{i}.png" for i in range(1, 9)]),
]
PER_GROUP = {"heic1010b": 12, "heic2412b": 16}  # others: 12
VALIDATION = [("heic2303e", "heic2303e.jpg"), ("heic1914a", "heic1914a.jpg"), ("opo1016b", "opo1016b.jpg")]
NIGHTS = [("calmo", 12.0), ("medio", 7.0), ("turbolento", 4.0)]  # r0 at 550 nm, cm

# Saturn (`--subject saturno`, 2026-11-17): `diam` is the ring span (35–46″ over the years,
# 42″ for the bench); the planet is dimmer than Jupiter, so captures lean to the higher gains.
# The five opo0115 tilts are one series (1996–2000): one group, so none is ever judged by its
# siblings. Validation: heic2312a (2022, spokes), heic0904a (2009, rings edge-on), opo2305a (2022).
SUBJECT = _sys.argv[_sys.argv.index("--subject") + 1] if "--subject" in _sys.argv else "giove"
GAINS = ["basso", "medio", "alto"]
DIAM, VAL_DIAM = (33, 49), 44.0
if SUBJECT == "saturno":
    TRAIN = [
        ("heic1917a", ["heic1917a.jpg"]), ("opo2043a", ["opo2043a.jpg"]), ("heic1814b", ["heic1814b.jpg"]),
        ("PIA06193", ["PIA06193.jpg"]), ("heic2113c", ["heic2113c.jpg"]), ("heic2612a", ["heic2612a_left.jpg"]),
        ("opo0115", [f"opo0115{c}.jpg" for c in "cdefg"]), ("opo0323c", ["opo0323c.jpg"]),
        ("opo0418a", ["opo0418a.jpg"]), ("opo9828c", ["opo9828c.jpg"]),
    ]
    PER_GROUP = {"opo0115": 15}
    VALIDATION = [("heic2312a", "heic2312a.jpg"), ("heic0904a", "heic0904a.jpg"), ("opo2305a", "opo2305a.jpg")]
    GAINS = ["basso", "medio", "medio", "alto", "alto"]
    DIAM, VAL_DIAM = (35, 46), 42.0


R0_CM = (4.0, 22.0)  # training nights, r0 at 550 nm (log-uniform)
D_R0_FLOOR = 0.6     # below this the capture is diffraction-limited anyway


def d_r0(scope, r0_cm, floor=1.5):
    return round(max(floor, min(15.0, SCOPES[scope] / (10.0 * r0_cm))), 2)


def train_rows(rng, done=None):
    done = done or {}
    per_group = []
    for g, files in TRAIN:
        n = PER_GROUP.get(g, 12) - done.get(g, 0)
        start = rng.randrange(7)
        rows = []
        for k in range(n):
            scope = SCOPE_ORDER[(start + k) % 7]
            camera = "asi678mc" if scope == "edgehd11" else rng.choice(["imx462c", "imx462c", "imx462m", "asi678mc"])
            r0 = math.exp(rng.uniform(math.log(R0_CM[0]), math.log(R0_CM[1])))
            gain = rng.choice(GAINS)
            bits = 12 if gain == "basso" else rng.choice([8, 8, 12])
            frames = int(round(math.exp(rng.uniform(math.log(FRAMES[0]), math.log(FRAMES[1]))), -1))
            rows.append({
                "split": "train", "group": g, "ref": files[k % len(files)], "scope": scope, "camera": camera,
                "d_r0": d_r0(scope, r0, D_R0_FLOOR), "r0_cm": round(r0, 1), "night": "-", "gain": gain, "bits": bits,
                "frames": frames, "keep": round(rng.uniform(*KEEP), 2), "diam": round(rng.uniform(*DIAM), 1),
                "ground": round(rng.uniform(0.45, 0.75), 2),
            })
        per_group.append(rows)
    # Round-robin over groups.
    out = []
    for r in range(max(len(x) for x in per_group)):
        for rows in per_group:
            if r < len(rows):
                out.append(rows[r])
    return out


def validation_rows(done_cells=frozenset()):
    per_group = []
    for g, f in VALIDATION:
        rows = []
        for i in range(21):
            scope = SCOPE_ORDER[i % 7]
            night, r0 = NIGHTS[(i + i // 7) % 3]
            if (g, scope, night) in done_cells:
                continue
            rows.append({
                "split": "val", "group": g, "ref": f, "scope": scope, "camera": "asi678mc" if scope == "edgehd11" else "imx462c",
                "d_r0": d_r0(scope, r0), "r0_cm": r0, "night": night, "gain": "medio", "bits": 8, "frames": VAL_FRAMES,
                "keep": VAL_KEEP, "diam": VAL_DIAM, "ground": 0.6,
            })
        per_group.append(rows)
    out = []
    for r in range(21):
        for rows in per_group:
            if r < len(rows):
                out.append(rows[r])
    return out


def main():
    import sys
    kept, header = [], None
    if "--keep" in sys.argv:
        k = sys.argv.index("--keep")
        lines = open(sys.argv[k + 1]).read().splitlines()
        header, kept = lines[0], lines[1:1 + int(sys.argv[k + 2])]
    cols = ["id", "split", "group", "ref", "scope", "camera", "night", "r0_cm", "d_r0", "gain", "bits", "frames", "keep", "diam", "ground", "seed"]
    done, cells = {}, set()
    for line in kept:
        f = dict(zip(cols, line.split("\t")))
        if f["split"] == "train":
            done[f["group"]] = done.get(f["group"], 0) + 1
        else:
            cells.add((f["group"], f["scope"], f["night"]))
    rng = random.Random(SEED + (2 if kept else 0))
    train, val = train_rows(rng, done), validation_rows(cells)
    rows, i, j = [], 0, 0
    # Merge in proportion: whichever list is behind goes next.
    while i < len(train) or j < len(val):
        if j >= len(val) or (i < len(train) and i / len(train) <= j / len(val)):
            rows.append(train[i]); i += 1
        else:
            rows.append(val[j]); j += 1
    rows = rows
    print("\t".join(cols))
    for line in kept:
        print(line)
    for n, r in enumerate(rows, start=len(kept)):
        r["id"] = f"g{n + 1:03d}"
        r["seed"] = 100000 + n + 1
        print("\t".join(str(r[c]) for c in cols))


main()
