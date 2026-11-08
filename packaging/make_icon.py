"""Draws packaging/icon_1024.png (a ringed planet on a dark rounded square). Needs Pillow."""

import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

S = 1024
SS = 4  # supersampling for smooth edges
N = S * SS
ACCENT = (240, 168, 72)


def lerp(a, b, t):
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


img = Image.new("RGBA", (N, N), (0, 0, 0, 0))
d = ImageDraw.Draw(img)

# macOS-style rounded square with a subtle vertical gradient.
margin, radius = int(N * 0.09), int(N * 0.2)
bg = Image.new("RGBA", (N, N))
bd = ImageDraw.Draw(bg)
for y in range(N):
    bd.line([(0, y), (N, y)], fill=lerp((30, 33, 42), (10, 11, 15), y / N) + (255,))
mask = Image.new("L", (N, N), 0)
ImageDraw.Draw(mask).rounded_rectangle([margin, margin, N - margin, N - margin], radius, fill=255)
img.paste(bg, (0, 0), mask)

# A few stars.
for sx, sy, sr in [(0.25, 0.24, 5), (0.74, 0.2, 4), (0.8, 0.72, 6), (0.22, 0.78, 3), (0.62, 0.3, 3)]:
    c = (int(sx * N), int(sy * N))
    d.ellipse([c[0] - sr * SS, c[1] - sr * SS, c[0] + sr * SS, c[1] + sr * SS], fill=(230, 232, 240, 220))

cx, cy, pr = N / 2, N / 2, N * 0.2
tilt = math.radians(-20)
rx, ry = pr * 2.05, pr * 0.55


def ring(layer_back):
    """Ring drawn as an ellipse outline; the back half goes behind the planet."""
    ring_img = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    rd = ImageDraw.Draw(ring_img)
    for k, (scale, width, color) in enumerate([(1.0, 0.07, ACCENT), (0.83, 0.035, (200, 140, 60))]):
        box = [cx - rx * scale, cy - ry * scale, cx + rx * scale, cy + ry * scale]
        rd.ellipse(box, outline=color + (255,), width=int(pr * width))
    ring_img = ring_img.rotate(math.degrees(-tilt), center=(cx, cy), resample=Image.BICUBIC)
    clip = Image.new("L", (N, N), 0)
    cd = ImageDraw.Draw(clip)
    # Back half = upper half in the ring's own frame (before rotation), approximated by a rotated half-plane.
    half = Image.new("L", (N, N), 0)
    ImageDraw.Draw(half).rectangle([0, 0, N, cy] if layer_back else [0, cy, N, N], fill=255)
    half = half.rotate(math.degrees(-tilt), center=(cx, cy))
    cd.bitmap((0, 0), half, fill=255)
    out = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    out.paste(ring_img, (0, 0), Image.composite(ring_img.split()[3], Image.new("L", (N, N), 0), clip))
    return out


img = Image.alpha_composite(img, ring(True))

# Planet: banded sphere with limb darkening.
planet = Image.new("RGBA", (N, N), (0, 0, 0, 0))
px = planet.load()
x0, x1, y0, y1 = int(cx - pr), int(cx + pr), int(cy - pr), int(cy + pr)
for y in range(y0, y1):
    for x in range(x0, x1):
        dx, dy = (x - cx) / pr, (y - cy) / pr
        r2 = dx * dx + dy * dy
        if r2 >= 1:
            continue
        # Bands follow the ring tilt.
        v = -dx * math.sin(tilt) + dy * math.cos(tilt)
        band = 0.82 + 0.18 * math.sin(v * 9.0)
        limb = (1 - r2) ** 0.3
        light = max(0.0, 1 - 0.35 * (dx + 0.6) ** 2 - 0.2 * (dy + 0.5) ** 2)
        k = band * limb * (0.55 + 0.45 * light)
        px[x, y] = tuple(min(255, int(c * k * 1.08)) for c in (246, 196, 120)) + (255,)
img = Image.alpha_composite(img, planet)
img = Image.alpha_composite(img, ring(False))

glow = img.filter(ImageFilter.GaussianBlur(SS * 2))
img = Image.alpha_composite(glow, img)
out = Path(__file__).with_name("icon_1024.png")
img.resize((S, S), Image.LANCZOS).save(out)
print(f"wrote {out}")
