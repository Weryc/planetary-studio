// Building blocks shared by several modules: separable filters with holes (starlet), box
// blurs (Gaussian approximation), bilateral filter and local minimum/maximum. Each mirrors a
// CPU function; see conv.rs, mask.rs, enhance.rs and dering.rs.

struct Params {
    w: u32,
    h: u32,
    // Kernel radius (taps = 2r + 1), box radius or window radius.
    r: u32,
    // Spacing between taps (à trous holes).
    step: u32,
    // Threads per row of the dispatch grid for element-wise kernels.
    stride: u32,
    // 0 = minimum, 1 = maximum.
    mode: u32,
    a: f32,
    b: f32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(4) var<storage, read> TAPS: array<f32>;

fn reflect(i: i32, n: i32) -> i32 {
    if n == 1 {
        return 0;
    }
    let period = 2 * n - 2;
    var m = i % period;
    if m < 0 {
        m += period;
    }
    if m >= n {
        return period - m;
    }
    return m;
}

fn inside(gid: vec3<u32>) -> bool {
    return gid.x < P.w && gid.y < P.h;
}

// ---- separable filter with holes, mirror edges (conv::convolve_separable) ----

@compute @workgroup_size(16, 16)
fn conv_rows(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !inside(gid) {
        return;
    }
    let w = i32(P.w);
    let r = i32(P.r);
    let row = i32(gid.y) * w;
    var acc = 0.0;
    for (var k = 0; k <= 2 * r; k++) {
        acc += TAPS[k] * A[row + reflect(i32(gid.x) + (k - r) * i32(P.step), w)];
    }
    OUT[gid.y * P.w + gid.x] = acc;
}

@compute @workgroup_size(16, 16)
fn conv_cols(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !inside(gid) {
        return;
    }
    let r = i32(P.r);
    var acc = 0.0;
    for (var k = 0; k <= 2 * r; k++) {
        acc += TAPS[k] * A[u32(reflect(i32(gid.y) + (k - r) * i32(P.step), i32(P.h))) * P.w + gid.x];
    }
    OUT[gid.y * P.w + gid.x] = acc;
}

// OUT = A − B
@compute @workgroup_size(256)
fn subtract(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * P.stride + gid.x;
    if i >= P.w * P.h {
        return;
    }
    OUT[i] = A[i] - B[i];
}

// ---- box average, clamped edges (mask::box_rows / box_cols) ----

@compute @workgroup_size(16, 16)
fn box_rows(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !inside(gid) {
        return;
    }
    let w = i32(P.w);
    let r = i32(P.r);
    let row = i32(gid.y) * w;
    var acc = 0.0;
    for (var k = -r; k <= r; k++) {
        acc += A[row + clamp(i32(gid.x) + k, 0, w - 1)];
    }
    OUT[gid.y * P.w + gid.x] = acc / f32(2 * r + 1);
}

@compute @workgroup_size(16, 16)
fn box_cols(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !inside(gid) {
        return;
    }
    let h = i32(P.h);
    let r = i32(P.r);
    var acc = 0.0;
    for (var k = -r; k <= r; k++) {
        acc += A[u32(clamp(i32(gid.y) + k, 0, h - 1)) * P.w + gid.x];
    }
    OUT[gid.y * P.w + gid.x] = acc / f32(2 * r + 1);
}

// ---- bilateral filter (enhance::bilateral): a = σ spatial, b = −1 / (2 σ range²) ----

@compute @workgroup_size(16, 16)
fn bilateral(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !inside(gid) {
        return;
    }
    let w = i32(P.w);
    let h = i32(P.h);
    let r = i32(P.r);
    let x = i32(gid.x);
    let y = i32(gid.y);
    let c = A[y * w + x];
    let inv_s = -1.0 / (2.0 * P.a * P.a);
    var sum = 0.0;
    var norm = 0.0;
    for (var dy = -r; dy <= r; dy++) {
        let sy = clamp(y + dy, 0, h - 1);
        for (var dx = -r; dx <= r; dx++) {
            let v = A[sy * w + clamp(x + dx, 0, w - 1)];
            let weight = exp(f32(dx * dx + dy * dy) * inv_s) * exp((v - c) * (v - c) * P.b);
            sum += weight * v;
            norm += weight;
        }
    }
    OUT[gid.y * P.w + gid.x] = sum / norm;
}

// ---- local minimum / maximum over a (2r+1)² window inside the image (dering::min_max) ----

fn pick(a: f32, b: f32) -> f32 {
    if P.mode == 0u {
        return min(a, b);
    }
    return max(a, b);
}

@compute @workgroup_size(16, 16)
fn extreme_rows(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !inside(gid) {
        return;
    }
    let w = i32(P.w);
    let r = i32(P.r);
    let row = i32(gid.y) * w;
    let x = i32(gid.x);
    var acc = A[row + x];
    for (var k = max(x - r, 0); k <= min(x + r, w - 1); k++) {
        acc = pick(acc, A[row + k]);
    }
    OUT[gid.y * P.w + gid.x] = acc;
}

@compute @workgroup_size(16, 16)
fn extreme_cols(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !inside(gid) {
        return;
    }
    let h = i32(P.h);
    let r = i32(P.r);
    let y = i32(gid.y);
    var acc = A[gid.y * P.w + gid.x];
    for (var k = max(y - r, 0); k <= min(y + r, h - 1); k++) {
        acc = pick(acc, A[u32(k) * P.w + gid.x]);
    }
    OUT[gid.y * P.w + gid.x] = acc;
}
