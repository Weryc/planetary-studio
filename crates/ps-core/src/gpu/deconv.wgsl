// Richardson–Lucy with a separable PSF, one kernel per step. Mirrors the CPU code in
// deconv.rs and conv.rs; see there for the maths.

struct Params {
    w: u32,
    h: u32,
    // Taps of the 1-D kernel: 2·radius + 1.
    radius: u32,
    // Threads per row of the dispatch grid for element-wise kernels (groups_x · 256).
    stride: u32,
    tv_lambda: f32,
    damp: f32,
    // Number of partial sums written by `reduce` and `residual`.
    partials: u32,
    // Data above this level is the subject (for `residual`).
    signal: f32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(4) var<storage, read> TAPS: array<f32>;
// [0] = acceleration step α; from [1]: (Σ a·b, Σ b²) per reduction group.
@group(0) @binding(5) var<storage, read_write> S: array<f32>;

const EPS: f32 = 1e-7;

// … 2 1 0 1 2 … : mirror without repeating the edge sample (conv::reflect).
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

fn element(gid: vec3<u32>) -> u32 {
    return gid.y * P.stride + gid.x;
}

@compute @workgroup_size(16, 16)
fn conv_rows(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= P.w || gid.y >= P.h {
        return;
    }
    let w = i32(P.w);
    let r = i32(P.radius);
    let row = i32(gid.y) * w;
    var acc = 0.0;
    for (var k = 0; k <= 2 * r; k++) {
        acc += TAPS[k] * A[row + reflect(i32(gid.x) + k - r, w)];
    }
    OUT[gid.y * P.w + gid.x] = acc;
}

@compute @workgroup_size(16, 16)
fn conv_cols(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= P.w || gid.y >= P.h {
        return;
    }
    let r = i32(P.radius);
    var acc = 0.0;
    for (var k = 0; k <= 2 * r; k++) {
        acc += TAPS[k] * A[u32(reflect(i32(gid.y) + k - r, i32(P.h))) * P.w + gid.x];
    }
    OUT[gid.y * P.w + gid.x] = acc;
}

// Starting estimate: max(d, ε)
@compute @workgroup_size(256)
fn start(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = element(gid);
    if i >= P.w * P.h {
        return;
    }
    OUT[i] = max(A[i], EPS);
}

// y = max(x + α (x − x_prev), ε)
@compute @workgroup_size(256)
fn extrapolate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = element(gid);
    if i >= P.w * P.h {
        return;
    }
    OUT[i] = max(A[i] + S[0] * (A[i] - B[i]), EPS);
}

// White (1994) damped ratio; d·ln(m/d) − m + d is computed by series near m = d, where
// the direct formula cancels catastrophically in single precision.
fn damped_ratio(d: f32, m: f32, t: f32) -> f32 {
    var log_term = 0.0;
    if d > 0.0 {
        let x = (m - d) / d;
        if abs(x) < 0.1 {
            // d (ln(1 + x) − x) = d (−x²/2 + x³/3 − x⁴/4 + x⁵/5)
            log_term = d * x * x * (-0.5 + x * (1.0 / 3.0 + x * (-0.25 + x * 0.2)));
        } else {
            log_term = d * log(m / d) - m + d;
        }
    } else {
        log_term = -m + d;
    }
    let u = min(-2.0 / (t * t) * log_term, 1.0);
    let n = 10.0;
    let factor = pow(max(u, 0.0), n - 1.0) * (n - (n - 1.0) * u);
    return 1.0 + factor * (d / m - 1.0);
}

// ratio = d / max(model, ε)   (A = observed, B = model)
@compute @workgroup_size(256)
fn ratio(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = element(gid);
    if i >= P.w * P.h {
        return;
    }
    let m = max(B[i], EPS);
    if P.damp > 0.0 {
        OUT[i] = damped_ratio(A[i], m, P.damp);
    } else {
        OUT[i] = A[i] / m;
    }
}

// Normalised gradient of `A` at (x, y), forward differences (zero past the edge).
fn normal(x: i32, y: i32) -> vec2<f32> {
    let w = i32(P.w);
    let i = y * w + x;
    var g = vec2<f32>(0.0, 0.0);
    if x + 1 < w {
        g.x = A[i + 1] - A[i];
    }
    if y + 1 < i32(P.h) {
        g.y = A[i + w] - A[i];
    }
    return g / sqrt(dot(g, g) + 1e-8);
}

// new = max(y · correction / (1 − λ div(∇y/|∇y|)), ε)   (A = y, B = correction)
@compute @workgroup_size(256)
fn update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = element(gid);
    if i >= P.w * P.h {
        return;
    }
    var denom = 1.0;
    if P.tv_lambda > 0.0 {
        let x = i32(i % P.w);
        let y = i32(i / P.w);
        let n = normal(x, y);
        var div = n.x + n.y;
        if x > 0 {
            div -= normal(x - 1, y).x;
        }
        if y > 0 {
            div -= normal(x, y - 1).y;
        }
        denom = max(1.0 - P.tv_lambda * div, 0.2);
    }
    OUT[i] = max(A[i] * B[i] / denom, EPS);
}

// OUT += A (summing the terms of a separable PSF).
@compute @workgroup_size(256)
fn add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = element(gid);
    if i >= P.w * P.h {
        return;
    }
    OUT[i] += A[i];
}

// g = new − y   (A = new, B = y)
@compute @workgroup_size(256)
fn difference(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = element(gid);
    if i >= P.w * P.h {
        return;
    }
    OUT[i] = A[i] - B[i];
}

var<workgroup> sums: array<vec2<f32>, 256>;

// Partial sums (Σ a·b, Σ b²) of the last two corrections, one per workgroup.
@compute @workgroup_size(256)
fn reduce(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let n = P.w * P.h;
    var acc = vec2<f32>(0.0, 0.0);
    for (var i = wid.x * 256u + lid.x; i < n; i += P.partials * 256u) {
        acc += vec2<f32>(A[i] * B[i], B[i] * B[i]);
    }
    sums[lid.x] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if lid.x < s {
            sums[lid.x] += sums[lid.x + s];
        }
        workgroupBarrier();
    }
    if lid.x == 0u {
        S[1u + 2u * wid.x] = sums[0].x;
        S[2u + 2u * wid.x] = sums[0].y;
    }
}

// Partial sums (Σ (d − m)², count) over the subject (d > signal), one per workgroup.
// A = observed, B = model.
@compute @workgroup_size(256)
fn residual(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let n = P.w * P.h;
    var acc = vec2<f32>(0.0, 0.0);
    for (var i = wid.x * 256u + lid.x; i < n; i += P.partials * 256u) {
        if A[i] > P.signal {
            let r = A[i] - B[i];
            acc += vec2<f32>(r * r, 1.0);
        }
    }
    sums[lid.x] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if lid.x < s {
            sums[lid.x] += sums[lid.x + s];
        }
        workgroupBarrier();
    }
    if lid.x == 0u {
        S[1u + 2u * wid.x] = sums[0].x;
        S[2u + 2u * wid.x] = sums[0].y;
    }
}

// Anti-ringing: OUT += k (clamp(OUT, lo, hi) − OUT), with (lo, hi, k) interleaved in A.
@compute @workgroup_size(256)
fn hold(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = element(gid);
    if i >= P.w * P.h {
        return;
    }
    let k = A[3u * i + 2u];
    if k > 0.0 {
        let v = OUT[i];
        OUT[i] = v + k * (clamp(v, A[3u * i], A[3u * i + 1u]) - v);
    }
}

// α = clamp(Σ a·b / Σ b², 0, 0.95) (Biggs & Andrews).
@compute @workgroup_size(1)
fn step_length() {
    var num = 0.0;
    var den = 0.0;
    for (var k = 0u; k < P.partials; k++) {
        num += S[1u + 2u * k];
        den += S[2u + 2u * k];
    }
    if den > 0.0 {
        S[0] = clamp(num / den, 0.0, 0.95);
    } else {
        S[0] = 0.0;
    }
}
