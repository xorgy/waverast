//! GPU wavelet coefficient computation.
//!
//! Implements the boundary-integral method for computing Haar wavelet coefficients
//! of a 2D indicator function, as described in:
//!   Manson & Schaefer, "Wavelet Rasterization" (2006), equations 5–8.
//!
//! One GPU thread per boundary segment. Each thread iterates over all quadtree
//! levels, computes the cell range from its segment's bounding box, and for each
//! overlapping cell computes the three detail coefficients c^(0,1), c^(1,0), c^(1,1).
//!
//! Lines, cubics, arcs: all three coefficients via the tent-function integral
//! (equation 7 with Ψ̄ from §2.1). Cubics include bisection verification for
//! crossings missed by the f32 analytic solver.
//! Quads, superellipses: c11 via equation-8 K/L decomposition, c01/c10 via tent.
//!
//! The cubic solver follows Blinn, "How to Solve a Cubic Equation" (Parts 1–5,
//! IEEE CG&A 2006–2007): trigonometric method for three real roots,
//! Cardano with safe addition for one real root.

const EPS: f32 = 1.4e-5;
const PI: f32 = 3.14159265358979323846;
const TAU: f32 = 6.28318530717958647692;

struct Params {
    max_j: u32,
    num_segments: u32,
}

/// K/L boundary integral terms (M&S equation 8, Appendix A).
///
/// K is the constant term (depends only on clipped endpoints),
/// L is the linear term (depends on the curve's interior path).
struct KL {
    kx: f32,
    ky: f32,
    lx: f32,
    ly: f32,
}

/// Zero-initialized KL.
fn kl_zero() -> KL {
    return KL(0.0, 0.0, 0.0, 0.0);
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> segments: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> coeffs: array<atomic<u32>>;

// ---- Atomic f32 add via compare-and-swap ----

/// Atomic f32 addition via CAS loop.
fn atomic_add_f32(idx: u32, val: f32) {
    if abs(val) < 1e-20 {
        return;
    }
    var old = atomicLoad(&coeffs[idx]);
    loop {
        let new_val = bitcast<f32>(old) + val;
        let result = atomicCompareExchangeWeak(&coeffs[idx], old, bitcast<u32>(new_val));
        if result.exchanged {
            break;
        }
        old = result.old_value;
    }
}

// ---- Shared utilities ----

/// Base index into coefficient array for level j: (4^j − 1) / 3.
fn level_offset(j: u32) -> u32 {
    return ((1u << (2u * j)) - 1u) / 3u;
}

/// Accumulate c11 from 4-quadrant KL values (M&S equation 8).
///
/// c^(1,1) = L00_x - L01_x + K10_x - L10_x - K11_x + L11_x.
fn accumulate_c11(cell_idx: u32, q00: KL, q01: KL, q10: KL, q11: KL) {
    let c11 = q00.lx - q01.lx + q10.kx - q10.lx - q11.kx + q11.lx;
    atomic_add_f32(cell_idx * 3u + 2u, c11);
}


// ---- Tent function helpers ----
//
// The tent function Ψ̄(t) is the antiderivative of the Haar wavelet ψ (M&S §2.1).
// Using Ψ̄ directly in the boundary integral (equation 7) gives c01 and c10
// without the four-quadrant decomposition of equation 8. This eliminates
// cancellation errors when segments lie on cell midpoint boundaries, where
// equation 8 produces equal-and-opposite contributions that must cancel exactly.

/// Tent function Ψ̄(t), the antiderivative of the Haar wavelet (M&S §2.1).
///
/// Ψ̄(t) = t for t ∈ [0, 0.5], 1-t for t ∈ [0.5, 1], 0 otherwise.
fn tent(t: f32) -> f32 {
    if t <= 0.0 || t >= 1.0 { return 0.0; }
    if t <= 0.5 { return t; }
    return 1.0 - t;
}

/// ∫Ψ̄(v) du for a line piece where v stays on one side of 0.5 (exact).
///
/// Since v is linear in u for a line, and Ψ̄ is linear on each half,
/// the trapezoidal rule is exact.
fn tent_line_piece(u0: f32, v0: f32, u1: f32, v1: f32) -> f32 {
    return (u1 - u0) * (tent(v0) + tent(v1)) * 0.5;
}

/// ∫Ψ̄(v) du for a line, split at v=0.5 where Ψ̄ has a slope discontinuity.
fn tent_line_integral(u0: f32, v0: f32, u1: f32, v1: f32) -> f32 {
    if (v0 < 0.5 && v1 > 0.5) || (v0 > 0.5 && v1 < 0.5) {
        let t = (0.5 - v0) / (v1 - v0);
        let u_mid = u0 + t * (u1 - u0);
        return tent_line_piece(u0, v0, u_mid, 0.5) + tent_line_piece(u_mid, 0.5, u1, v1);
    }
    return tent_line_piece(u0, v0, u1, v1);
}

// ---- Line K/L (M&S Appendix A, line segment case) ----

/// Liang-Barsky line clipping to the unit square [0,1]².
fn liang_barsky(x0: f32, y0: f32, x1: f32, y1: f32) -> vec4<f32> {
    var t0: f32 = 0.0;
    var t1: f32 = 1.0;
    let dx = x1 - x0;
    let dy = y1 - y0;

    let p = array<f32, 4>(-dx, dx, dy, -dy);
    let q = array<f32, 4>(x0, 1.0 - x0, 1.0 - y0, y0);

    for (var i: u32 = 0u; i < 4u; i++) {
        if p[i] == 0.0 && q[i] < 0.0 {
            return vec4(-1.0);
        }
        if p[i] < 0.0 {
            let r = q[i] / p[i];
            if r > t1 { return vec4(-1.0); }
            t0 = max(t0, r);
        } else if p[i] > 0.0 {
            let r = q[i] / p[i];
            if r < t0 { return vec4(-1.0); }
            t1 = min(t1, r);
        }
    }
    return vec4(x0 + t0 * dx, y0 + t0 * dy, x0 + t1 * dx, y0 + t1 * dy);
}

/// Line segment K/L clipped to [0,1]² (M&S Appendix A).
///
/// K = ¼(Δy), L = ⅛(Δy)(x₀+x₁). `skip_x`/`skip_y`: boundary coincidence
/// skip for equation-8 c11.
fn line_get_kl(x0: f32, y0: f32, x1: f32, y1: f32, skip_x: bool, skip_y: bool) -> KL {
    var kl = kl_zero();
    let clip = liang_barsky(x0, y0, x1, y1);
    if clip.x < -0.5 { return kl; }

    let cx0 = clip.x; let cy0 = clip.y;
    let cx1 = clip.z; let cy1 = clip.w;

    if skip_x && ((abs(cx0 - 1.0) < EPS && abs(cx1 - 1.0) < EPS) || (abs(cx0) < EPS && abs(cx1) < EPS)) { return kl; }
    if skip_y && ((abs(cy0 - 1.0) < EPS && abs(cy1 - 1.0) < EPS) || (abs(cy0) < EPS && abs(cy1) < EPS)) { return kl; }

    kl.kx = 0.25 * (cy1 - cy0);
    kl.ky = 0.25 * (cx0 - cx1);
    kl.lx = 0.125 * (cy1 - cy0) * (cx1 + cx0);
    kl.ly = 0.125 * (cx0 - cx1) * (cy1 + cy0);
    return kl;
}

// ---- Line tent (equation 7 with Ψ̄; c01, c10, c11) ----

/// Line tent integral in cell coordinates [0,1]².
///
/// Returns vec3(c01, c10, c11). c11 = Σ sign_v · tent_u.
fn line_tent(u0: f32, v0: f32, u1: f32, v1: f32) -> vec3<f32> {
    let clip = liang_barsky(u0, v0, u1, v1);
    if clip.x < -0.5 { return vec3(0.0); }
    let cu0 = clip.x; let cv0 = clip.y;
    let cu1 = clip.z; let cv1 = clip.w;

    let du = cu1 - cu0;
    let dv = cv1 - cv0;

    // Split at u=0.5 and v=0.5 for c11
    var ts: array<f32, 4>;
    var n: u32 = 0u;
    ts[n] = 0.0; n++;
    if abs(du) > EPS {
        let t = (0.5 - cu0) / du;
        if t > EPS && t < 1.0 - EPS { ts[n] = t; n++; }
    }
    if abs(dv) > EPS {
        let t = (0.5 - cv0) / dv;
        if t > EPS && t < 1.0 - EPS { ts[n] = t; n++; }
    }
    ts[n] = 1.0; n++;

    for (var i: u32 = 0u; i < n; i++) {
        for (var j: u32 = i + 1u; j < n; j++) {
            if ts[j] < ts[i] { let tmp = ts[i]; ts[i] = ts[j]; ts[j] = tmp; }
        }
    }

    var c01: f32 = 0.0;
    var c10: f32 = 0.0;
    var c11: f32 = 0.0;

    for (var i: u32 = 0u; i + 1u < n; i++) {
        let ta = ts[i]; let tb = ts[i + 1u];
        if tb - ta < EPS { continue; }
        let ua = cu0 + ta * du; let va = cv0 + ta * dv;
        let ub = cu0 + tb * du; let vb = cv0 + tb * dv;

        let tent_v = tent_line_piece(ua, va, ub, vb);
        let tent_u = tent_line_piece(va, ua, vb, ub);

        c01 -= tent_v;
        c10 += tent_u;

        let mid_v = (va + vb) * 0.5;
        let sign_v = select(-1.0, 1.0, mid_v < 0.5);
        c11 += sign_v * tent_u;
    }

    return vec3(c01, c10, c11);
}


// ---- Cubic Bezier K/L (M&S §2.2, Appendix A) ----

/// Cubic Bernstein evaluation B(t) = Σ bᵢ·Bᵢ³(t).
fn cubic_eval(x0: f32, y0: f32, x1: f32, y1: f32,
              x2: f32, y2: f32, x3: f32, y3: f32, t: f32) -> vec2<f32> {
    let u = 1.0 - t;
    let u2 = u * u;
    let t2 = t * t;
    return vec2(
        x0 * u2 * u + 3.0 * x1 * u2 * t + 3.0 * x2 * u * t2 + x3 * t * t2,
        y0 * u2 * u + 3.0 * y1 * u2 * t + 3.0 * y2 * u * t2 + y3 * t * t2,
    );
}

/// Test if point is in [0,1]² with EPS tolerance.
fn is_in_unit(p: vec2<f32>) -> bool {
    return p.x >= -EPS && p.x <= 1.0 + EPS && p.y >= -EPS && p.y <= 1.0 + EPS;
}

/// Test if point is in [0,1]² with wider tolerance (for tent crossing filters).
fn is_in_unit_wide(p: vec2<f32>) -> bool {
    return p.x >= -0.01 && p.x <= 1.01 && p.y >= -0.01 && p.y <= 1.01;
}

/// Blinn's σ function (P5 §3): sign(x), but never 0.
///
/// Used in safe addition to pick the sum with larger magnitude.
fn sigma(x: f32) -> f32 {
    if x < 0.0 { return -1.0; }
    return 1.0;
}

/// Cube root that handles negative values.
fn cbrt(x: f32) -> f32 {
    return sigma(x) * pow(abs(x), 1.0 / 3.0);
}

/// Quadratic solver for ax² + bx + c = 0. Count in .x, roots in .yz.
///
/// Uses safe addition (Blinn P5 §3): first root -b - σ(b)√Δ, second via
/// Vieta's formula c/(a·r₁).
fn solve_quadratic_f(a: f32, b: f32, c: f32) -> vec4<f32> {
    if abs(a) < 1e-6 {
        if abs(b) < 1e-6 { return vec4(0.0); }
        return vec4(1.0, -c / b, 0.0, 0.0);
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 { return vec4(0.0); }
    let sd = sqrt(disc);
    // Safe addition (Blinn P5): pick the sum with larger magnitude
    let r1 = (-b - sigma(b) * sd) / (2.0 * a);
    // Second root via Vieta's formula c/(a·r1), guarded against r1 ≈ 0
    var r2: f32;
    if abs(r1) < 1e-20 {
        r2 = (-b + sigma(b) * sd) / (2.0 * a);
    } else {
        r2 = c / (a * r1);
    }
    return vec4(2.0, r1, r2, 0.0);
}

/// Cubic solver for ax³ + bx² + cx + d = 0. Count in .x, roots in .yzw.
///
/// Based on Blinn, "How to Solve a Cubic Equation" (Parts 1–5, IEEE CG&A 2006–2007):
///   - Depressed cubic via x = t - b/(3a) (Blinn P1 §2)
///   - Discriminant from Hessian (s₁, s₂, s₃) without division (Blinn P5 §4)
///   - Three real roots: atan2 trig method with m from s₁ (Blinn P4/P5)
///   - One real root: Cardano with safe addition (Blinn P5 Eq 26)
///   - Double root (disc = 0): handled by trig path (neg_delta_81 ≤ 0)
///
/// Each root is Newton-refined via the Bernstein form for f32 precision.
fn solve_cubic_f(a: f32, b: f32, c: f32, d: f32) -> vec4<f32> {
    if abs(a) < EPS {
        return solve_quadratic_f(b, c, d);
    }

    // Depress the cubic: substitute x = t - b/(3a) to get t³ + pt + q = 0
    let inv_a = 1.0 / a;
    let shift = b * inv_a / 3.0;

    // p = (3ac - b²) / (3a²), q = (2b³ - 9abc + 27a²d) / (27a³)
    let p = (3.0 * a * c - b * b) / (3.0 * a * a);
    let q = (2.0 * b * b * b - 9.0 * a * b * c + 27.0 * a * a * d) / (27.0 * a * a * a);

    let p_over_3 = p / 3.0;

    // Discriminant via Hessian invariants (Blinn P5 §4, Eq 17–19).
    // s₁ = 3ac - b², s₂ = 9ad - bc, s₃ = 3bd - c²
    // -81Δ = s₂² - 4·s₁·s₃ (no division by a, preserves f32 precision)
    let s1 = 3.0 * a * c - b * b;
    let s2 = 9.0 * a * d - b * c;
    let s3 = 3.0 * b * d - c * c;
    let neg_delta_81 = s2 * s2 - 4.0 * s1 * s3;

    var all_roots = vec3(0.0);
    var n_real: u32;

    if neg_delta_81 <= 0.0 {
        // Three real roots (or double root when disc = 0) — trigonometric method (Blinn P4).
        // Uses ≤ 0 (not < 0) so that zero-discriminant double roots are handled
        // by the trig path, which naturally produces all roots including the single root.
        // m = √(-p/3) = √(-s₁) / (3|a|) — computed from s₁ directly (Blinn P5 §5)
        // to avoid precision loss from dividing by a² in the depressed form.
        let m = sqrt(max(-s1, 0.0)) / (3.0 * abs(a));
        let disc_term = q * q + 4.0 * p_over_3 * p_over_3 * p_over_3;
        let theta = atan2(sqrt(max(-disc_term, 0.0)), -q) / 3.0;
        let cos_t = cos(theta);
        let sin_t = sin(theta);
        let two_m = 2.0 * m;
        all_roots[0] = two_m * cos_t - shift;
        all_roots[1] = two_m * (-0.5 * cos_t + 0.8660254 * sin_t) - shift;
        all_roots[2] = two_m * (-0.5 * cos_t - 0.8660254 * sin_t) - shift;
        n_real = 3u;
    } else {
        // One real root — Cardano with safe addition (Blinn P5 §3, Eq 26).
        let disc_term = q * q + 4.0 * p_over_3 * p_over_3 * p_over_3;
        let sqrt_disc = sqrt(max(disc_term, 0.0));
        let T0 = -sigma(q) * sqrt_disc;
        let T1 = -q + T0;
        let p_val = cbrt(T1 * 0.5);
        var q_val: f32;
        if T1 == T0 {
            q_val = -p_val;
        } else {
            q_val = -p_over_3 / p_val;
        }
        all_roots[0] = p_val + q_val - shift;
        n_real = 1u;
    }

    // Filter to [0, 1] with wide tolerance — caller refines via Bernstein Newton.
    // f32 closed-form roots can be far off; use generous window so Newton can fix them.
    var filtered = vec3(0.0);
    var count: u32 = 0u;
    for (var i: u32 = 0u; i < n_real; i++) {
        let r = all_roots[i];
        if r >= -0.1 && r <= 1.1 {
            filtered[count] = clamp(r, 0.0, 1.0);
            count++;
        }
    }

    return vec4(f32(count), filtered.x, filtered.y, filtered.z);
}

/// Cubic Bezier sub-curve extraction for t ∈ [t0, t1] (de Casteljau).
fn cubic_subsection(x0: f32, y0: f32, x1: f32, y1: f32,
                    x2: f32, y2: f32, x3: f32, y3: f32,
                    t0: f32, t1: f32) -> array<f32, 8> {
    let u0 = 1.0 - t0;
    let u1 = 1.0 - t1;
    let qxa = x0*u0*u0 + x1*2.0*t0*u0 + x2*t0*t0;
    let qxb = x0*u1*u1 + x1*2.0*t1*u1 + x2*t1*t1;
    let qxc = x1*u0*u0 + x2*2.0*t0*u0 + x3*t0*t0;
    let qxd = x1*u1*u1 + x2*2.0*t1*u1 + x3*t1*t1;
    let qya = y0*u0*u0 + y1*2.0*t0*u0 + y2*t0*t0;
    let qyb = y0*u1*u1 + y1*2.0*t1*u1 + y2*t1*t1;
    let qyc = y1*u0*u0 + y2*2.0*t0*u0 + y3*t0*t0;
    let qyd = y1*u1*u1 + y2*2.0*t1*u1 + y3*t1*t1;
    return array<f32, 8>(
        qxa*u0 + qxc*t0, qya*u0 + qyc*t0,
        qxa*u1 + qxc*t1, qya*u1 + qyc*t1,
        qxb*u0 + qxd*t0, qyb*u0 + qyd*t0,
        qxb*u1 + qxd*t1, qyb*u1 + qyd*t1,
    );
}

/// K/L accumulator for one clipped cubic sub-curve (M&S §2.2, Appendix A).
///
/// K uses endpoints; L uses the Bernstein product integral for ∫xy dθ,
/// factored into two-term determinants (d01..d23) plus endpoint self-products.
fn accumulate_cubic_sub_kl(kl: ptr<function, KL>,
    sx0: f32, sy0: f32, sx1: f32, sy1: f32,
    sx2: f32, sy2: f32, sx3: f32, sy3: f32,
    skip_x: bool, skip_y: bool) {
    // v0=end, v1=cp2, v2=cp1, v3=start (paper's reversed convention)
    let v0x = sx3; let v0y = sy3;
    let v1x = sx2; let v1y = sy2;
    let v2x = sx1; let v2y = sy1;
    let v3x = sx0; let v3y = sy0;

    if skip_x && ((abs(v0x-1.0)<EPS && abs(v1x-1.0)<EPS && abs(v2x-1.0)<EPS && abs(v3x-1.0)<EPS)
              || (abs(v0x)<EPS && abs(v1x)<EPS && abs(v2x)<EPS && abs(v3x)<EPS)) { return; }
    if skip_y && ((abs(v0y-1.0)<EPS && abs(v1y-1.0)<EPS && abs(v2y-1.0)<EPS && abs(v3y-1.0)<EPS)
              || (abs(v0y)<EPS && abs(v1y)<EPS && abs(v2y)<EPS && abs(v3y)<EPS)) { return; }

    (*kl).kx += 0.25 * (v0y - v3y);
    (*kl).ky += 0.25 * (v3x - v0x);

    let d01 = v0x * v1y - v0y * v1x;
    let d02 = v0x * v2y - v0y * v2x;
    let d03 = v0x * v3y - v0y * v3x;
    let d12 = v1x * v2y - v1y * v2x;
    let d13 = v1x * v3y - v1y * v3x;
    let d23 = v2x * v3y - v2y * v3x;
    let cross_sum = -6.0*d01 - 3.0*d02 - d03 - 3.0*d12 - 3.0*d13 - 6.0*d23;
    let self_diff = 10.0 * (v0x * v0y - v3x * v3y);
    (*kl).lx += (1.0/80.0) * (cross_sum + self_diff);
    (*kl).ly += (1.0/80.0) * (cross_sum - self_diff);
}

/// Newton refinement for cubic Bernstein B(t) = val.
///
/// 2 iterations using the Bernstein derivative (3·ΔB) polish closed-form
/// f32 roots to working precision.
fn newton_bernstein(b0: f32, b1: f32, b2: f32, b3: f32, val: f32, t_in: f32) -> f32 {
    var t = clamp(t_in, 0.0, 1.0);
    let d0 = b1 - b0;
    let d1 = b2 - b1;
    let d2 = b3 - b2;
    for (var iter: u32 = 0u; iter < 2u; iter++) {
        let u = 1.0 - t;
        let ft = b0*u*u*u + 3.0*b1*u*u*t + 3.0*b2*u*t*t + b3*t*t*t - val;
        let fpt = 3.0 * (d0*u*u + 2.0*d1*u*t + d2*t*t);
        if abs(fpt) > 1e-20 {
            t = clamp(t - ft / fpt, 0.0, 1.0);
        }
    }
    return t;
}

/// Cubic Bezier K/L clipped to [0,1]² (M&S §2.2, Appendix A).
fn cubic_get_kl(x0: f32, y0: f32, x1: f32, y1: f32,
                x2: f32, y2: f32, x3: f32, y3: f32,
                skip_x: bool, skip_y: bool) -> KL {
    var kl = kl_zero();

    let xn = min(min(x0,x1),min(x2,x3));
    let xx = max(max(x0,x1),max(x2,x3));
    let yn = min(min(y0,y1),min(y2,y3));
    let yx = max(max(y0,y1),max(y2,y3));

    let ax = -x0 + 3.0*x1 - 3.0*x2 + x3;
    let bx = 3.0*x0 - 6.0*x1 + 3.0*x2;
    let cx = 3.0*x1 - 3.0*x0;
    let dx = x0;
    let ay = -y0 + 3.0*y1 - 3.0*y2 + y3;
    let by = 3.0*y0 - 6.0*y1 + 3.0*y2;
    let cy = 3.0*y1 - 3.0*y0;
    let dy = y0;

    // Collect crossing parameters
    var ts: array<f32, 16>;
    var ts_len: u32 = 0u;
    ts[0] = 0.0; ts_len = 1u;

    // x-boundary crossings
    for (var vi: u32 = 0u; vi < 2u; vi++) {
        let val = f32(vi);
        if xn < val + EPS && xx > val - EPS {
            let roots = solve_cubic_f(ax, bx, cx, dx - val);
            let n = u32(roots.x);
            if n >= 1u { ts[ts_len] = newton_bernstein(x0, x1, x2, x3, val, roots.y); ts_len++; }
            if n >= 2u { ts[ts_len] = newton_bernstein(x0, x1, x2, x3, val, roots.z); ts_len++; }
            if n >= 3u { ts[ts_len] = newton_bernstein(x0, x1, x2, x3, val, roots.w); ts_len++; }
        }
    }
    // y-boundary crossings
    for (var vi: u32 = 0u; vi < 2u; vi++) {
        let val = select(0.0, 1.0, vi == 0u);
        if yn < val + EPS && yx > val - EPS {
            let roots = solve_cubic_f(ay, by, cy, dy - val);
            let n = u32(roots.x);
            if n >= 1u { ts[ts_len] = newton_bernstein(y0, y1, y2, y3, val, roots.y); ts_len++; }
            if n >= 2u { ts[ts_len] = newton_bernstein(y0, y1, y2, y3, val, roots.z); ts_len++; }
            if n >= 3u { ts[ts_len] = newton_bernstein(y0, y1, y2, y3, val, roots.w); ts_len++; }
        }
    }
    ts[ts_len] = 1.0; ts_len++;

    // Filter to [0,1] and in-unit-square (wide tolerance for crossing points)
    var filtered: array<f32, 16>;
    var flen: u32 = 0u;
    for (var i: u32 = 0u; i < ts_len; i++) {
        let t = ts[i];
        if t >= 0.0 && t <= 1.0 {
            let p = cubic_eval(x0, y0, x1, y1, x2, y2, x3, y3, t);
            if is_in_unit(p) {
                filtered[flen] = t;
                flen++;
            }
        }
    }

    // Bubble sort (max 16 elements)
    for (var i: u32 = 0u; i < flen; i++) {
        for (var j: u32 = i + 1u; j < flen; j++) {
            if filtered[j] < filtered[i] {
                let tmp = filtered[i];
                filtered[i] = filtered[j];
                filtered[j] = tmp;
            }
        }
    }

    // Process consecutive pairs
    var prev: f32 = -1e30;
    var last_valid: bool = false;
    var last_t: f32 = 0.0;
    for (var i: u32 = 0u; i < flen; i++) {
        let t = filtered[i];
        if t == prev { continue; }
        prev = t;
        if last_valid {
            let mid = (last_t + t) * 0.5;
            let mp = cubic_eval(x0, y0, x1, y1, x2, y2, x3, y3, mid);
            if is_in_unit(mp) {
                let sub = cubic_subsection(x0, y0, x1, y1, x2, y2, x3, y3, last_t, t);
                accumulate_cubic_sub_kl(&kl, sub[0], sub[1], sub[2], sub[3],
                                             sub[4], sub[5], sub[6], sub[7],
                                             skip_x, skip_y);
            }
        }
        last_valid = true;
        last_t = t;
    }

    return kl;
}

// ---- Cubic tent (equation 7 with Ψ̄; c01, c10, c11) ----

/// Exact ∫₀¹ v(t)·u'(t) dt for a cubic Bezier (Bernstein product).
///
/// v·u' is degree 5; Bernstein integration gives closed-form weights.
fn cubic_integral_v_du(u0: f32, v0: f32, u1: f32, v1: f32,
                        u2: f32, v2: f32, u3: f32, v3: f32) -> f32 {
    let du0 = u1 - u0;
    let du1 = u2 - u1;
    let du2 = u3 - u2;
    return (1.0 / 20.0) * (
        (10.0*v0 + 6.0*v1 + 3.0*v2 + v3) * du0
      + (4.0*v0 + 6.0*v1 + 6.0*v2 + 4.0*v3) * du1
      + (v0 + 3.0*v1 + 6.0*v2 + 10.0*v3) * du2
    );
}

/// Exact ∫₀¹ u(t)·v'(t) dt for a cubic Bezier (Bernstein product).
///
/// Computed directly (not via integration by parts) for f32 stability.
fn cubic_integral_u_dv(u0: f32, v0: f32, u1: f32, v1: f32,
                        u2: f32, v2: f32, u3: f32, v3: f32) -> f32 {
    let dv0 = v1 - v0;
    let dv1 = v2 - v1;
    let dv2 = v3 - v2;
    return (1.0 / 20.0) * (
        (10.0*u0 + 6.0*u1 + 3.0*u2 + u3) * dv0
      + (4.0*u0 + 6.0*u1 + 6.0*u2 + 4.0*u3) * dv1
      + (u0 + 3.0*u1 + 6.0*u2 + 10.0*u3) * dv2
    );
}

/// Cubic Bezier tent integral in cell coordinates [0,1]².
///
/// Returns vec3(c01, c10, c11). Both ∫Ψ̄(v)du and ∫Ψ̄(u)dv are computed
/// directly via Bernstein product for f32 stability. c11 = Σ sign_v · tent_u.
/// Includes bisection verification for crossings missed by the analytic solver.
fn cubic_tent(u0: f32, v0: f32, u1: f32, v1: f32,
               u2: f32, v2: f32, u3: f32, v3: f32) -> vec3<f32> {
    let u_min = min(min(u0,u1),min(u2,u3));
    let u_max = max(max(u0,u1),max(u2,u3));
    let v_min = min(min(v0,v1),min(v2,v3));
    let v_max = max(max(v0,v1),max(v2,v3));
    if u_max < -EPS || u_min > 1.0+EPS || v_max < -EPS || v_min > 1.0+EPS {
        return vec3(0.0);
    }

    let au = -u0 + 3.0*u1 - 3.0*u2 + u3;
    let bu = 3.0*u0 - 6.0*u1 + 3.0*u2;
    let cu_c = 3.0*u1 - 3.0*u0;
    let du_c = u0;
    let av = -v0 + 3.0*v1 - 3.0*v2 + v3;
    let bv = 3.0*v0 - 6.0*v1 + 3.0*v2;
    let cv_c = 3.0*v1 - 3.0*v0;
    let dv_c = v0;

    var ts: array<f32, 36>;
    var ts_len: u32 = 0u;
    ts[0] = 0.0; ts_len = 1u;

    // u crossings at 0, 0.5, 1
    for (var vi: u32 = 0u; vi < 3u; vi++) {
        let val = f32(vi) * 0.5;
        if u_min < val + EPS && u_max > val - EPS {
            let roots = solve_cubic_f(au, bu, cu_c, du_c - val);
            let n = u32(roots.x);
            if n >= 1u { ts[ts_len] = newton_bernstein(u0, u1, u2, u3, val, roots.y); ts_len++; }
            if n >= 2u { ts[ts_len] = newton_bernstein(u0, u1, u2, u3, val, roots.z); ts_len++; }
            if n >= 3u { ts[ts_len] = newton_bernstein(u0, u1, u2, u3, val, roots.w); ts_len++; }
        }
    }
    // v crossings at 0, 0.5, 1
    for (var vi: u32 = 0u; vi < 3u; vi++) {
        let val = f32(vi) * 0.5;
        if v_min < val + EPS && v_max > val - EPS {
            let roots = solve_cubic_f(av, bv, cv_c, dv_c - val);
            let n = u32(roots.x);
            if n >= 1u { ts[ts_len] = newton_bernstein(v0, v1, v2, v3, val, roots.y); ts_len++; }
            if n >= 2u { ts[ts_len] = newton_bernstein(v0, v1, v2, v3, val, roots.z); ts_len++; }
            if n >= 3u { ts[ts_len] = newton_bernstein(v0, v1, v2, v3, val, roots.w); ts_len++; }
        }
    }
    ts[ts_len] = 1.0; ts_len++;

    // When all control points are inside (0,1)², the convex hull property
    // guarantees the entire curve is inside. Skip bisection, is_in_unit filter,
    // and midpoint checks — saves many cubic_eval calls at coarse levels.
    let hull_inside = u_min > EPS && u_max < 1.0 - EPS && v_min > EPS && v_max < 1.0 - EPS;

    // Sort the analytic roots so we can bisection-verify adjacent pairs
    for (var i: u32 = 0u; i < ts_len; i++) {
        for (var j: u32 = i + 1u; j < ts_len; j++) {
            if ts[j] < ts[i] { let tmp = ts[i]; ts[i] = ts[j]; ts[j] = tmp; }
        }
    }

    // Bisection safety net: catch crossings the analytic solver missed.
    var extra: array<f32, 8>;
    var n_extra: u32 = 0u;
    for (var i: u32 = 0u; i + 1u < ts_len && n_extra < 8u; i++) {
        let ta = ts[i]; let tb = ts[i + 1u];
        if tb - ta < EPS { continue; }
        let pa = cubic_eval(u0, v0, u1, v1, u2, v2, u3, v3, ta);
        let pb = cubic_eval(u0, v0, u1, v1, u2, v2, u3, v3, tb);
        for (var bvi: u32 = 0u; bvi < 3u && n_extra < 8u; bvi++) {
            let bnd = f32(bvi) * 0.5;
            if v_min < bnd + EPS && v_max > bnd - EPS && (pa.y - bnd) * (pb.y - bnd) < 0.0 {
                var lo = ta; var hi = tb;
                for (var it: u32 = 0u; it < 16u; it++) {
                    let mid = (lo + hi) * 0.5;
                    if mid == lo || mid == hi { break; }
                    let vm = cubic_eval(u0, v0, u1, v1, u2, v2, u3, v3, mid).y;
                    if (vm - bnd) * (pa.y - bnd) > 0.0 { lo = mid; } else { hi = mid; }
                }
                extra[n_extra] = (lo + hi) * 0.5; n_extra++;
            }
        }
        for (var bui: u32 = 0u; bui < 3u && n_extra < 8u; bui++) {
            let bnd = f32(bui) * 0.5;
            if u_min < bnd + EPS && u_max > bnd - EPS && (pa.x - bnd) * (pb.x - bnd) < 0.0 {
                var lo = ta; var hi = tb;
                for (var it: u32 = 0u; it < 16u; it++) {
                    let mid = (lo + hi) * 0.5;
                    if mid == lo || mid == hi { break; }
                    let um = cubic_eval(u0, v0, u1, v1, u2, v2, u3, v3, mid).x;
                    if (um - bnd) * (pa.x - bnd) > 0.0 { lo = mid; } else { hi = mid; }
                }
                extra[n_extra] = (lo + hi) * 0.5; n_extra++;
            }
        }
    }
    for (var i: u32 = 0u; i < n_extra; i++) {
        ts[ts_len] = extra[i]; ts_len++;
    }

    var filtered: array<f32, 36>;
    var flen: u32 = 0u;
    for (var i: u32 = 0u; i < ts_len; i++) {
        let t = ts[i];
        if t >= 0.0 && t <= 1.0 {
            if hull_inside {
                filtered[flen] = t; flen++;
            } else {
                let p = cubic_eval(u0, v0, u1, v1, u2, v2, u3, v3, t);
                if is_in_unit_wide(p) { filtered[flen] = t; flen++; }
            }
        }
    }
    for (var i: u32 = 0u; i < flen; i++) {
        for (var j: u32 = i + 1u; j < flen; j++) {
            if filtered[j] < filtered[i] {
                let tmp = filtered[i]; filtered[i] = filtered[j]; filtered[j] = tmp;
            }
        }
    }

    var c01: f32 = 0.0;
    var c10: f32 = 0.0;
    var c11: f32 = 0.0;
    var prev: f32 = -1e30;
    var last_valid: bool = false;
    var last_t: f32 = 0.0;
    for (var i: u32 = 0u; i < flen; i++) {
        let t = filtered[i];
        if t == prev { continue; }
        prev = t;
        if last_valid {
            let mid_t = (last_t + t) * 0.5;
            let mp = cubic_eval(u0, v0, u1, v1, u2, v2, u3, v3, mid_t);
            if hull_inside || is_in_unit(mp) {
                let sub = cubic_subsection(u0, v0, u1, v1, u2, v2, u3, v3, last_t, t);
                let mid_p = cubic_eval(sub[0], sub[1], sub[2], sub[3], sub[4], sub[5], sub[6], sub[7], 0.5);
                let int_v_du = cubic_integral_v_du(sub[0], sub[1], sub[2], sub[3], sub[4], sub[5], sub[6], sub[7]);
                let int_u_dv = cubic_integral_u_dv(sub[0], sub[1], sub[2], sub[3], sub[4], sub[5], sub[6], sub[7]);

                let tent_v = select((sub[6] - sub[0]) - int_v_du, int_v_du, mid_p.y < 0.5);
                let tent_u = select((sub[7] - sub[1]) - int_u_dv, int_u_dv, mid_p.x < 0.5);

                c01 -= tent_v;
                c10 += tent_u;

                let sign_v = select(-1.0, 1.0, mid_p.y < 0.5);
                c11 += sign_v * tent_u;
            }
        }
        last_valid = true;
        last_t = t;
    }
    return vec3(c01, c10, c11);
}

// ---- Circular Arc K/L (waverast extension; M&S covers only polynomial segments) ----
//
// K uses endpoints (same formula as all types). L uses the closed-form trig
// antiderivative: Lx = r·cx/4·Δsinθ + r²(Δθ/8 + Δsin2θ/16).

/// Circular arc evaluation at parameter t ∈ [0,1] (linear angle interpolation).
fn arc_eval(cx: f32, cy: f32, r: f32, t0: f32, t1: f32, t: f32) -> vec2<f32> {
    let theta = t0 + t * (t1 - t0);
    return vec2(cx + r * cos(theta), cy + r * sin(theta));
}

/// Arc-axis crossing finder via bisection + Newton refinement.
fn arc_find_crossings(cx: f32, cy: f32, r: f32, theta0: f32, dtheta: f32,
                      axis: u32, val: f32,
                      ts: ptr<function, array<f32, 32>>, len: ptr<function, u32>) {
    var bounds: array<f32, 8>;
    var nb: u32 = 0u;
    bounds[nb] = 0.0; nb++;

    let offset = select(0.0, PI * 0.5, axis == 1u);
    let theta_end = theta0 + dtheta;
    let lo = min(theta0, theta_end);
    let hi = max(theta0, theta_end);
    let k_min = i32(ceil((lo - offset) / PI));
    let k_max = i32(floor((hi - offset) / PI));
    for (var k: i32 = k_min; k <= k_max; k++) {
        let theta = offset + f32(k) * PI;
        let t = (theta - theta0) / dtheta;
        if t > 0.0 && t < 1.0 && nb < 7u {
            bounds[nb] = t; nb++;
        }
    }
    bounds[nb] = 1.0; nb++;

    // Sort
    for (var i: u32 = 0u; i < nb; i++) {
        for (var j: u32 = i + 1u; j < nb; j++) {
            if bounds[j] < bounds[i] {
                let tmp = bounds[i]; bounds[i] = bounds[j]; bounds[j] = tmp;
            }
        }
    }

    // Bisect each monotone interval
    for (var i: u32 = 0u; i < nb - 1u; i++) {
        if *len >= 32u { break; }
        let t0_b = bounds[i];
        let t1_b = bounds[i + 1u];

        let p0 = arc_eval(cx, cy, r, theta0, theta0 + dtheta, t0_b);
        let p1 = arc_eval(cx, cy, r, theta0, theta0 + dtheta, t1_b);
        let v0 = select(p0.x, p0.y, axis == 1u) - val;
        let v1 = select(p1.x, p1.y, axis == 1u) - val;

        if abs(v0) < 1e-10 && i == 0u {
            (*ts)[*len] = t0_b; *len = *len + 1u; continue;
        }
        if (v0 > 0.0) == (v1 > 0.0) { continue; }

        // Bisect
        var lo_t = t0_b;
        var hi_t = t1_b;
        var flo = v0;
        for (var iter: u32 = 0u; iter < 24u; iter++) {
            let mid = (lo_t + hi_t) * 0.5;
            if mid == lo_t || mid == hi_t { break; }
            let pm = arc_eval(cx, cy, r, theta0, theta0 + dtheta, mid);
            let vm = select(pm.x, pm.y, axis == 1u) - val;
            if (vm > 0.0) == (flo > 0.0) { lo_t = mid; flo = vm; }
            else { hi_t = mid; }
        }
        // Newton refinement
        var t_root = (lo_t + hi_t) * 0.5;
        let theta_r = theta0 + t_root * dtheta;
        let sin_r = sin(theta_r);
        let cos_r = cos(theta_r);
        var ft: f32;
        var fpt: f32;
        if axis == 0u {
            ft = cx + r * cos_r - val;
            fpt = -r * sin_r * dtheta;
        } else {
            ft = cy + r * sin_r - val;
            fpt = r * cos_r * dtheta;
        }
        if abs(fpt) > 1e-20 {
            t_root = clamp(t_root - ft / fpt, 0.0, 1.0);
        }
        (*ts)[*len] = t_root;
        *len = *len + 1u;
    }
}

// ---- Arc tent (equation 7 with Ψ̄; c01, c10, c11) ----

/// Circular arc tent integral in cell coordinates [0,1]².
///
/// All three coefficients via closed-form trig expressions. Uses conditioned
/// forms to avoid catastrophic cancellation when the sub-arc nearly fills a
/// cell half:
///   tent_v (v≥0.5): r·Δcos·(1-cy) + r²(Δθ/2 - Δsin2θ/4)
///   tent_u (u≥0.5): r·Δsin·(1-cx) - r²(Δθ/2 + Δsin2θ/4)
/// Trig differences use product-to-sum identities for stability.
fn arc_tent(cx: f32, cy: f32, r: f32, theta0: f32, theta1: f32) -> vec3<f32> {
    let dtheta = theta1 - theta0;
    if abs(dtheta) < EPS { return vec3(0.0); }

    var ts: array<f32, 32>;
    var ts_len: u32 = 0u;
    ts[0] = 0.0; ts_len = 1u;

    // x crossings at 0, 0.5, 1
    for (var vi: u32 = 0u; vi < 3u; vi++) {
        arc_find_crossings(cx, cy, r, theta0, dtheta, 0u, f32(vi) * 0.5, &ts, &ts_len);
    }
    // y crossings at 0, 0.5, 1
    for (var vi: u32 = 0u; vi < 3u; vi++) {
        arc_find_crossings(cx, cy, r, theta0, dtheta, 1u, f32(vi) * 0.5, &ts, &ts_len);
    }
    ts[ts_len] = 1.0; ts_len++;

    var filtered: array<f32, 32>;
    var flen: u32 = 0u;
    for (var i: u32 = 0u; i < ts_len; i++) {
        let t = ts[i];
        if t >= 0.0 && t <= 1.0 {
            let p = arc_eval(cx, cy, r, theta0, theta1, t);
            if is_in_unit_wide(p) { filtered[flen] = t; flen++; }
        }
    }
    for (var i: u32 = 0u; i < flen; i++) {
        for (var j: u32 = i+1u; j < flen; j++) {
            if filtered[j] < filtered[i] {
                let tmp = filtered[i]; filtered[i] = filtered[j]; filtered[j] = tmp;
            }
        }
    }

    var c01_sum: f32 = 0.0;
    var c10_sum: f32 = 0.0;
    var c11_sum: f32 = 0.0;
    var prev: f32 = -1e30;
    var last_valid: bool = false;
    var last_t: f32 = 0.0;
    for (var i: u32 = 0u; i < flen; i++) {
        let t = filtered[i];
        if t == prev { continue; }
        prev = t;
        if last_valid {
            let mid_param = (last_t + t) * 0.5;
            let mp = arc_eval(cx, cy, r, theta0, theta1, mid_param);
            if is_in_unit(mp) {
                let alpha = theta0 + last_t * dtheta;
                let beta = theta0 + t * dtheta;
                let dt = beta - alpha;

                let half_sum = (alpha + beta) * 0.5;
                let half_dt = dt * 0.5;
                let dsin = 2.0 * cos(half_sum) * sin(half_dt);
                let dcos = -2.0 * sin(half_sum) * sin(half_dt);
                let dsin2 = 2.0 * cos(alpha + beta) * sin(dt);

                let r2 = r * r;
                let theta_mid = (alpha + beta) * 0.5;
                let v_mid = cy + r * sin(theta_mid);
                let u_mid = cx + r * cos(theta_mid);

                // Conditioned tent integrals: avoid catastrophic cancellation
                // when delta ≈ int (near-full coverage).
                // tent_v = ∫Ψ̄(v)du: when v<0.5, = int_v_du; when v≥0.5, = delta_u - int_v_du
                //   delta_u - int_v_du = r*dcos*(1-cy) + r²*(dt/2 - dsin2/4)
                // tent_u = ∫Ψ̄(u)dv: when u<0.5, = int_u_dv; when u≥0.5, = delta_v - int_u_dv
                //   delta_v - int_u_dv = r*dsin*(1-cx) - r²*(dt/2 + dsin2/4)
                var tent_v: f32;
                if v_mid < 0.5 {
                    tent_v = r * cy * dcos - r2 * dt * 0.5 + r2 * dsin2 * 0.25;
                } else {
                    tent_v = r * dcos * (1.0 - cy) + r2 * (dt * 0.5 - dsin2 * 0.25);
                }
                var tent_u: f32;
                if u_mid < 0.5 {
                    tent_u = r * cx * dsin + r2 * dt * 0.5 + r2 * dsin2 * 0.25;
                } else {
                    tent_u = r * dsin * (1.0 - cx) - r2 * (dt * 0.5 + dsin2 * 0.25);
                }

                c01_sum += tent_v;
                c10_sum += tent_u;

                // c11 = sign_v * tent_u
                let sign_v = select(-1.0, 1.0, v_mid < 0.5);
                c11_sum += sign_v * tent_u;
            }
        }
        last_valid = true;
        last_t = t;
    }
    return vec3(-c01_sum, c10_sum, c11_sum);
}

// ---- Quadratic Bezier K/L (M&S Appendix A, degree-2 case) ----

/// Quadratic Bernstein evaluation B(t) = Σ bᵢ·Bᵢ²(t).
fn quad_eval(x0: f32, y0: f32, x1: f32, y1: f32,
             x2: f32, y2: f32, t: f32) -> vec2<f32> {
    let u = 1.0 - t;
    return vec2(
        x0 * u * u + 2.0 * x1 * u * t + x2 * t * t,
        y0 * u * u + 2.0 * y1 * u * t + y2 * t * t,
    );
}

/// Quadratic Bezier sub-curve extraction for t ∈ [t0, t1].
fn quad_subsection(x0: f32, y0: f32, x1: f32, y1: f32,
                   x2: f32, y2: f32,
                   t0: f32, t1: f32) -> array<f32, 6> {
    let u0 = 1.0 - t0;
    let xm = u0 * x1 + t0 * x2;
    let ym = u0 * y1 + t0 * y2;
    let sx0 = x0*u0*u0 + 2.0*x1*u0*t0 + x2*t0*t0;
    let sy0 = y0*u0*u0 + 2.0*y1*u0*t0 + y2*t0*t0;
    let sx2 = x0*(1.0-t1)*(1.0-t1) + 2.0*x1*(1.0-t1)*t1 + x2*t1*t1;
    let sy2 = y0*(1.0-t1)*(1.0-t1) + 2.0*y1*(1.0-t1)*t1 + y2*t1*t1;
    let t_rel = (t1 - t0) / u0;
    let sx1 = (1.0 - t_rel) * sx0 + t_rel * xm;
    let sy1 = (1.0 - t_rel) * sy0 + t_rel * ym;
    return array<f32, 6>(sx0, sy0, sx1, sy1, sx2, sy2);
}

/// K/L accumulator for one clipped quadratic sub-curve (M&S Appendix A).
///
/// L uses the Bernstein product integral for ∫xy dθ.
fn accumulate_quad_sub_kl(kl: ptr<function, KL>,
    sx0: f32, sy0: f32, sx1: f32, sy1: f32, sx2: f32, sy2: f32,
    skip_x: bool, skip_y: bool) {
    // v0=end, v1=cp, v2=start (reversed convention)
    let v0x = sx2; let v0y = sy2;
    let v1x = sx1; let v1y = sy1;
    let v2x = sx0; let v2y = sy0;

    if skip_x && ((abs(v0x-1.0)<EPS && abs(v1x-1.0)<EPS && abs(v2x-1.0)<EPS)
              || (abs(v0x)<EPS && abs(v1x)<EPS && abs(v2x)<EPS)) { return; }
    if skip_y && ((abs(v0y-1.0)<EPS && abs(v1y-1.0)<EPS && abs(v2y-1.0)<EPS)
              || (abs(v0y)<EPS && abs(v1y)<EPS && abs(v2y)<EPS)) { return; }

    (*kl).kx += 0.25 * (v0y - v2y);
    (*kl).ky += 0.25 * (v2x - v0x);

    let d01 = v0x * v1y - v0y * v1x;
    let d02 = v0x * v2y - v0y * v2x;
    let d12 = v1x * v2y - v1y * v2x;
    let cross_sum = -2.0*d01 - d02 - 2.0*d12;
    let self_diff = 3.0 * (v0x * v0y - v2x * v2y);
    (*kl).lx += (1.0/24.0) * (cross_sum + self_diff);
    (*kl).ly += (1.0/24.0) * (cross_sum - self_diff);
}

/// Newton refinement for quadratic Bernstein B(t) = val.
fn newton_bernstein_quad(b0: f32, b1: f32, b2: f32, val: f32, t_in: f32) -> f32 {
    var t = clamp(t_in, 0.0, 1.0);
    for (var iter: u32 = 0u; iter < 4u; iter++) {
        let u = 1.0 - t;
        let ft = b0*u*u + 2.0*b1*u*t + b2*t*t - val;
        let fpt = 2.0 * ((b1 - b0)*u + (b2 - b1)*t);
        if abs(fpt) > 1e-20 {
            t = clamp(t - ft / fpt, 0.0, 1.0);
        }
    }
    return t;
}

/// Quadratic Bezier K/L clipped to [0,1]² (M&S Appendix A, degree-2).
fn quad_get_kl(x0: f32, y0: f32, x1: f32, y1: f32, x2: f32, y2: f32,
               skip_x: bool, skip_y: bool) -> KL {
    var kl = kl_zero();

    let xn = min(min(x0,x1),x2);
    let xx = max(max(x0,x1),x2);
    let yn = min(min(y0,y1),y2);
    let yx = max(max(y0,y1),y2);

    let ax = x0 - 2.0*x1 + x2;
    let bx = -2.0*x0 + 2.0*x1;
    let cx = x0;
    let ay = y0 - 2.0*y1 + y2;
    let by = -2.0*y0 + 2.0*y1;
    let cy = y0;

    var ts: array<f32, 12>;
    var ts_len: u32 = 0u;
    ts[0] = 0.0; ts_len = 1u;

    for (var vi: u32 = 0u; vi < 2u; vi++) {
        let val = f32(vi);
        if xn < val + EPS && xx > val - EPS {
            let roots = solve_quadratic_f(ax, bx, cx - val);
            let n = u32(roots.x);
            if n >= 1u { ts[ts_len] = newton_bernstein_quad(x0, x1, x2, val, roots.y); ts_len++; }
            if n >= 2u { ts[ts_len] = newton_bernstein_quad(x0, x1, x2, val, roots.z); ts_len++; }
        }
    }
    for (var vi: u32 = 0u; vi < 2u; vi++) {
        let val = select(0.0, 1.0, vi == 0u);
        if yn < val + EPS && yx > val - EPS {
            let roots = solve_quadratic_f(ay, by, cy - val);
            let n = u32(roots.x);
            if n >= 1u { ts[ts_len] = newton_bernstein_quad(y0, y1, y2, val, roots.y); ts_len++; }
            if n >= 2u { ts[ts_len] = newton_bernstein_quad(y0, y1, y2, val, roots.z); ts_len++; }
        }
    }
    ts[ts_len] = 1.0; ts_len++;

    // Filter, sort, process pairs
    var filtered: array<f32, 12>;
    var flen: u32 = 0u;
    for (var i: u32 = 0u; i < ts_len; i++) {
        let t = ts[i];
        if t >= 0.0 && t <= 1.0 {
            let p = quad_eval(x0, y0, x1, y1, x2, y2, t);
            if is_in_unit(p) { filtered[flen] = t; flen++; }
        }
    }
    for (var i: u32 = 0u; i < flen; i++) {
        for (var j: u32 = i+1u; j < flen; j++) {
            if filtered[j] < filtered[i] {
                let tmp = filtered[i]; filtered[i] = filtered[j]; filtered[j] = tmp;
            }
        }
    }
    var prev: f32 = -1e30;
    var last_valid: bool = false;
    var last_t: f32 = 0.0;
    for (var i: u32 = 0u; i < flen; i++) {
        let t = filtered[i];
        if t == prev { continue; }
        prev = t;
        if last_valid {
            let mid = (last_t + t) * 0.5;
            let mp = quad_eval(x0, y0, x1, y1, x2, y2, mid);
            if is_in_unit(mp) {
                let sub = quad_subsection(x0, y0, x1, y1, x2, y2, last_t, t);
                accumulate_quad_sub_kl(&kl, sub[0], sub[1], sub[2], sub[3], sub[4], sub[5],
                                       skip_x, skip_y);
            }
        }
        last_valid = true;
        last_t = t;
    }
    return kl;
}

// ---- Quad tent (equation 7 with Ψ̄; c01/c10 only) ----

/// Exact ∫₀¹ v(t)·u'(t) dt for a quadratic Bezier (Bernstein product).
fn quad_integral_v_du(u0: f32, v0: f32, u1: f32, v1: f32, u2: f32, v2: f32) -> f32 {
    let du0 = u1 - u0;
    let du1 = u2 - u1;
    return (1.0/6.0) * ((3.0*v0 + 2.0*v1 + v2) * du0 + (v0 + 2.0*v1 + 3.0*v2) * du1);
}

/// ∫Ψ̄(v) du for a quadratic Bezier, with clipping and midpoint splitting.
fn quad_tent_integral(u0: f32, v0: f32, u1: f32, v1: f32, u2: f32, v2: f32) -> f32 {
    let u_min = min(min(u0,u1),u2);
    let u_max = max(max(u0,u1),u2);
    let v_min = min(min(v0,v1),v2);
    let v_max = max(max(v0,v1),v2);
    if u_max < -EPS || u_min > 1.0+EPS || v_max < -EPS || v_min > 1.0+EPS {
        return 0.0;
    }

    let au = u0 - 2.0*u1 + u2;
    let bu = -2.0*u0 + 2.0*u1;
    let cu_c = u0;
    let av = v0 - 2.0*v1 + v2;
    let bv = -2.0*v0 + 2.0*v1;
    let cv_c = v0;

    var ts: array<f32, 16>;
    var ts_len: u32 = 0u;
    ts[0] = 0.0; ts_len = 1u;

    // u crossings at 0, 1
    for (var vi: u32 = 0u; vi < 2u; vi++) {
        let val = f32(vi);
        if u_min < val + EPS && u_max > val - EPS {
            let roots = solve_quadratic_f(au, bu, cu_c - val);
            let n = u32(roots.x);
            if n >= 1u { ts[ts_len] = newton_bernstein_quad(u0, u1, u2, val, roots.y); ts_len++; }
            if n >= 2u { ts[ts_len] = newton_bernstein_quad(u0, u1, u2, val, roots.z); ts_len++; }
        }
    }
    // v crossings at 0, 0.5, 1
    for (var vi: u32 = 0u; vi < 3u; vi++) {
        let val = f32(vi) * 0.5;
        if v_min < val + EPS && v_max > val - EPS {
            let roots = solve_quadratic_f(av, bv, cv_c - val);
            let n = u32(roots.x);
            if n >= 1u { ts[ts_len] = newton_bernstein_quad(v0, v1, v2, val, roots.y); ts_len++; }
            if n >= 2u { ts[ts_len] = newton_bernstein_quad(v0, v1, v2, val, roots.z); ts_len++; }
        }
    }
    ts[ts_len] = 1.0; ts_len++;

    var filtered: array<f32, 16>;
    var flen: u32 = 0u;
    for (var i: u32 = 0u; i < ts_len; i++) {
        let t = ts[i];
        if t >= 0.0 && t <= 1.0 {
            let p = quad_eval(u0, v0, u1, v1, u2, v2, t);
            if is_in_unit_wide(p) { filtered[flen] = t; flen++; }
        }
    }
    for (var i: u32 = 0u; i < flen; i++) {
        for (var j: u32 = i+1u; j < flen; j++) {
            if filtered[j] < filtered[i] {
                let tmp = filtered[i]; filtered[i] = filtered[j]; filtered[j] = tmp;
            }
        }
    }

    var result: f32 = 0.0;
    var prev_val: f32 = -1e30;
    var last_valid: bool = false;
    var last_t: f32 = 0.0;
    for (var i: u32 = 0u; i < flen; i++) {
        let t = filtered[i];
        if t == prev_val { continue; }
        prev_val = t;
        if last_valid {
            let mid_t = (last_t + t) * 0.5;
            let mp = quad_eval(u0, v0, u1, v1, u2, v2, mid_t);
            if is_in_unit(mp) {
                let sub = quad_subsection(u0, v0, u1, v1, u2, v2, last_t, t);
                let v_mid = quad_eval(sub[0], sub[1], sub[2], sub[3], sub[4], sub[5], 0.5).y;
                let int_v_du_val = quad_integral_v_du(sub[0], sub[1], sub[2], sub[3], sub[4], sub[5]);
                if v_mid < 0.5 {
                    result += int_v_du_val;
                } else {
                    result += (sub[4] - sub[0]) - int_v_du_val;
                }
            }
        }
        last_valid = true;
        last_t = t;
    }
    return result;
}

/// Quadratic Bezier tent integral returning vec2(c01, c10).
fn quad_tent(u0: f32, v0: f32, u1: f32, v1: f32, u2: f32, v2: f32) -> vec2<f32> {
    let c01 = -quad_tent_integral(u0, v0, u1, v1, u2, v2);
    let c10 = quad_tent_integral(v0, u0, v1, u1, v2, u2);
    return vec2(c01, c10);
}

// ---- Superellipse K/L (waverast extension; not in M&S) ----
//
// A convex superellipse |x/a|^n + |y/b|^n = 1 (n ≥ 1) is parameterized per-quadrant
// by u ∈ [0,1]: x = cx + sx·a·u, y = cy + sy·b·(1-u^n)^(1/n). The L integral
// uses GL-16 numerical quadrature on ∫y dx (smooth: dx/du = sx·a is constant),
// and obtains ∫x dy via integration by parts: ∫x dy = [xy] - ∫y dx.
// This avoids the singular dy/du derivative at u = 1 (the profile cusp).
// Sub-intervals are split at u=0.75 and u=0.9375 near the cusp for GL-16 accuracy.

/// Superellipse profile: (1 - u^n)^(1/n).
fn se_profile(u: f32, n: f32) -> f32 {
    return pow(max(1.0 - pow(u, n), 0.0), 1.0 / n);
}

/// Superellipse quadrant evaluation at parameter u ∈ [0, 1].
fn se_evaluate(cx: f32, cy: f32, a: f32, b: f32, n: f32,
               sx: f32, sy: f32, u: f32) -> vec2<f32> {
    return vec2(cx + sx * a * u, cy + sy * b * se_profile(u, n));
}

// Gauss-Legendre 16-point quadrature nodes and weights (symmetric, 8 pairs).
const GL_NODES: array<f32, 8> = array<f32, 8>(
    0.09501251, 0.28160355, 0.45801678, 0.61787624,
    0.75540441, 0.86563120, 0.94457502, 0.98940093,
);
const GL_WEIGHTS: array<f32, 8> = array<f32, 8>(
    0.18945061, 0.18260342, 0.16915652, 0.14959599,
    0.12462897, 0.09515851, 0.06225352, 0.02715246,
);

/// K/L for one superellipse quadrant, clipped to [0,1]².
fn se_quadrant_kl(kl: ptr<function, KL>,
    cx: f32, cy: f32, a: f32, b: f32, n: f32, sx: f32, sy: f32,
    skip_x: bool, skip_y: bool) {

    var ts: array<f32, 12>;
    var ts_len: u32 = 0u;
    ts[0] = 0.0; ts_len = 1u;
    // Split near u=1 where the profile has a cusp, improving GL-16 accuracy
    ts[ts_len] = 0.75; ts_len++;
    ts[ts_len] = 0.9375; ts_len++;

    // x = val crossings: u = (val - cx) / (sx * a)
    if abs(sx) * a > EPS {
        for (var vi: u32 = 0u; vi < 2u; vi++) {
            let val = f32(vi);
            let u_cross = (val - cx) / (sx * a);
            if u_cross >= 0.0 && u_cross <= 1.0 {
                ts[ts_len] = u_cross; ts_len++;
            }
        }
    }

    // y = val crossings: u = profile(d, n) where d = (val - cy) / (sy * b)
    if abs(sy) * b > EPS {
        for (var vi: u32 = 0u; vi < 2u; vi++) {
            let val = f32(vi);
            let d = (val - cy) / (sy * b);
            if d >= 0.0 && d <= 1.0 {
                ts[ts_len] = se_profile(d, n); ts_len++;
            }
        }
    }

    ts[ts_len] = 1.0; ts_len++;

    // Filter and sort
    var filtered: array<f32, 12>;
    var flen: u32 = 0u;
    for (var i: u32 = 0u; i < ts_len; i++) {
        let t = ts[i];
        if t >= 0.0 && t <= 1.0 {
            let p = se_evaluate(cx, cy, a, b, n, sx, sy, t);
            if is_in_unit(p) { filtered[flen] = t; flen++; }
        }
    }
    for (var i: u32 = 0u; i < flen; i++) {
        for (var j: u32 = i+1u; j < flen; j++) {
            if filtered[j] < filtered[i] {
                let tmp = filtered[i]; filtered[i] = filtered[j]; filtered[j] = tmp;
            }
        }
    }

    // Process consecutive pairs
    var prev: f32 = -1e30;
    var last_valid: bool = false;
    var last_t: f32 = 0.0;
    for (var i: u32 = 0u; i < flen; i++) {
        let t = filtered[i];
        if t == prev { continue; }
        prev = t;
        if last_valid {
            let mid = (last_t + t) * 0.5;
            let mp = se_evaluate(cx, cy, a, b, n, sx, sy, mid);
            if is_in_unit(mp) {
                let dir = -sx * sy;
                var u_start = last_t;
                var u_end = t;
                if dir < 0.0 { let tmp = u_start; u_start = u_end; u_end = tmp; }
                let pa = se_evaluate(cx, cy, a, b, n, sx, sy, u_start);
                let pb = se_evaluate(cx, cy, a, b, n, sx, sy, u_end);

                let skip_edge = (skip_x && ((abs(pa.x - 1.0) < EPS && abs(pb.x - 1.0) < EPS)
                                          || (abs(pa.x) < EPS && abs(pb.x) < EPS)))
                             || (skip_y && ((abs(pa.y - 1.0) < EPS && abs(pb.y - 1.0) < EPS)
                                          || (abs(pa.y) < EPS && abs(pb.y) < EPS)));
                if !skip_edge {
                    (*kl).kx += 0.25 * (pb.y - pa.y);
                    (*kl).ky += 0.25 * (pa.x - pb.x);

                    // ∫y dx via GL-16
                    let half = (t - last_t) * 0.5;
                    let mid_u = (last_t + t) * 0.5;
                    var gl_sum: f32 = 0.0;
                    for (var gi: u32 = 0u; gi < 8u; gi++) {
                        let x_node = half * GL_NODES[gi];
                        let y_neg = cy + sy * b * se_profile(mid_u - x_node, n);
                        let y_pos = cy + sy * b * se_profile(mid_u + x_node, n);
                        gl_sum += GL_WEIGHTS[gi] * (y_neg + y_pos);
                    }
                    let i_y = gl_sum * half;
                    let int_y_dx = dir * sx * a * i_y;
                    let int_x_dy = (pb.x * pb.y - pa.x * pa.y) - int_y_dx;

                    (*kl).lx += 0.25 * int_x_dy;
                    (*kl).ly += -0.25 * int_y_dx;
                }
            }
        }
        last_valid = true;
        last_t = t;
    }
}

const SE_QUADRANTS: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2(1.0, 1.0), vec2(-1.0, 1.0), vec2(-1.0, -1.0), vec2(1.0, -1.0),
);

/// Superellipse K/L for all active quadrants.
fn se_get_kl(cx: f32, cy: f32, a: f32, b: f32, n: f32, quadrants: u32,
             skip_x: bool, skip_y: bool) -> KL {
    var kl = kl_zero();
    for (var qi: u32 = 0u; qi < 4u; qi++) {
        if (quadrants & (1u << qi)) != 0u {
            let q = SE_QUADRANTS[qi];
            se_quadrant_kl(&kl, cx, cy, a, b, n, q.x, q.y, skip_x, skip_y);
        }
    }
    return kl;
}

// ---- Superellipse tent (equation 7 with Ψ̄; c01/c10 only) ----

/// Tent integral for one superellipse quadrant, using GL-16 for ∫y dx.
fn se_quadrant_tent(cx: f32, cy: f32, a: f32, b: f32, n: f32,
                     sx: f32, sy: f32) -> vec2<f32> {
    var ts: array<f32, 16>;
    var ts_len: u32 = 0u;
    ts[0] = 0.0; ts_len = 1u;
    // Split near u=1 where the profile has a cusp, improving GL-16 accuracy
    ts[ts_len] = 0.75; ts_len++;
    ts[ts_len] = 0.9375; ts_len++;

    // x crossings at 0, 0.5, 1
    if abs(sx) * a > EPS {
        for (var vi: u32 = 0u; vi < 3u; vi++) {
            let val = f32(vi) * 0.5;
            let u_cross = (val - cx) / (sx * a);
            if u_cross >= 0.0 && u_cross <= 1.0 {
                ts[ts_len] = u_cross; ts_len++;
            }
        }
    }

    // y crossings at 0, 0.5, 1
    if abs(sy) * b > EPS {
        for (var vi: u32 = 0u; vi < 3u; vi++) {
            let val = f32(vi) * 0.5;
            let d = (val - cy) / (sy * b);
            if d >= 0.0 && d <= 1.0 {
                ts[ts_len] = se_profile(d, n); ts_len++;
            }
        }
    }
    ts[ts_len] = 1.0; ts_len++;

    // Filter, sort
    var filtered: array<f32, 16>;
    var flen: u32 = 0u;
    for (var i: u32 = 0u; i < ts_len; i++) {
        let t = ts[i];
        if t >= 0.0 && t <= 1.0 {
            let p = se_evaluate(cx, cy, a, b, n, sx, sy, t);
            if is_in_unit_wide(p) { filtered[flen] = t; flen++; }
        }
    }
    for (var i: u32 = 0u; i < flen; i++) {
        for (var j: u32 = i+1u; j < flen; j++) {
            if filtered[j] < filtered[i] {
                let tmp = filtered[i]; filtered[i] = filtered[j]; filtered[j] = tmp;
            }
        }
    }

    let dir = -sx * sy;
    var c01_sum: f32 = 0.0;
    var c10_sum: f32 = 0.0;
    var prev: f32 = -1e30;
    var last_valid: bool = false;
    var last_t: f32 = 0.0;
    for (var i: u32 = 0u; i < flen; i++) {
        let t = filtered[i];
        if t == prev { continue; }
        prev = t;
        if last_valid {
            let mid_param = (last_t + t) * 0.5;
            let mp = se_evaluate(cx, cy, a, b, n, sx, sy, mid_param);
            if is_in_unit(mp) {
                var u_start = last_t;
                var u_end = t;
                if dir < 0.0 { let tmp = u_start; u_start = u_end; u_end = tmp; }
                let pa = se_evaluate(cx, cy, a, b, n, sx, sy, u_start);
                let pb = se_evaluate(cx, cy, a, b, n, sx, sy, u_end);

                let half = (t - last_t) * 0.5;
                let mid_u = (last_t + t) * 0.5;
                var gl_sum: f32 = 0.0;
                for (var gi: u32 = 0u; gi < 8u; gi++) {
                    let x_node = half * GL_NODES[gi];
                    let y_neg = cy + sy * b * se_profile(mid_u - x_node, n);
                    let y_pos = cy + sy * b * se_profile(mid_u + x_node, n);
                    gl_sum += GL_WEIGHTS[gi] * (y_neg + y_pos);
                }
                let i_y = gl_sum * half;
                let int_y_dx = dir * sx * a * i_y;
                let int_x_dy = (pb.x * pb.y - pa.x * pa.y) - int_y_dx;

                let delta_x = pb.x - pa.x;
                let delta_y = pb.y - pa.y;

                if mp.y < 0.5 {
                    c01_sum += int_y_dx;
                } else {
                    c01_sum += delta_x - int_y_dx;
                }
                if mp.x < 0.5 {
                    c10_sum += int_x_dy;
                } else {
                    c10_sum += delta_y - int_x_dy;
                }
            }
        }
        last_valid = true;
        last_t = t;
    }
    return vec2(c01_sum, c10_sum);
}

/// Superellipse tent integral returning vec2(c01, c10).
fn se_tent(cx: f32, cy: f32, a: f32, b: f32, n: f32, quadrants: u32) -> vec2<f32> {
    var c01: f32 = 0.0;
    var c10: f32 = 0.0;
    for (var qi: u32 = 0u; qi < 4u; qi++) {
        if (quadrants & (1u << qi)) != 0u {
            let q = SE_QUADRANTS[qi];
            let r = se_quadrant_tent(cx, cy, a, b, n, q.x, q.y);
            c01 += r.x;
            c10 += r.y;
        }
    }
    return vec2(-c01, c10);
}

// ---- Entry points ----
//
// Each entry point iterates over wavelet levels and overlapping cells.
// Lines, cubics, arcs: all three coefficients via tent integrals (no equation-8).
// Quads, superellipses: c11 via equation-8, c01/c10 via tent integrals.

/// Line segment entry point. All three coefficients via tent integral.
@compute @workgroup_size(64)
fn compute_lines(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.num_segments { return; }

    let s = segments[idx];
    let x0 = s.x; let y0 = s.y; let x1 = s.z; let y1 = s.w;
    let bbox_min = vec2(min(x0, x1), min(y0, y1));
    let bbox_max = vec2(max(x0, x1), max(y0, y1));

    for (var j: u32 = 0u; j <= params.max_j; j++) {
        let cells = 1u << j;
        let cellsf = f32(cells);
        let base = level_offset(j);

        let kx_lo = u32(max(floor(bbox_min.x * cellsf), 0.0));
        let kx_hi = min(u32(ceil(bbox_max.x * cellsf)), cells);
        let ky_lo = u32(max(floor(bbox_min.y * cellsf), 0.0));
        let ky_hi = min(u32(ceil(bbox_max.y * cellsf)), cells);

        for (var kx: u32 = kx_lo; kx < kx_hi; kx++) {
            let kxf = f32(kx);
            for (var ky: u32 = ky_lo; ky < ky_hi; ky++) {
                let kyf = f32(ky);
                let cell_idx = base + kx * cells + ky;

                let tent_result = line_tent(
                    cellsf * x0 - kxf, cellsf * y0 - kyf,
                    cellsf * x1 - kxf, cellsf * y1 - kyf);
                atomic_add_f32(cell_idx * 3u + 0u, tent_result.x);
                atomic_add_f32(cell_idx * 3u + 1u, tent_result.y);
                atomic_add_f32(cell_idx * 3u + 2u, tent_result.z);
            }
        }
    }
}

/// Cubic Bezier entry point. All three coefficients via tent integral.
///
/// Segments: 2 × vec4 per cubic ([0] = x0,y0,x1,y1; [1] = x2,y2,x3,y3).
@compute @workgroup_size(256)
fn compute_cubics(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.num_segments { return; }

    let s0 = segments[idx * 2u];
    let s1 = segments[idx * 2u + 1u];
    let x0 = s0.x; let y0 = s0.y; let x1 = s0.z; let y1 = s0.w;
    let x2 = s1.x; let y2 = s1.y; let x3 = s1.z; let y3 = s1.w;

    let bbox_min = vec2(min(min(x0,x1),min(x2,x3)), min(min(y0,y1),min(y2,y3)));
    let bbox_max = vec2(max(max(x0,x1),max(x2,x3)), max(max(y0,y1),max(y2,y3)));

    for (var j: u32 = 0u; j <= params.max_j; j++) {
        let cells = 1u << j;
        let cellsf = f32(cells);
        let base = level_offset(j);

        let kx_lo = u32(max(floor(bbox_min.x * cellsf), 0.0));
        let kx_hi = min(u32(ceil(bbox_max.x * cellsf)), cells);
        let ky_lo = u32(max(floor(bbox_min.y * cellsf), 0.0));
        let ky_hi = min(u32(ceil(bbox_max.y * cellsf)), cells);

        for (var kx: u32 = kx_lo; kx < kx_hi; kx++) {
            let kxf = f32(kx);
            for (var ky: u32 = ky_lo; ky < ky_hi; ky++) {
                let kyf = f32(ky);
                let cell_idx = base + kx * cells + ky;

                let cu0 = cellsf*x0 - kxf; let cv0 = cellsf*y0 - kyf;
                let cu1 = cellsf*x1 - kxf; let cv1 = cellsf*y1 - kyf;
                let cu2 = cellsf*x2 - kxf; let cv2 = cellsf*y2 - kyf;
                let cu3 = cellsf*x3 - kxf; let cv3 = cellsf*y3 - kyf;

                let tent_result = cubic_tent(
                    cu0, cv0, cu1, cv1, cu2, cv2, cu3, cv3);
                atomic_add_f32(cell_idx * 3u + 0u, tent_result.x);
                atomic_add_f32(cell_idx * 3u + 1u, tent_result.y);
                atomic_add_f32(cell_idx * 3u + 2u, tent_result.z);
            }
        }
    }
}

/// Circular arc entry point. All three coefficients via tent integral.
///
/// Segments: 3 × vec4 per arc ([0] = cx,cy,r,_; [1] = θ0,θ1,bb_min; [2] = bb_max).
@compute @workgroup_size(64)
fn compute_arcs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.num_segments { return; }

    let s0 = segments[idx * 3u];
    let s1 = segments[idx * 3u + 1u];
    let s2 = segments[idx * 3u + 2u];

    let arc_cx = s0.x; let arc_cy = s0.y; let arc_r = s0.z;
    let t0 = s1.x; let t1 = s1.y;
    let bbox_min = vec2(s1.z, s1.w);
    let bbox_max = vec2(s2.x, s2.y);

    for (var j: u32 = 0u; j <= params.max_j; j++) {
        let cells = 1u << j;
        let cellsf = f32(cells);
        let base = level_offset(j);

        let kx_lo = u32(max(floor(bbox_min.x * cellsf), 0.0));
        let kx_hi = min(u32(ceil(bbox_max.x * cellsf)), cells);
        let ky_lo = u32(max(floor(bbox_min.y * cellsf), 0.0));
        let ky_hi = min(u32(ceil(bbox_max.y * cellsf)), cells);

        for (var kx: u32 = kx_lo; kx < kx_hi; kx++) {
            let kxf = f32(kx);
            for (var ky: u32 = ky_lo; ky < ky_hi; ky++) {
                let kyf = f32(ky);
                let cell_idx = base + kx * cells + ky;

                let tent_result = arc_tent(
                    cellsf * arc_cx - kxf, cellsf * arc_cy - kyf,
                    cellsf * arc_r, t0, t1);
                atomic_add_f32(cell_idx * 3u + 0u, tent_result.x);
                atomic_add_f32(cell_idx * 3u + 1u, tent_result.y);
                atomic_add_f32(cell_idx * 3u + 2u, tent_result.z);
            }
        }
    }
}

/// Quadratic Bezier entry point. c11 via equation-8, c01/c10 via tent.
///
/// Segments: 2 × vec4 per quad ([0] = x0,y0,x1,y1; [1] = x2,y2,_,_).
@compute @workgroup_size(64)
fn compute_quads(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.num_segments { return; }

    let s0 = segments[idx * 2u];
    let s1 = segments[idx * 2u + 1u];
    let x0 = s0.x; let y0 = s0.y; let x1 = s0.z; let y1 = s0.w;
    let x2 = s1.x; let y2 = s1.y;

    let bbox_min = vec2(min(min(x0,x1),x2), min(min(y0,y1),y2));
    let bbox_max = vec2(max(max(x0,x1),x2), max(max(y0,y1),y2));

    for (var j: u32 = 0u; j <= params.max_j; j++) {
        let cells = 1u << j;
        let cellsf = f32(cells);
        let scale = f32(2u << j);
        let base = level_offset(j);

        let kx_lo = u32(max(floor(bbox_min.x * cellsf), 0.0));
        let kx_hi = min(u32(ceil(bbox_max.x * cellsf)), cells);
        let ky_lo = u32(max(floor(bbox_min.y * cellsf), 0.0));
        let ky_hi = min(u32(ceil(bbox_max.y * cellsf)), cells);

        for (var kx: u32 = kx_lo; kx < kx_hi; kx++) {
            let kxf = f32(kx);
            for (var ky: u32 = ky_lo; ky < ky_hi; ky++) {
                let kyf = f32(ky);
                let cell_idx = base + kx * cells + ky;

                // c11: equation-8 four-quadrant decomposition (in quadrant-scale coordinates)
                var kls: array<KL, 4>;
                for (var qi: u32 = 0u; qi < 4u; qi++) {
                    let qx = f32(qi >> 1u);
                    let qy = f32(qi & 1u);
                    kls[qi] = quad_get_kl(
                        scale * x0 - kxf * 2.0 - qx, scale * y0 - kyf * 2.0 - qy,
                        scale * x1 - kxf * 2.0 - qx, scale * y1 - kyf * 2.0 - qy,
                        scale * x2 - kxf * 2.0 - qx, scale * y2 - kyf * 2.0 - qy,
                        qx > 0.5, qy > 0.5);
                }
                accumulate_c11(cell_idx, kls[0], kls[1], kls[2], kls[3]);

                // c01/c10: tent integral (in cell-scale coordinates)
                let tent_result = quad_tent(
                    cellsf * x0 - kxf, cellsf * y0 - kyf,
                    cellsf * x1 - kxf, cellsf * y1 - kyf,
                    cellsf * x2 - kxf, cellsf * y2 - kyf);
                atomic_add_f32(cell_idx * 3u + 0u, tent_result.x);
                atomic_add_f32(cell_idx * 3u + 1u, tent_result.y);
            }
        }
    }
}

/// Superellipse entry point. c11 via equation-8, c01/c10 via tent.
///
/// Segments: 3 × vec4 ([0] = cx,cy,a,b; [1] = n,quads,bb_min; [2] = bb_max).
@compute @workgroup_size(64)
fn compute_superellipses(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.num_segments { return; }

    let s0 = segments[idx * 3u];
    let s1 = segments[idx * 3u + 1u];
    let s2 = segments[idx * 3u + 2u];

    let se_cx = s0.x; let se_cy = s0.y; let a = s0.z; let b_se = s0.w;
    let n = s1.x; let quadrants = u32(s1.y);
    let bbox_min = vec2(s1.z, s1.w);
    let bbox_max = vec2(s2.x, s2.y);

    for (var j: u32 = 0u; j <= params.max_j; j++) {
        let cells = 1u << j;
        let cellsf = f32(cells);
        let scale = f32(2u << j);
        let base = level_offset(j);

        let kx_lo = u32(max(floor(bbox_min.x * cellsf), 0.0));
        let kx_hi = min(u32(ceil(bbox_max.x * cellsf)), cells);
        let ky_lo = u32(max(floor(bbox_min.y * cellsf), 0.0));
        let ky_hi = min(u32(ceil(bbox_max.y * cellsf)), cells);

        for (var kx: u32 = kx_lo; kx < kx_hi; kx++) {
            let kxf = f32(kx);
            for (var ky: u32 = ky_lo; ky < ky_hi; ky++) {
                let kyf = f32(ky);
                let cell_idx = base + kx * cells + ky;

                // c11: equation-8 four-quadrant decomposition (in quadrant-scale coordinates)
                var kls: array<KL, 4>;
                for (var qi: u32 = 0u; qi < 4u; qi++) {
                    let qx = f32(qi >> 1u);
                    let qy = f32(qi & 1u);
                    kls[qi] = se_get_kl(
                        scale * se_cx - kxf * 2.0 - qx,
                        scale * se_cy - kyf * 2.0 - qy,
                        scale * a, scale * b_se, n, quadrants,
                        qx > 0.5, qy > 0.5);
                }
                accumulate_c11(cell_idx, kls[0], kls[1], kls[2], kls[3]);

                // c01/c10: tent integral (in cell-scale coordinates)
                let tent_result = se_tent(
                    cellsf * se_cx - kxf, cellsf * se_cy - kyf,
                    cellsf * a, cellsf * b_se, n, quadrants);
                atomic_add_f32(cell_idx * 3u + 0u, tent_result.x);
                atomic_add_f32(cell_idx * 3u + 1u, tent_result.y);
            }
        }
    }
}

// ============================================================================
// Sparse coefficient mode
// ============================================================================
//
// In sparse mode, only active cells (those overlapping at least one segment's
// bounding box) have entries in the coefficient buffer. The mapping from
// (level, cell_key) to compact buffer index is performed via binary search
// in a sorted per-level cell index array.
//
// Bindings 3 and 4 are only referenced by the sparse entry points below;
// the dense entry points above do not touch them.

struct LevelInfo {
    start: u32,
    count: u32,
}

@group(0) @binding(3) var<storage, read> level_infos: array<LevelInfo>;
@group(0) @binding(4) var<storage, read> cell_indices: array<u32>;

/// Binary search for cell_key in the sorted cell_indices slice for level j.
/// Returns the global compact index (usable to index into coeffs[]), or -1
/// if not found (should not happen if active cells are computed correctly).
fn find_compact_index(j: u32, cell_key: u32) -> i32 {
    let info = level_infos[j];
    var lo: u32 = info.start;
    var hi: u32 = info.start + info.count;
    while lo < hi {
        let mid = (lo + hi) / 2u;
        let val = cell_indices[mid];
        if val < cell_key {
            lo = mid + 1u;
        } else if val > cell_key {
            hi = mid;
        } else {
            return i32(mid);
        }
    }
    return -1;
}

/// Accumulate c11 from 4-quadrant KL values (sparse variant).
fn accumulate_c11_sparse(j: u32, cell_key: u32, q00: KL, q01: KL, q10: KL, q11: KL) {
    let compact = find_compact_index(j, cell_key);
    if compact >= 0 {
        let c11 = q00.lx - q01.lx + q10.kx - q10.lx - q11.kx + q11.lx;
        atomic_add_f32(u32(compact) * 3u + 2u, c11);
    }
}

// ---- Sparse entry points ----

@compute @workgroup_size(64)
fn compute_lines_sparse(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.num_segments { return; }

    let s = segments[idx];
    let x0 = s.x; let y0 = s.y; let x1 = s.z; let y1 = s.w;
    let bbox_min = vec2(min(x0, x1), min(y0, y1));
    let bbox_max = vec2(max(x0, x1), max(y0, y1));

    for (var j: u32 = 0u; j <= params.max_j; j++) {
        let cells = 1u << j;
        let cellsf = f32(cells);

        let kx_lo = u32(max(floor(bbox_min.x * cellsf), 0.0));
        let kx_hi = min(u32(ceil(bbox_max.x * cellsf)), cells);
        let ky_lo = u32(max(floor(bbox_min.y * cellsf), 0.0));
        let ky_hi = min(u32(ceil(bbox_max.y * cellsf)), cells);

        for (var kx: u32 = kx_lo; kx < kx_hi; kx++) {
            let kxf = f32(kx);
            for (var ky: u32 = ky_lo; ky < ky_hi; ky++) {
                let kyf = f32(ky);
                let cell_key = kx * cells + ky;
                let compact = find_compact_index(j, cell_key);
                if compact >= 0 {
                    let tent_result = line_tent(
                        cellsf * x0 - kxf, cellsf * y0 - kyf,
                        cellsf * x1 - kxf, cellsf * y1 - kyf);
                    atomic_add_f32(u32(compact) * 3u + 0u, tent_result.x);
                    atomic_add_f32(u32(compact) * 3u + 1u, tent_result.y);
                    atomic_add_f32(u32(compact) * 3u + 2u, tent_result.z);
                }
            }
        }
    }
}

@compute @workgroup_size(256)
fn compute_cubics_sparse(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.num_segments { return; }

    let s0 = segments[idx * 2u];
    let s1 = segments[idx * 2u + 1u];
    let x0 = s0.x; let y0 = s0.y; let x1 = s0.z; let y1 = s0.w;
    let x2 = s1.x; let y2 = s1.y; let x3 = s1.z; let y3 = s1.w;

    let bbox_min = vec2(min(min(x0,x1),min(x2,x3)), min(min(y0,y1),min(y2,y3)));
    let bbox_max = vec2(max(max(x0,x1),max(x2,x3)), max(max(y0,y1),max(y2,y3)));

    for (var j: u32 = 0u; j <= params.max_j; j++) {
        let cells = 1u << j;
        let cellsf = f32(cells);

        let kx_lo = u32(max(floor(bbox_min.x * cellsf), 0.0));
        let kx_hi = min(u32(ceil(bbox_max.x * cellsf)), cells);
        let ky_lo = u32(max(floor(bbox_min.y * cellsf), 0.0));
        let ky_hi = min(u32(ceil(bbox_max.y * cellsf)), cells);

        for (var kx: u32 = kx_lo; kx < kx_hi; kx++) {
            let kxf = f32(kx);
            for (var ky: u32 = ky_lo; ky < ky_hi; ky++) {
                let kyf = f32(ky);
                let cell_key = kx * cells + ky;
                let compact = find_compact_index(j, cell_key);
                if compact >= 0 {
                    let tent_result = cubic_tent(
                        cellsf*x0 - kxf, cellsf*y0 - kyf,
                        cellsf*x1 - kxf, cellsf*y1 - kyf,
                        cellsf*x2 - kxf, cellsf*y2 - kyf,
                        cellsf*x3 - kxf, cellsf*y3 - kyf);
                    atomic_add_f32(u32(compact) * 3u + 0u, tent_result.x);
                    atomic_add_f32(u32(compact) * 3u + 1u, tent_result.y);
                    atomic_add_f32(u32(compact) * 3u + 2u, tent_result.z);
                }
            }
        }
    }
}

@compute @workgroup_size(64)
fn compute_arcs_sparse(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.num_segments { return; }

    let s0 = segments[idx * 3u];
    let s1 = segments[idx * 3u + 1u];
    let s2 = segments[idx * 3u + 2u];

    let arc_cx = s0.x; let arc_cy = s0.y; let arc_r = s0.z;
    let t0 = s1.x; let t1 = s1.y;
    let bbox_min = vec2(s1.z, s1.w);
    let bbox_max = vec2(s2.x, s2.y);

    for (var j: u32 = 0u; j <= params.max_j; j++) {
        let cells = 1u << j;
        let cellsf = f32(cells);

        let kx_lo = u32(max(floor(bbox_min.x * cellsf), 0.0));
        let kx_hi = min(u32(ceil(bbox_max.x * cellsf)), cells);
        let ky_lo = u32(max(floor(bbox_min.y * cellsf), 0.0));
        let ky_hi = min(u32(ceil(bbox_max.y * cellsf)), cells);

        for (var kx: u32 = kx_lo; kx < kx_hi; kx++) {
            let kxf = f32(kx);
            for (var ky: u32 = ky_lo; ky < ky_hi; ky++) {
                let kyf = f32(ky);
                let cell_key = kx * cells + ky;
                let compact = find_compact_index(j, cell_key);
                if compact >= 0 {
                    let tent_result = arc_tent(
                        cellsf * arc_cx - kxf, cellsf * arc_cy - kyf,
                        cellsf * arc_r, t0, t1);
                    atomic_add_f32(u32(compact) * 3u + 0u, tent_result.x);
                    atomic_add_f32(u32(compact) * 3u + 1u, tent_result.y);
                    atomic_add_f32(u32(compact) * 3u + 2u, tent_result.z);
                }
            }
        }
    }
}

@compute @workgroup_size(64)
fn compute_quads_sparse(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.num_segments { return; }

    let s0 = segments[idx * 2u];
    let s1 = segments[idx * 2u + 1u];
    let x0 = s0.x; let y0 = s0.y; let x1 = s0.z; let y1 = s0.w;
    let x2 = s1.x; let y2 = s1.y;

    let bbox_min = vec2(min(min(x0,x1),x2), min(min(y0,y1),y2));
    let bbox_max = vec2(max(max(x0,x1),x2), max(max(y0,y1),y2));

    for (var j: u32 = 0u; j <= params.max_j; j++) {
        let cells = 1u << j;
        let cellsf = f32(cells);
        let scale = f32(2u << j);

        let kx_lo = u32(max(floor(bbox_min.x * cellsf), 0.0));
        let kx_hi = min(u32(ceil(bbox_max.x * cellsf)), cells);
        let ky_lo = u32(max(floor(bbox_min.y * cellsf), 0.0));
        let ky_hi = min(u32(ceil(bbox_max.y * cellsf)), cells);

        for (var kx: u32 = kx_lo; kx < kx_hi; kx++) {
            let kxf = f32(kx);
            for (var ky: u32 = ky_lo; ky < ky_hi; ky++) {
                let kyf = f32(ky);
                let cell_key = kx * cells + ky;
                let compact = find_compact_index(j, cell_key);
                if compact >= 0 {
                    // c11: equation-8 four-quadrant decomposition
                    var kls: array<KL, 4>;
                    for (var qi: u32 = 0u; qi < 4u; qi++) {
                        let qx = f32(qi >> 1u);
                        let qy = f32(qi & 1u);
                        kls[qi] = quad_get_kl(
                            scale * x0 - kxf * 2.0 - qx, scale * y0 - kyf * 2.0 - qy,
                            scale * x1 - kxf * 2.0 - qx, scale * y1 - kyf * 2.0 - qy,
                            scale * x2 - kxf * 2.0 - qx, scale * y2 - kyf * 2.0 - qy,
                            qx > 0.5, qy > 0.5);
                    }
                    let c11 = kls[0].lx - kls[1].lx + kls[2].kx - kls[2].lx - kls[3].kx + kls[3].lx;
                    atomic_add_f32(u32(compact) * 3u + 2u, c11);

                    // c01/c10: tent integral
                    let tent_result = quad_tent(
                        cellsf * x0 - kxf, cellsf * y0 - kyf,
                        cellsf * x1 - kxf, cellsf * y1 - kyf,
                        cellsf * x2 - kxf, cellsf * y2 - kyf);
                    atomic_add_f32(u32(compact) * 3u + 0u, tent_result.x);
                    atomic_add_f32(u32(compact) * 3u + 1u, tent_result.y);
                }
            }
        }
    }
}

@compute @workgroup_size(64)
fn compute_superellipses_sparse(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.num_segments { return; }

    let s0 = segments[idx * 3u];
    let s1 = segments[idx * 3u + 1u];
    let s2 = segments[idx * 3u + 2u];

    let se_cx = s0.x; let se_cy = s0.y; let a = s0.z; let b_se = s0.w;
    let n = s1.x; let quadrants = u32(s1.y);
    let bbox_min = vec2(s1.z, s1.w);
    let bbox_max = vec2(s2.x, s2.y);

    for (var j: u32 = 0u; j <= params.max_j; j++) {
        let cells = 1u << j;
        let cellsf = f32(cells);
        let scale = f32(2u << j);

        let kx_lo = u32(max(floor(bbox_min.x * cellsf), 0.0));
        let kx_hi = min(u32(ceil(bbox_max.x * cellsf)), cells);
        let ky_lo = u32(max(floor(bbox_min.y * cellsf), 0.0));
        let ky_hi = min(u32(ceil(bbox_max.y * cellsf)), cells);

        for (var kx: u32 = kx_lo; kx < kx_hi; kx++) {
            let kxf = f32(kx);
            for (var ky: u32 = ky_lo; ky < ky_hi; ky++) {
                let kyf = f32(ky);
                let cell_key = kx * cells + ky;
                let compact = find_compact_index(j, cell_key);
                if compact >= 0 {
                    // c11: equation-8 four-quadrant decomposition
                    var kls: array<KL, 4>;
                    for (var qi: u32 = 0u; qi < 4u; qi++) {
                        let qx = f32(qi >> 1u);
                        let qy = f32(qi & 1u);
                        kls[qi] = se_get_kl(
                            scale * se_cx - kxf * 2.0 - qx,
                            scale * se_cy - kyf * 2.0 - qy,
                            scale * a, scale * b_se, n, quadrants,
                            qx > 0.5, qy > 0.5);
                    }
                    let c11 = kls[0].lx - kls[1].lx + kls[2].kx - kls[2].lx - kls[3].kx + kls[3].lx;
                    atomic_add_f32(u32(compact) * 3u + 2u, c11);

                    // c01/c10: tent integral
                    let tent_result = se_tent(
                        cellsf * se_cx - kxf, cellsf * se_cy - kyf,
                        cellsf * a, cellsf * b_se, n, quadrants);
                    atomic_add_f32(u32(compact) * 3u + 0u, tent_result.x);
                    atomic_add_f32(u32(compact) * 3u + 1u, tent_result.y);
                }
            }
        }
    }
}
