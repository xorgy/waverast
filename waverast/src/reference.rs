//! Reference analytic rasterizer for ground truth validation.
//!
//! Computes exact pixel coverage by scan-line integration. For each row,
//! boundary segment crossings are collected and a right-to-left prefix sum
//! propagates winding contributions. Each pixel gets exact analytic area
//! coverage — no point sampling or approximation (curves are integrated
//! using their closed-form parametric integrals).

use crate::contour::{CircularArc, CubicBez, Line, Point, QuadBez, Segment, Shape, Superellipse};
use crate::solver::{solve_cubic, solve_quadratic};

use std::f64::consts::PI;

/// Fill rule. The caller applies this to the raw signed coverage values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FillRule {
    NonZero,
    EvenOdd,
}

/// Compute exact analytic pixel coverage for a shape.
///
/// Returns a row-major `Vec<f32>` of size `h × w` with signed coverage.
/// The caller applies fill rules and sRGB conversion identically to the
/// wavelet rasterizer path.
pub fn reference_rasterize(shape: &Shape, w: usize, h: usize, _fill_rule: FillRule) -> Vec<f32> {
    let mut pixels = vec![0.0f64; w * h];

    for row in 0..h {
        let row_f = row as f64;
        let mut local = vec![0.0f64; w];
        let mut wind = vec![0.0f64; w + 1];

        for seg in &shape.segments {
            let (y_min, y_max) = seg_y_range(seg);
            if y_max < row_f || y_min > row_f + 1.0 {
                continue;
            }
            contribute_segment_to_row(seg, row_f, w, &mut local, &mut wind);
        }

        let mut running = 0.0;
        for col in (0..w).rev() {
            running += wind[col + 1];
            pixels[row * w + col] = local[col] + running;
        }
    }

    pixels.iter().map(|&v| v as f32).collect()
}

fn seg_y_range(seg: &Segment) -> (f64, f64) {
    let bb = seg.bbox();
    (bb.y0, bb.y1)
}

/// Compute a segment's contribution to a single pixel row.
/// "Local" contributions go to `local[col]`. "Right" contributions go
/// to `wind[col]` for prefix-sum propagation.
fn contribute_segment_to_row(
    seg: &Segment,
    row: f64,
    w: usize,
    local: &mut [f64],
    wind: &mut [f64],
) {
    match seg {
        Segment::Line(Line { p0, p1 }) => line_row(p0, p1, row, w, local, wind),
        Segment::QuadBez(QuadBez { p0, p1, p2 }) => quad_row(p0, p1, p2, row, w, local, wind),
        Segment::CubicBez(CubicBez { p0, p1, p2, p3 }) => {
            cubic_row(p0, p1, p2, p3, row, w, local, wind)
        }
        Segment::CircularArc(CircularArc {
            center,
            radius,
            theta0,
            theta1,
        }) => arc_row(center, *radius, *theta0, *theta1, row, w, local, wind),
        Segment::Superellipse(Superellipse {
            center,
            a,
            b,
            n,
            quadrants,
        }) => se_row(center, *a, *b, *n, *quadrants, row, w, local, wind),
    }
}

/// Bisect to find t where coordinate `axis` (0=x, 1=y) equals `boundary`.
fn bisect_coord(
    t_lo: f64,
    t_hi: f64,
    val_lo: f64,
    boundary: f64,
    eval: &dyn Fn(f64) -> (f64, f64),
    axis: usize,
) -> f64 {
    let mut lo = t_lo;
    let mut hi = t_hi;
    let sign_lo = val_lo - boundary;
    for _ in 0..60 {
        let mid = (lo + hi) * 0.5;
        if mid == lo || mid == hi {
            break;
        }
        let p = eval(mid);
        let val = if axis == 0 { p.0 } else { p.1 };
        if (val - boundary) * sign_lo > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) * 0.5
}

/// Process sub-intervals for a segment at a given row.
fn process_row_intervals(
    ts: &mut Vec<f64>,
    row: f64,
    w: usize,
    eval: &dyn Fn(f64) -> (f64, f64),
    integrate_xy: &dyn Fn(f64, f64) -> f64,
    local: &mut [f64],
    wind: &mut [f64],
) {
    ts.retain(|&t| (0.0..=1.0).contains(&t));

    let row_top = row + 1.0;

    ts.sort_by(|a, b| a.total_cmp(b));

    // Robustly add missed crossings via bisection. For each adjacent pair of
    // t-values, check if y straddles a row boundary or x straddles a column
    // boundary that wasn't captured by the analytic solver (e.g., near tangent
    // points where roots are numerically lost).
    let mut extra = Vec::new();
    for i in 0..ts.len().saturating_sub(1) {
        let (xa, ya) = eval(ts[i]);
        let (xb, yb) = eval(ts[i + 1]);
        // y-boundary crossings
        for boundary in [row, row_top] {
            if (ya - boundary) * (yb - boundary) < 0.0 {
                extra.push(bisect_coord(ts[i], ts[i + 1], ya, boundary, eval, 1));
            }
        }
        // x-boundary crossings at integer columns
        let x_lo = xa.min(xb).floor() as isize;
        let x_hi = xa.max(xb).ceil() as isize;
        for c in x_lo..=x_hi {
            let boundary = c as f64;
            if (xa - boundary) * (xb - boundary) < 0.0 {
                extra.push(bisect_coord(ts[i], ts[i + 1], xa, boundary, eval, 0));
            }
        }
    }
    ts.extend_from_slice(&extra);

    ts.sort_by(|a, b| a.total_cmp(b));
    ts.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

    for i in 0..ts.len().saturating_sub(1) {
        let ta = ts[i];
        let tb = ts[i + 1];
        if (tb - ta).abs() < 1e-14 {
            continue;
        }

        let mid_t = (ta + tb) * 0.5;
        let (mx, my) = eval(mid_t);

        // Skip sub-intervals outside the row's y-range
        if my < row || my > row_top {
            continue;
        }

        let (_, ya) = eval(ta);
        let (_, yb) = eval(tb);
        let dy = yb - ya;

        let col = mx.floor() as isize;

        if col < 0 {
            continue;
        }

        let col_u = col as usize;

        if col_u >= w {
            // Segment to the right of all pixels: its winding affects every
            // pixel in the row. wind[w] propagates via the right-to-left
            // prefix sum to all columns.
            wind[w] += dy;
            continue;
        }

        let col_f = col_u as f64;

        // Area contribution: ∫(x - col)·y'dt = ∫x·y'dt - col·dy
        let area = integrate_xy(ta, tb);
        local[col_u] += area - col_f * dy;

        // Winding: all pixels to the LEFT (col < col_u) get +dy
        wind[col_u] += dy;
    }
}

// ---- Lines ----

fn line_row(p0: &Point, p1: &Point, row: f64, w: usize, local: &mut [f64], wind: &mut [f64]) {
    let dx = p1.x - p0.x;
    let dy = p1.y - p0.y;

    let mut ts = vec![0.0, 1.0];

    if dy.abs() > 1e-14 {
        ts.push((row - p0.y) / dy);
        ts.push((row + 1.0 - p0.y) / dy);
    }
    // x-crossings at every integer column within range
    let x_min = p0.x.min(p1.x);
    let x_max = p0.x.max(p1.x);
    if dx.abs() > 1e-14 {
        let c_lo = x_min.floor() as isize;
        let c_hi = x_max.ceil() as isize;
        for c in c_lo..=c_hi {
            let t = (c as f64 - p0.x) / dx;
            ts.push(t);
        }
    }

    let eval = |t: f64| (p0.x + t * dx, p0.y + t * dy);
    let integrate_xy =
        |ta: f64, tb: f64| -> f64 { dy * (p0.x * (tb - ta) + dx * (tb * tb - ta * ta) * 0.5) };

    process_row_intervals(&mut ts, row, w, &eval, &integrate_xy, local, wind);
}

// ---- Quadratic Bezier ----

fn quad_row(
    p0: &Point,
    p1: &Point,
    p2: &Point,
    row: f64,
    w: usize,
    local: &mut [f64],
    wind: &mut [f64],
) {
    let ax = p0.x - 2.0 * p1.x + p2.x;
    let bx = 2.0 * (p1.x - p0.x);
    let cx = p0.x;
    let ay = p0.y - 2.0 * p1.y + p2.y;
    let by = 2.0 * (p1.y - p0.y);
    let cy = p0.y;

    let mut ts = vec![0.0, 1.0];

    for val in [row, row + 1.0] {
        let (n, roots) = solve_quadratic(ay, by, cy - val);
        for &r in &roots[..n] {
            ts.push(r);
        }
    }
    let x_min = p0.x.min(p1.x).min(p2.x).floor() as isize;
    let x_max = p0.x.max(p1.x).max(p2.x).ceil() as isize;
    for c in x_min..=x_max {
        let (n, roots) = solve_quadratic(ax, bx, cx - c as f64);
        for &r in &roots[..n] {
            ts.push(r);
        }
    }

    let eval = |t: f64| ((ax * t + bx) * t + cx, (ay * t + by) * t + cy);
    let integrate_xy = |ta: f64, tb: f64| -> f64 {
        let c3 = 2.0 * ax * ay;
        let c2 = ax * by + 2.0 * bx * ay;
        let c1 = bx * by + 2.0 * cx * ay;
        let c0 = cx * by;
        poly_integral(&[c0, c1, c2, c3], ta, tb)
    };

    process_row_intervals(&mut ts, row, w, &eval, &integrate_xy, local, wind);
}

// ---- Cubic Bezier ----

fn cubic_row(
    p0: &Point,
    p1: &Point,
    p2: &Point,
    p3: &Point,
    row: f64,
    w: usize,
    local: &mut [f64],
    wind: &mut [f64],
) {
    let ax = -p0.x + 3.0 * p1.x - 3.0 * p2.x + p3.x;
    let bx = 3.0 * p0.x - 6.0 * p1.x + 3.0 * p2.x;
    let cx = 3.0 * (p1.x - p0.x);
    let dx = p0.x;
    let ay = -p0.y + 3.0 * p1.y - 3.0 * p2.y + p3.y;
    let by = 3.0 * p0.y - 6.0 * p1.y + 3.0 * p2.y;
    let cy = 3.0 * (p1.y - p0.y);
    let dy = p0.y;

    let mut ts = vec![0.0, 1.0];

    for val in [row, row + 1.0] {
        let (n, roots) = solve_cubic(ay, by, cy, dy - val);
        for &r in &roots[..n] {
            ts.push(r);
        }
    }
    let x_min = p0.x.min(p1.x).min(p2.x).min(p3.x).floor() as isize;
    let x_max = p0.x.max(p1.x).max(p2.x).max(p3.x).ceil() as isize;
    for c in x_min..=x_max {
        let (n, roots) = solve_cubic(ax, bx, cx, dx - c as f64);
        for &r in &roots[..n] {
            ts.push(r);
        }
    }

    let eval = |t: f64| {
        (
            ((ax * t + bx) * t + cx) * t + dx,
            ((ay * t + by) * t + cy) * t + dy,
        )
    };
    let integrate_xy = |ta: f64, tb: f64| -> f64 {
        // x·y' where x is degree 3, y' is degree 2 → degree 5
        let c5 = 3.0 * ax * ay;
        let c4 = 2.0 * ax * by + 3.0 * bx * ay;
        let c3 = ax * cy + 2.0 * bx * by + 3.0 * cx * ay;
        let c2 = bx * cy + 2.0 * cx * by + 3.0 * dx * ay;
        let c1 = cx * cy + 2.0 * dx * by;
        let c0 = dx * cy;
        poly_integral(&[c0, c1, c2, c3, c4, c5], ta, tb)
    };

    process_row_intervals(&mut ts, row, w, &eval, &integrate_xy, local, wind);
}

// ---- Circular Arc ----

fn arc_row(
    center: &Point,
    r: f64,
    theta0: f64,
    theta1: f64,
    row: f64,
    w: usize,
    local: &mut [f64],
    wind: &mut [f64],
) {
    let dtheta = theta1 - theta0;
    if dtheta.abs() < 1e-14 {
        return;
    }

    let mut ts = vec![0.0, 1.0];

    // y-crossings
    for val in [row, row + 1.0] {
        let d = (val - center.y) / r;
        if d.abs() <= 1.0 {
            let asin_d = d.asin();
            for base in [asin_d, PI - asin_d] {
                push_arc_crossings(theta0, dtheta, base, &mut ts);
            }
        }
    }

    // x-crossings at integer columns
    let x_min = (center.x - r).floor() as isize;
    let x_max = (center.x + r).ceil() as isize;
    for c in x_min..=x_max {
        let d = (c as f64 - center.x) / r;
        if d.abs() <= 1.0 {
            let acos_d = d.acos();
            for base in [acos_d, -acos_d] {
                push_arc_crossings(theta0, dtheta, base, &mut ts);
            }
        }
    }

    // Also split at extrema
    for offset in [0.0, PI * 0.5, PI, PI * 1.5] {
        push_arc_crossings(theta0, dtheta, offset, &mut ts);
    }

    let eval = |t: f64| {
        let theta = theta0 + t * dtheta;
        (center.x + r * theta.cos(), center.y + r * theta.sin())
    };
    let integrate_xy = |ta: f64, tb: f64| -> f64 {
        let theta_a = theta0 + ta * dtheta;
        let theta_b = theta0 + tb * dtheta;
        r * center.x * (theta_b.sin() - theta_a.sin())
            + r * r
                * ((theta_b - theta_a) * 0.5
                    + ((2.0 * theta_b).sin() - (2.0 * theta_a).sin()) * 0.25)
    };

    process_row_intervals(&mut ts, row, w, &eval, &integrate_xy, local, wind);
}

fn push_arc_crossings(theta0: f64, dtheta: f64, base: f64, ts: &mut Vec<f64>) {
    let (lo, hi) = if dtheta > 0.0 {
        (theta0, theta0 + dtheta)
    } else {
        (theta0 + dtheta, theta0)
    };
    let k_min = ((lo - base) / std::f64::consts::TAU).ceil() as i64;
    let k_max = ((hi - base) / std::f64::consts::TAU).floor() as i64;
    for k in k_min..=k_max {
        let theta = base + k as f64 * std::f64::consts::TAU;
        let t = (theta - theta0) / dtheta;
        ts.push(t);
    }
}

// ---- Superellipse ----

const SE_QUADRANTS: [(f64, f64); 4] = [(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)];

fn se_profile(u: f64, n: f64) -> f64 {
    (1.0 - u.powf(n)).max(0.0).powf(1.0 / n)
}

fn se_row(
    center: &Point,
    a: f64,
    b: f64,
    n: f64,
    quadrants: u8,
    row: f64,
    w: usize,
    local: &mut [f64],
    wind: &mut [f64],
) {
    for (qi, &(sx, sy)) in SE_QUADRANTS.iter().enumerate() {
        if quadrants & (1 << qi) == 0 {
            continue;
        }
        let dir = -sx * sy;

        let eval = |t: f64| -> (f64, f64) {
            let u = if dir < 0.0 { 1.0 - t } else { t };
            (center.x + sx * a * u, center.y + sy * b * se_profile(u, n))
        };

        let mut ts = vec![0.0, 1.0];

        // y-crossings
        if (sy * b).abs() > 1e-14 {
            for val in [row, row + 1.0] {
                let d = (val - center.y) / (sy * b);
                if (0.0..=1.0).contains(&d) {
                    let u = se_profile(d, n);
                    if (0.0..=1.0).contains(&u) {
                        ts.push(if dir < 0.0 { 1.0 - u } else { u });
                    }
                }
            }
        }

        // x-crossings at integer columns
        if (sx * a).abs() > 1e-14 {
            let x_lo = (center.x - a.abs()).floor() as isize;
            let x_hi = (center.x + a.abs()).ceil() as isize;
            for c in x_lo..=x_hi {
                let u = (c as f64 - center.x) / (sx * a);
                if (0.0..=1.0).contains(&u) {
                    ts.push(if dir < 0.0 { 1.0 - u } else { u });
                }
            }
        }

        let integrate_xy = |ta: f64, tb: f64| -> f64 {
            // ∫ x·y' dt via integration by parts: [x·y] - ∫ x'·y dt
            // x'(t) = sx*a*dir (constant), so ∫ x'·y dt = sx*a*dir * ∫ y dt
            // ∫ y dt is smooth (no singularity) → perfect for GL-16.
            let (xa, ya) = eval(ta);
            let (xb, yb) = eval(tb);
            let boundary = xb * yb - xa * ya;

            const NODES: [f64; 8] = [
                0.095_012_509_837_637_44,
                0.281_603_550_779_258_9,
                0.458_016_777_657_227_39,
                0.617_876_244_402_643_8,
                0.755_404_408_355_003,
                0.865_631_202_387_831_8,
                0.944_575_023_073_232_6,
                0.989_400_934_991_649_9,
            ];
            const WEIGHTS: [f64; 8] = [
                0.189_450_610_455_068_5,
                0.182_603_415_044_924_59,
                0.169_156_519_395_002_54,
                0.149_595_988_816_576_73,
                0.124_628_971_255_533_87,
                0.095_158_511_682_492_78,
                0.062_253_523_938_647_89,
                0.027_152_459_411_754_09,
            ];
            let half = (tb - ta) * 0.5;
            let mid_t = (ta + tb) * 0.5;
            let mut sum = 0.0;
            for i in 0..8 {
                let x_node = half * NODES[i];
                let (_, y_neg) = eval(mid_t - x_node);
                let (_, y_pos) = eval(mid_t + x_node);
                sum += WEIGHTS[i] * (y_neg + y_pos);
            }
            let int_y = sum * half;
            let xp = sx * a * dir;
            boundary - xp * int_y
        };

        process_row_intervals(&mut ts, row, w, &eval, &integrate_xy, local, wind);
    }
}

// ---- Polynomial integration ----

/// Integrate polynomial c[0] + c[1]·t + c[2]·t² + ... from ta to tb.
fn poly_integral(coeffs: &[f64], ta: f64, tb: f64) -> f64 {
    // Antiderivative: c[0]·t + c[1]/2·t² + c[2]/3·t³ + ...
    let anti = |t: f64| -> f64 {
        let mut result = 0.0;
        let mut t_power = t;
        for (i, &c) in coeffs.iter().enumerate() {
            result += c / (i as f64 + 1.0) * t_power;
            t_power *= t;
        }
        result
    };
    anti(tb) - anti(ta)
}

// ---- Comparison and analysis utilities ----

/// Compare wavelet and reference outputs, returning the top errors.
pub fn compare_outputs(
    wav_pixels: &[f32],
    ref_pixels: &[f32],
    w: usize,
    h: usize,
    top_n: usize,
) -> Vec<(usize, usize, f32, f32, f32)> {
    let mut errors: Vec<(usize, usize, f32, f32, f32)> = Vec::new();
    for row in 0..h {
        for col in 0..w {
            let i = row * w + col;
            let diff = (wav_pixels[i] - ref_pixels[i]).abs();
            if diff > 1e-4 {
                errors.push((col, row, wav_pixels[i], ref_pixels[i], diff));
            }
        }
    }
    errors.sort_by(|a, b| b.4.total_cmp(&a.4));
    errors.truncate(top_n);
    errors
}

/// Find which segments overlap a given pixel-space rectangle.
pub fn segments_in_cell(
    shape: &Shape,
    cell_x_lo: f64,
    cell_x_hi: f64,
    cell_y_lo: f64,
    cell_y_hi: f64,
) -> Vec<(usize, f64, f64, f64, f64)> {
    let mut result = Vec::new();
    for (i, seg) in shape.segments.iter().enumerate() {
        let bb = seg.bbox();
        if bb.x1 > cell_x_lo && bb.x0 < cell_x_hi && bb.y1 > cell_y_lo && bb.y0 < cell_y_hi {
            result.push((i, bb.x0, bb.y0, bb.x1, bb.y1));
        }
    }
    result
}

/// Compute the wavelet coefficient at a specific cell for a subset of segments.
pub fn coefficient_at_cell(
    shape: &Shape,
    j: u32,
    kx: usize,
    ky: usize,
    seg_indices: &[usize],
    wh: f64,
) -> (f64, f64, f64) {
    use crate::contour::{KL, arc_get_kl, cubic_get_kl, line_get_kl, quad_get_kl, se_get_kl};

    let _cells = 1usize << j;
    let scale = (1u64 << (j + 1)) as f64;
    let kxf = kx as f64;
    let kyf = ky as f64;

    let quadrants: [(f64, f64); 4] = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)];

    let mut c01 = 0.0f64;
    let mut c10 = 0.0f64;
    let mut c11 = 0.0f64;

    let mut shape_clone = shape.clone();
    shape_clone.normalize(wh);

    for &si in seg_indices {
        let seg = &shape_clone.segments[si];
        let mut kls = [KL::zero(); 4];

        for (qi, &(qx, qy)) in quadrants.iter().enumerate() {
            let tx = |x: f64| scale * x - kxf * 2.0 - qx;
            let ty = |y: f64| scale * y - kyf * 2.0 - qy;
            let skip_x0 = qx > 0.5;
            let skip_y0 = qy > 0.5;

            kls[qi] = match seg {
                Segment::Line(Line { p0, p1 }) => {
                    line_get_kl(tx(p0.x), ty(p0.y), tx(p1.x), ty(p1.y), skip_x0, skip_y0)
                }
                Segment::CubicBez(CubicBez { p0, p1, p2, p3 }) => cubic_get_kl(
                    tx(p0.x),
                    ty(p0.y),
                    tx(p1.x),
                    ty(p1.y),
                    tx(p2.x),
                    ty(p2.y),
                    tx(p3.x),
                    ty(p3.y),
                    skip_x0,
                    skip_y0,
                ),
                Segment::QuadBez(QuadBez { p0, p1, p2 }) => quad_get_kl(
                    tx(p0.x),
                    ty(p0.y),
                    tx(p1.x),
                    ty(p1.y),
                    tx(p2.x),
                    ty(p2.y),
                    skip_x0,
                    skip_y0,
                ),
                Segment::CircularArc(CircularArc {
                    center,
                    radius,
                    theta0,
                    theta1,
                }) => arc_get_kl(
                    scale * center.x - kxf * 2.0 - qx,
                    scale * center.y - kyf * 2.0 - qy,
                    scale * radius,
                    *theta0,
                    *theta1,
                    skip_x0,
                    skip_y0,
                ),
                Segment::Superellipse(Superellipse {
                    center,
                    a,
                    b,
                    n,
                    quadrants: q,
                }) => se_get_kl(
                    scale * center.x - kxf * 2.0 - qx,
                    scale * center.y - kyf * 2.0 - qy,
                    scale * a,
                    scale * b,
                    *n,
                    *q,
                    skip_x0,
                    skip_y0,
                ),
            };
        }

        let q00 = kls[0];
        let q01 = kls[1];
        let q10 = kls[2];
        let q11 = kls[3];
        c10 += q00.lx + q01.lx + q10.kx - q10.lx + q11.kx - q11.lx;
        c01 += q00.ly + q10.ly + q01.ky - q01.ly + q11.ky - q11.ly;
        c11 += q00.lx - q01.lx + q10.kx - q10.lx - q11.kx + q11.lx;
    }

    (c01, c10, c11)
}

/// Compute expected wavelet coefficient from reference pixel coverage.
pub fn expected_coefficient_from_reference(
    ref_pixels: &[f32],
    w: usize,
    _h: usize,
    j: u32,
    kx: usize,
    ky: usize,
    wh: f32,
    area: f32,
) -> (f32, f32, f32) {
    let cells = 1usize << j;
    let cell_size = wh / cells as f32;
    let x_lo = kx as f32 * cell_size;
    let y_lo = ky as f32 * cell_size;
    let x_mid = x_lo + cell_size * 0.5;
    let y_mid = y_lo + cell_size * 0.5;

    let mut c01 = 0.0f64;
    let mut c10 = 0.0f64;
    let mut c11 = 0.0f64;
    let mut n = 0;

    for row in (y_lo as usize)..(y_lo as usize + cell_size as usize).min(_h) {
        let py = row as f32 + 0.5;
        let sign_y: f64 = if py < y_mid { 1.0 } else { -1.0 };
        for col in (x_lo as usize)..(x_lo as usize + cell_size as usize).min(w) {
            let px = col as f32 + 0.5;
            let sign_x: f64 = if px < x_mid { 1.0 } else { -1.0 };
            let cov = (ref_pixels[row * w + col] - area) as f64;
            c01 += cov * sign_y;
            c10 += cov * sign_x;
            c11 += cov * sign_x * sign_y;
            n += 1;
        }
    }

    let scale = 1.0 / n as f64;
    (
        (c01 * scale) as f32,
        (c10 * scale) as f32,
        (c11 * scale) as f32,
    )
}

/// Compute wavelet coefficient c^(0,1) via the direct boundary integral
/// (M&S eq. 7 with Ψ̄) instead of the four-quadrant decomposition (eq. 8).
///
/// c^(0,1) = -∮ Ψ̄(cell_y) dx, where Ψ̄ is the tent function.
/// The integration is clipped to the cell's y-range and split at the tent kink.
pub fn direct_c01(
    shape: &Shape,
    j: u32,
    _kx: usize,
    ky: usize,
    seg_indices: &[usize],
    wh: f64,
) -> f64 {
    use crate::solver::solve_cubic;

    let cells = 1usize << j;
    let kyf = ky as f64;

    let mut shape_clone = shape.clone();
    shape_clone.normalize(wh);

    // Ψ̄(t) = t for t ∈ [0, 0.5), (1-t) for t ∈ [0.5, 1), 0 otherwise
    let psi_bar = |t: f64| -> f64 {
        if t <= 0.0 || t >= 1.0 {
            0.0
        } else if t < 0.5 {
            t
        } else {
            1.0 - t
        }
    };

    let mut c01 = 0.0f64;
    let cells_f = cells as f64;

    for &si in seg_indices {
        let seg = &shape_clone.segments[si];
        match seg {
            Segment::Line(Line { p0, p1 }) => {
                let dx = p1.x - p0.x;
                // cell_y(t) = cells * (p0.y + t*dy) - ky
                // Find t where cell_y = 0, 0.5, 1 (boundaries of Ψ̄ pieces)
                let cy0 = cells as f64 * p0.y - kyf;
                let cy1 = cells as f64 * p1.y - kyf;
                let cdy = cy1 - cy0;
                let mut ts = vec![0.0, 1.0];
                if cdy.abs() > 1e-14 {
                    for val in [0.0, 0.5, 1.0] {
                        let t = (val - cy0) / cdy;
                        if t > 0.0 && t < 1.0 {
                            ts.push(t);
                        }
                    }
                }
                ts.sort_by(|a, b| a.total_cmp(b));
                // Integrate piecewise
                for i in 0..ts.len() - 1 {
                    let ta = ts[i];
                    let tb = ts[i + 1];
                    let mid = (ta + tb) * 0.5;
                    let cy_mid = cy0 + mid * cdy;
                    if cy_mid <= 0.0 || cy_mid >= 1.0 {
                        continue;
                    }
                    // Within one piece of Ψ̄: either t or 1-t
                    // ∫_{ta}^{tb} Ψ̄(cy(t)) * dx dt
                    // cy(t) = cy0 + t*cdy. Ψ̄(cy) = cy if cy < 0.5, else 1-cy.
                    // For the piece where cy < 0.5: Ψ̄ = cy = cy0 + t*cdy.
                    //   ∫ (cy0 + t*cdy) * dx dt = dx * [cy0*(tb-ta) + cdy*(tb²-ta²)/2]
                    // For cy >= 0.5: Ψ̄ = 1 - cy = 1 - cy0 - t*cdy.
                    //   ∫ (1 - cy0 - t*cdy) * dx dt = dx * [(1-cy0)*(tb-ta) - cdy*(tb²-ta²)/2]
                    let dt = tb - ta;
                    let dt2 = tb * tb - ta * ta;
                    let integral = if cy_mid < 0.5 {
                        dx * (cy0 * dt + cdy * dt2 * 0.5)
                    } else {
                        dx * ((1.0 - cy0) * dt - cdy * dt2 * 0.5)
                    };
                    c01 -= integral;
                }
            }
            Segment::CubicBez(CubicBez { p0, p1, p2, p3 }) => {
                let ax = -p0.x + 3.0 * p1.x - 3.0 * p2.x + p3.x;
                let bx = 3.0 * p0.x - 6.0 * p1.x + 3.0 * p2.x;
                let cx = 3.0 * (p1.x - p0.x);
                let ay = -p0.y + 3.0 * p1.y - 3.0 * p2.y + p3.y;
                let by = 3.0 * p0.y - 6.0 * p1.y + 3.0 * p2.y;
                let cy = 3.0 * (p1.y - p0.y);
                // cell_y(t) = cells * y(t) - ky
                // y(t) = ay*t³ + by*t² + cy*t + p0.y
                // cell_y(t) = cells*(ay*t³ + by*t² + cy*t + p0.y) - ky
                let cay = cells as f64 * ay;
                let cby = cells as f64 * by;
                let ccy = cells as f64 * cy;
                let cdy = cells as f64 * p0.y - kyf;
                // Find t where cell_y = 0, 0.5, 1
                let mut ts = vec![0.0, 1.0];
                for val in [0.0, 0.5, 1.0] {
                    let (n, roots) = solve_cubic(cay, cby, ccy, cdy - val);
                    for &r in &roots[..n] {
                        if r > 0.0 && r < 1.0 {
                            ts.push(r);
                        }
                    }
                }
                ts.sort_by(|a, b| a.total_cmp(b));
                ts.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
                // Integrate piecewise using GL-16 on each piece
                const NODES: [f64; 8] = [
                    0.095_012_509_837_637_44,
                    0.281_603_550_779_258_9,
                    0.458_016_777_657_227_39,
                    0.617_876_244_402_643_8,
                    0.755_404_408_355_003,
                    0.865_631_202_387_831_8,
                    0.944_575_023_073_232_6,
                    0.989_400_934_991_649_9,
                ];
                const WEIGHTS: [f64; 8] = [
                    0.189_450_610_455_068_5,
                    0.182_603_415_044_924_59,
                    0.169_156_519_395_002_54,
                    0.149_595_988_816_576_73,
                    0.124_628_971_255_533_87,
                    0.095_158_511_682_492_78,
                    0.062_253_523_938_647_89,
                    0.027_152_459_411_754_09,
                ];
                for i in 0..ts.len() - 1 {
                    let ta = ts[i];
                    let tb = ts[i + 1];
                    let mid_t = (ta + tb) * 0.5;
                    let mid_cy = ((cay * mid_t + cby) * mid_t + ccy) * mid_t + cdy;
                    if mid_cy <= 0.0 || mid_cy >= 1.0 {
                        continue;
                    }
                    let half = (tb - ta) * 0.5;
                    let mid = (ta + tb) * 0.5;
                    let mut sum = 0.0;
                    for k in 0..8 {
                        let xn = half * NODES[k];
                        for &t in &[mid - xn, mid + xn] {
                            let cell_y = ((cay * t + cby) * t + ccy) * t + cdy;
                            let xp = (3.0 * ax * t + 2.0 * bx) * t + cx; // x'(t)
                            sum += WEIGHTS[k] * psi_bar(cell_y) * xp;
                        }
                    }
                    c01 -= sum * half;
                }
            }
            _ => {} // Arcs and superellipses: not used by this comparison
        }
    }

    c01 * cells_f
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contour::Shape;

    #[test]
    fn test_unit_square_coverage() {
        let shape = Shape::polygon(&[(10.0, 10.0), (11.0, 10.0), (11.0, 11.0), (10.0, 11.0)]);
        let pixels = reference_rasterize(&shape, 20, 20, FillRule::NonZero);
        assert!(
            (pixels[10 * 20 + 10] - 1.0).abs() < 1e-6,
            "unit square pixel: {}",
            pixels[10 * 20 + 10]
        );
        assert!(pixels[0].abs() < 1e-6, "outside pixel: {}", pixels[0]);
    }

    #[test]
    fn test_half_pixel_coverage() {
        let shape = Shape::polygon(&[(10.0, 10.0), (10.5, 10.0), (10.5, 11.0), (10.0, 11.0)]);
        let pixels = reference_rasterize(&shape, 20, 20, FillRule::NonZero);
        assert!(
            (pixels[10 * 20 + 10] - 0.5).abs() < 1e-6,
            "half-pixel coverage: {}",
            pixels[10 * 20 + 10]
        );
    }

    #[test]
    fn test_diagonal_coverage() {
        let shape = Shape::polygon(&[(10.0, 10.0), (11.0, 10.0), (10.0, 11.0)]);
        let pixels = reference_rasterize(&shape, 20, 20, FillRule::NonZero);
        assert!(
            (pixels[10 * 20 + 10] - 0.5).abs() < 1e-6,
            "diagonal half: {}",
            pixels[10 * 20 + 10]
        );
    }

    #[test]
    fn test_interior_pixel() {
        // Large rectangle — interior pixel should be exactly 1.0
        let shape = Shape::polygon(&[(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)]);
        let pixels = reference_rasterize(&shape, 20, 20, FillRule::NonZero);
        assert!(
            (pixels[10 * 20 + 10] - 1.0).abs() < 1e-10,
            "interior: {}",
            pixels[10 * 20 + 10]
        );
        assert!(pixels[0].abs() < 1e-10, "exterior: {}", pixels[0]);
    }

    #[test]
    fn test_circle_coverage() {
        let shape = Shape::circle(32.0, 32.0, 20.0);
        let pixels = reference_rasterize(&shape, 64, 64, FillRule::NonZero);
        assert!(
            (pixels[32 * 64 + 32].abs() - 1.0).abs() < 1e-4,
            "circle center: {}",
            pixels[32 * 64 + 32]
        );
        assert!(pixels[0].abs() < 1e-4, "circle outside: {}", pixels[0]);
    }
}
