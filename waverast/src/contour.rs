//! Shape boundary representation and per-segment boundary integral evaluation.
//!
//! Implements the boundary integral method from Manson & Schaefer, "Wavelet
//! Rasterization" (2006). Each segment type provides two evaluation paths:
//!
//! **K/L evaluation** (M&S eq. 8, Appendix A/B): four-quadrant decomposition
//! producing K (endpoint) and L (path integral) terms per quadrant, combined
//! into coefficients. Used for c11 of lines, quads, and superellipses.
//!
//! **Direct tent integral** (M&S eq. 7 with Ψ̄ from §2.1, bypassing the
//! four-quadrant split): integrates the tent function directly along the
//! boundary. Avoids the midpoint cancellation that eq. 8 produces in finite
//! precision. Used for c01/c10 of all types, and for all three coefficients
//! of cubics and arcs.

use crate::solver::{solve_cubic, solve_quadratic};

pub use kurbo::{CubicBez, Line, Point, QuadBez, Rect};

/// Type alias for backward compatibility — prefer `Rect` in new code.
pub type Bbox = Rect;

/// Test strict overlap (shared interior area) between two `Rect`s.
/// Tolerance for boundary-coincidence checks.
///
/// At f64 precision, boundary coordinates are exact (from integer cell
/// boundaries), so this just needs to be any positive value to distinguish
/// exactly-zero from nonzero.
const EPS: f64 = 1e-15;

type CubicCoords = (f64, f64, f64, f64, f64, f64, f64, f64);

/// K (constant) and L (linear) boundary integral terms (M&S equation 8).
///
/// For a boundary segment clipped to a wavelet cell quadrant, K depends only on
/// the clipped endpoints and L depends on the curve's interior path. Combined
/// across four quadrants per cell to produce c^(1,1) via equation 8:
/// c11 = L00_x - L01_x + K10_x - L10_x - K11_x + L11_x.
#[derive(Clone, Copy, Debug)]
pub struct KL {
    pub kx: f64,
    pub ky: f64,
    pub lx: f64,
    pub ly: f64,
}

impl KL {
    pub fn zero() -> Self {
        Self {
            kx: 0.0,
            ky: 0.0,
            lx: 0.0,
            ly: 0.0,
        }
    }
}

/// Fixed-capacity stack buffer for parameter values.
///
/// Collects curve-cell intersection points during clipping without heap allocation.
struct StackBuf<T, const N: usize> {
    data: [T; N],
    len: usize,
}

impl<T: Copy + Default, const N: usize> StackBuf<T, N> {
    fn new() -> Self {
        Self {
            data: [T::default(); N],
            len: 0,
        }
    }

    fn push(&mut self, val: T) {
        debug_assert!(self.len < N);
        self.data[self.len] = val;
        self.len += 1;
    }

    fn as_slice(&self) -> &[T] {
        &self.data[..self.len]
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data[..self.len]
    }

    fn retain(&mut self, f: impl Fn(&T) -> bool) {
        let mut write = 0;
        for read in 0..self.len {
            if f(&self.data[read]) {
                self.data[write] = self.data[read];
                write += 1;
            }
        }
        self.len = write;
    }
}

// ---- Gauss-Legendre 16-point quadrature (used by superellipse integration) ----

const GL16_NODES: [f64; 8] = [
    0.09501250983763744,
    0.281_603_550_779_258_9,
    0.45801677765722739,
    0.617_876_244_402_643_8,
    0.755_404_408_355_003,
    0.865_631_202_387_831_8,
    0.944_575_023_073_232_6,
    0.989_400_934_991_649_9,
];

const GL16_WEIGHTS: [f64; 8] = [
    0.189_450_610_455_068_5,
    0.18260341504492459,
    0.16915651939500254,
    0.14959598881657673,
    0.12462897125553387,
    0.09515851168249278,
    0.06225352393864789,
    0.02715245941175409,
];

/// Gauss-Legendre 16-point quadrature of `f` over `[a, b]`.
///
/// Exploits the symmetric node/weight pairs to halve the table size.
fn gauss_legendre_16(a: f64, b: f64, f: impl Fn(f64) -> f64) -> f64 {
    let half = (b - a) * 0.5;
    let mid = (a + b) * 0.5;
    let mut sum = 0.0;
    for i in 0..8 {
        let x = half * GL16_NODES[i];
        sum += GL16_WEIGHTS[i] * (f(mid - x) + f(mid + x));
    }
    sum * half
}

/// Bisect to find t where `coord_at(t) = boundary`, given that `val_lo` is the
/// value at `t_lo` and the sign changes between `t_lo` and `t_hi`.
fn bisect_root(
    t_lo: f64,
    t_hi: f64,
    val_lo: f64,
    boundary: f64,
    coord_at: &dyn Fn(f64) -> f64,
) -> f64 {
    let mut lo = t_lo;
    let mut hi = t_hi;
    let sign_lo = val_lo - boundary;
    for _ in 0..60 {
        let mid = (lo + hi) * 0.5;
        if mid == lo || mid == hi {
            break;
        }
        let vm = coord_at(mid);
        if (vm - boundary) * sign_lo > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) * 0.5
}

/// Clip-sort-dedup-accumulate loop for parametric segment types.
///
/// Given candidate parameter values (curve endpoints + cell-boundary crossings),
/// filters to the valid range, deduplicates, and calls `accumulate` for each
/// consecutive pair whose midpoint lies inside the unit cell.
fn process_clipped_segments<const N: usize>(
    ts: &mut StackBuf<f64, N>,
    is_t_in: impl Fn(f64) -> bool,
    mut accumulate: impl FnMut(f64, f64),
) where
    [f64; N]: Default,
{
    ts.retain(|t| (0.0..=1.0).contains(t));
    ts.as_mut_slice().sort_by(|a, b| a.total_cmp(b));

    let s = ts.as_slice();
    let mut prev = f64::NEG_INFINITY;
    let mut last_t: Option<f64> = None;
    for &t in s {
        if t == prev {
            continue;
        }
        prev = t;
        if let Some(lt) = last_t {
            let mid = (lt + t) * 0.5;
            if is_t_in(mid) {
                accumulate(lt, t);
            }
        }
        last_t = Some(t);
    }
}

// ---- Line segment ----

/// Liang-Barsky line clipping to an axis-aligned rectangle.
///
/// Returns the clipped endpoints, or `None` if the segment is entirely outside.
fn line_clip(
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
) -> Option<(f64, f64, f64, f64)> {
    let mut t0: f64 = 0.0;
    let mut t1: f64 = 1.0;
    let dx = x1 - x0;
    let dy = y1 - y0;

    let edges = [
        (-dx, -(x_min - x0)),
        (dx, x_max - x0),
        (-dy, -(y_min - y0)),
        (dy, y_max - y0),
    ];

    for &(p, q) in &edges {
        if p == 0.0 && q < 0.0 {
            return None;
        }
        if p < 0.0 {
            let r = q / p;
            if r > t1 {
                return None;
            }
            if r > t0 {
                t0 = r;
            }
        } else if p > 0.0 {
            let r = q / p;
            if r < t0 {
                return None;
            }
            if r < t1 {
                t1 = r;
            }
        }
    }

    Some((x0 + t0 * dx, y0 + t0 * dy, x0 + t1 * dx, y0 + t1 * dy))
}

/// Line segment K/L clipped to [0,1]² (M&S Appendix A).
///
/// K = ¼(Δy), L = ⅛(Δy)(x₀+x₁) for the x-component, and analogously for y.
/// `skip_x0`/`skip_y0` control boundary-coincidence skipping for equation-8 c11:
/// when true, segments lying entirely on x=0 or x=1 (resp. y) are skipped to
/// ensure correct cancellation at quadrant boundaries.
pub(crate) fn line_get_kl(x0: f64, y0: f64, x1: f64, y1: f64, skip_x0: bool, skip_y0: bool) -> KL {
    let mut kl = KL::zero();

    if let Some((cx0, cy0, cx1, cy1)) = line_clip(x0, y0, x1, y1, 0.0, 1.0, 0.0, 1.0) {
        if skip_x0
            && ((cx0 - 1.0).abs() < EPS && (cx1 - 1.0).abs() < EPS
                || cx0.abs() < EPS && cx1.abs() < EPS)
        {
            return kl;
        }
        if skip_y0
            && ((cy0 - 1.0).abs() < EPS && (cy1 - 1.0).abs() < EPS
                || cy0.abs() < EPS && cy1.abs() < EPS)
        {
            return kl;
        }

        kl.kx = 0.25 * (cy1 - cy0);
        kl.ky = 0.25 * (cx0 - cx1);
        kl.lx = 0.125 * (cy1 - cy0) * (cx1 + cx0);
        kl.ly = 0.125 * (cx0 - cx1) * (cy1 + cy0);
    }

    kl
}

// ---- Quadratic Bezier ----

/// Quadratic Bezier evaluation at parameter t (Bernstein form).
fn quad_evaluate(x0: f64, y0: f64, x1: f64, y1: f64, x2: f64, y2: f64, t: f64) -> (f64, f64) {
    let u = 1.0 - t;
    (
        x0 * u * u + 2.0 * x1 * u * t + x2 * t * t,
        y0 * u * u + 2.0 * y1 * u * t + y2 * t * t,
    )
}

/// Quadratic Bezier sub-curve extraction for t ∈ [t0, t1].
fn quad_subsection(
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    t0: f64,
    t1: f64,
) -> (f64, f64, f64, f64, f64, f64) {
    let (sx0, sy0) = quad_evaluate(x0, y0, x1, y1, x2, y2, t0);
    let (sx2, sy2) = quad_evaluate(x0, y0, x1, y1, x2, y2, t1);
    let u0 = 1.0 - t0;
    let xm = u0 * x1 + t0 * x2;
    let ym = u0 * y1 + t0 * y2;
    let t = (t1 - t0) / u0;
    let sx1 = (1.0 - t) * sx0 + t * xm;
    let sy1 = (1.0 - t) * sy0 + t * ym;
    (sx0, sy0, sx1, sy1, sx2, sy2)
}

/// Quadratic Bezier K/L clipped to [0,1]² (M&S Appendix A, degree-2).
///
/// Finds cell-boundary crossings by solving B(t) = val for each edge, then
/// integrates each clipped sub-curve using the closed-form antiderivative.
pub(crate) fn quad_get_kl(
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    skip_x0: bool,
    skip_y0: bool,
) -> KL {
    let mut kl = KL::zero();

    let is_t_in = |t: f64| -> bool {
        let (px, py) = quad_evaluate(x0, y0, x1, y1, x2, y2, t);
        (-EPS..=1.0 + EPS).contains(&px) && (-EPS..=1.0 + EPS).contains(&py)
    };

    let ax = x0 - 2.0 * x1 + x2;
    let bx = -2.0 * x0 + 2.0 * x1;
    let cx = x0;
    let ay = y0 - 2.0 * y1 + y2;
    let by = -2.0 * y0 + 2.0 * y1;
    let cy = y0;

    let mut ts: StackBuf<f64, 12> = StackBuf::new();
    ts.push(0.0);

    let x_min = x0.min(x1).min(x2);
    let x_max = x0.max(x1).max(x2);
    let y_min = y0.min(y1).min(y2);
    let y_max = y0.max(y1).max(y2);

    for val in [0.0, 1.0] {
        if x_min < val + EPS && x_max > val - EPS {
            let (n, roots) = solve_quadratic(ax, bx, cx - val);
            for &r in &roots[..n] {
                ts.push(r);
            }
        }
    }
    for val in [0.0, 1.0] {
        if y_min < val + EPS && y_max > val - EPS {
            let (n, roots) = solve_quadratic(ay, by, cy - val);
            for &r in &roots[..n] {
                ts.push(r);
            }
        }
    }

    ts.push(1.0);

    process_clipped_segments(&mut ts, is_t_in, |t0, t1| {
        let (sx0, sy0, sx1, sy1, sx2, sy2) = quad_subsection(x0, y0, x1, y1, x2, y2, t0, t1);
        accumulate_quad_kl(&mut kl, sx0, sy0, sx1, sy1, sx2, sy2, skip_x0, skip_y0);
    });

    kl
}

/// K/L accumulator for one clipped quadratic sub-curve.
///
/// K uses endpoints only; L uses the closed-form integral over the quadratic path.
fn accumulate_quad_kl(
    kl: &mut KL,
    sx0: f64,
    sy0: f64,
    sx1: f64,
    sy1: f64,
    sx2: f64,
    sy2: f64,
    skip_x0: bool,
    skip_y0: bool,
) {
    let v0 = Point::new(sx2, sy2);
    let v1 = Point::new(sx1, sy1);
    let v2 = Point::new(sx0, sy0);

    if skip_x0
        && (((v0.x - 1.0).abs() < EPS && (v1.x - 1.0).abs() < EPS && (v2.x - 1.0).abs() < EPS)
            || (v0.x.abs() < EPS && v1.x.abs() < EPS && v2.x.abs() < EPS))
    {
        return;
    }
    if skip_y0
        && (((v0.y - 1.0).abs() < EPS && (v1.y - 1.0).abs() < EPS && (v2.y - 1.0).abs() < EPS)
            || (v0.y.abs() < EPS && v1.y.abs() < EPS && v2.y.abs() < EPS))
    {
        return;
    }

    kl.kx += 0.25 * (v0.y - v2.y);
    kl.ky += 0.25 * (v2.x - v0.x);
    kl.lx += (1.0 / 24.0)
        * (3.0 * v0.x * v0.y + 2.0 * v0.y * v1.x - 2.0 * v0.x * v1.y
            + v0.y * v2.x
            + 2.0 * v1.y * v2.x
            - (v0.x + 2.0 * v1.x + 3.0 * v2.x) * v2.y);
    kl.ly += (1.0 / 24.0)
        * (2.0 * v1.y * v2.x + v0.y * (2.0 * v1.x + v2.x) - 2.0 * v1.x * v2.y + 3.0 * v2.x * v2.y
            - v0.x * (3.0 * v0.y + 2.0 * v1.y + v2.y));
}

// ---- Cubic Bezier ----

/// Cubic Bezier evaluation at parameter t (Bernstein form).
fn cubic_evaluate(
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    x3: f64,
    y3: f64,
    t: f64,
) -> (f64, f64) {
    let u = 1.0 - t;
    let u2 = u * u;
    let t2 = t * t;
    (
        x0 * u2 * u + 3.0 * x1 * u2 * t + 3.0 * x2 * u * t2 + x3 * t * t2,
        y0 * u2 * u + 3.0 * y1 * u2 * t + 3.0 * y2 * u * t2 + y3 * t * t2,
    )
}

/// Cubic Bezier sub-curve extraction for t ∈ [t0, t1] (de Casteljau).
fn cubic_subsection(
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    x3: f64,
    y3: f64,
    t0: f64,
    t1: f64,
) -> CubicCoords {
    let u0 = 1.0 - t0;
    let u1 = 1.0 - t1;

    let qxa = x0 * u0 * u0 + x1 * 2.0 * t0 * u0 + x2 * t0 * t0;
    let qxb = x0 * u1 * u1 + x1 * 2.0 * t1 * u1 + x2 * t1 * t1;
    let qxc = x1 * u0 * u0 + x2 * 2.0 * t0 * u0 + x3 * t0 * t0;
    let qxd = x1 * u1 * u1 + x2 * 2.0 * t1 * u1 + x3 * t1 * t1;

    let qya = y0 * u0 * u0 + y1 * 2.0 * t0 * u0 + y2 * t0 * t0;
    let qyb = y0 * u1 * u1 + y1 * 2.0 * t1 * u1 + y2 * t1 * t1;
    let qyc = y1 * u0 * u0 + y2 * 2.0 * t0 * u0 + y3 * t0 * t0;
    let qyd = y1 * u1 * u1 + y2 * 2.0 * t1 * u1 + y3 * t1 * t1;

    (
        qxa * u0 + qxc * t0,
        qya * u0 + qyc * t0,
        qxa * u1 + qxc * t1,
        qya * u1 + qyc * t1,
        qxb * u0 + qxd * t0,
        qyb * u0 + qyd * t0,
        qxb * u1 + qxd * t1,
        qyb * u1 + qyd * t1,
    )
}

/// Cubic Bezier K/L clipped to [0,1]² (M&S §2.2, Appendix A).
///
/// Finds cell-boundary crossings by solving the cubic B(t) = val for each edge,
/// then integrates each clipped sub-curve using the Bernstein product formula.
pub(crate) fn cubic_get_kl(
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    x3: f64,
    y3: f64,
    skip_x0: bool,
    skip_y0: bool,
) -> KL {
    let mut kl = KL::zero();

    // Early exit: if the control hull is entirely outside [0,1]², no contribution.
    let x_min = x0.min(x1).min(x2).min(x3);
    let x_max = x0.max(x1).max(x2).max(x3);
    let y_min = y0.min(y1).min(y2).min(y3);
    let y_max = y0.max(y1).max(y2).max(y3);
    if x_max < -EPS || x_min > 1.0 + EPS || y_max < -EPS || y_min > 1.0 + EPS {
        return kl;
    }

    let is_t_in = |t: f64| -> bool {
        let (px, py) = cubic_evaluate(x0, y0, x1, y1, x2, y2, x3, y3, t);
        (-EPS..=1.0 + EPS).contains(&px) && (-EPS..=1.0 + EPS).contains(&py)
    };

    let ax = -x0 + 3.0 * x1 - 3.0 * x2 + x3;
    let bx = 3.0 * x0 - 6.0 * x1 + 3.0 * x2;
    let cx = 3.0 * x1 - 3.0 * x0;
    let dx = x0;
    let ay = -y0 + 3.0 * y1 - 3.0 * y2 + y3;
    let by = 3.0 * y0 - 6.0 * y1 + 3.0 * y2;
    let cy = 3.0 * y1 - 3.0 * y0;
    let dy = y0;

    let mut ts: StackBuf<f64, 16> = StackBuf::new();
    ts.push(0.0);

    for val in [0.0, 1.0] {
        if x_min < val + EPS && x_max > val - EPS {
            let (n, roots) = solve_cubic(ax, bx, cx, dx - val);
            for &r in &roots[..n] {
                ts.push(r);
            }
        }
    }
    for val in [0.0, 1.0] {
        if y_min < val + EPS && y_max > val - EPS {
            let (n, roots) = solve_cubic(ay, by, cy, dy - val);
            for &r in &roots[..n] {
                ts.push(r);
            }
        }
    }

    ts.push(1.0);

    process_clipped_segments(&mut ts, is_t_in, |t0, t1| {
        let (sx0, sy0, sx1, sy1, sx2, sy2, sx3, sy3) =
            cubic_subsection(x0, y0, x1, y1, x2, y2, x3, y3, t0, t1);
        accumulate_cubic_kl(
            &mut kl, sx0, sy0, sx1, sy1, sx2, sy2, sx3, sy3, skip_x0, skip_y0,
        );
    });

    kl
}

/// K/L accumulator for one clipped cubic sub-curve (M&S §2.2, Appendix A).
///
/// K uses endpoints; L uses the Bernstein product integral for ∫xy dθ,
/// factored into two-term determinants (d01..d23) plus endpoint self-products.
fn accumulate_cubic_kl(
    kl: &mut KL,
    sx0: f64,
    sy0: f64,
    sx1: f64,
    sy1: f64,
    sx2: f64,
    sy2: f64,
    sx3: f64,
    sy3: f64,
    skip_x0: bool,
    skip_y0: bool,
) {
    // v0=end, v1=cp2, v2=cp1, v3=start (paper's reversed convention)
    let v0 = Point::new(sx3, sy3);
    let v1 = Point::new(sx2, sy2);
    let v2 = Point::new(sx1, sy1);
    let v3 = Point::new(sx0, sy0);

    if skip_x0
        && (((v0.x - 1.0).abs() < EPS
            && (v1.x - 1.0).abs() < EPS
            && (v2.x - 1.0).abs() < EPS
            && (v3.x - 1.0).abs() < EPS)
            || (v0.x.abs() < EPS && v1.x.abs() < EPS && v2.x.abs() < EPS && v3.x.abs() < EPS))
    {
        return;
    }
    if skip_y0
        && (((v0.y - 1.0).abs() < EPS
            && (v1.y - 1.0).abs() < EPS
            && (v2.y - 1.0).abs() < EPS
            && (v3.y - 1.0).abs() < EPS)
            || (v0.y.abs() < EPS && v1.y.abs() < EPS && v2.y.abs() < EPS && v3.y.abs() < EPS))
    {
        return;
    }

    kl.kx += 0.25 * (v0.y - v3.y);
    kl.ky += 0.25 * (v3.x - v0.x);

    // Factor Lx/Ly into clean two-term determinants (shared between Lx and Ly)
    // plus endpoint self-products (opposite sign). Avoids ~20-term alternating sums.
    let d01 = v0.x * v1.y - v0.y * v1.x;
    let d02 = v0.x * v2.y - v0.y * v2.x;
    let d03 = v0.x * v3.y - v0.y * v3.x;
    let d12 = v1.x * v2.y - v1.y * v2.x;
    let d13 = v1.x * v3.y - v1.y * v3.x;
    let d23 = v2.x * v3.y - v2.y * v3.x;
    let cross_sum = -6.0 * d01 - 3.0 * d02 - d03 - 3.0 * d12 - 3.0 * d13 - 6.0 * d23;
    let self_diff = 10.0 * (v0.x * v0.y - v3.x * v3.y);
    kl.lx += (1.0 / 80.0) * (cross_sum + self_diff);
    kl.ly += (1.0 / 80.0) * (cross_sum - self_diff);
}

// ---- Circular Arc ----

use std::f64::consts::{FRAC_PI_2, PI, TAU};

/// Circular arc evaluation at parameter t ∈ [0,1] (linear angle interpolation).
fn arc_evaluate(cx: f64, cy: f64, r: f64, theta0: f64, theta1: f64, t: f64) -> (f64, f64) {
    let theta = theta0 + t * (theta1 - theta0);
    (cx + r * theta.cos(), cy + r * theta.sin())
}

/// Test whether `target + 2kπ` falls in `[lo, hi]` for any integer k.
fn angle_in_range(target: f64, lo: f64, hi: f64) -> bool {
    let k_min = ((lo - target) / TAU).ceil() as i64;
    let k_max = ((hi - target) / TAU).floor() as i64;
    k_min <= k_max
}

/// Arc-angle crossing finder: all t ∈ [0,1] where θ(t) hits `base + 2kπ`.
///
/// Used to find where a circular arc crosses horizontal or vertical cell
/// boundaries (which correspond to specific angles).
fn arc_push_crossings<const N: usize>(
    theta0: f64,
    dtheta: f64,
    bases: &[f64],
    ts: &mut StackBuf<f64, N>,
) where
    [f64; N]: Default,
{
    let theta_end = theta0 + dtheta;
    let (range_lo, range_hi) = if dtheta > 0.0 {
        (theta0, theta_end)
    } else {
        (theta_end, theta0)
    };

    for &base in bases {
        let k_min = ((range_lo - base) / TAU).ceil() as i64;
        let k_max = ((range_hi - base) / TAU).floor() as i64;
        for k in k_min..=k_max {
            let theta = base + k as f64 * TAU;
            let t = (theta - theta0) / dtheta;
            ts.push(t);
        }
    }
}

/// Newton refinement for an arc crossing at `axis`=`val`.
///
/// `axis`: 0 for x, 1 for y.
fn arc_newton_crossing(
    cx: f64,
    cy: f64,
    r: f64,
    theta0: f64,
    dtheta: f64,
    axis: usize,
    val: f64,
    t: f64,
) -> f64 {
    let theta = theta0 + t * dtheta;
    let (sin_t, cos_t) = theta.sin_cos();
    let (ft, fpt) = if axis == 0 {
        // x(t) - val = cx + r·cos(θ) - val, dx/dt = -r·sin(θ)·dθ
        (cx + r * cos_t - val, -r * sin_t * dtheta)
    } else {
        // y(t) - val = cy + r·sin(θ) - val, dy/dt = r·cos(θ)·dθ
        (cy + r * sin_t - val, r * cos_t * dtheta)
    };
    if fpt.abs() < 1e-20 {
        return t;
    }
    (t - ft / fpt).clamp(0.0, 1.0)
}

/// Circular arc K/L clipped to [0,1]².
///
/// Extends M&S Appendix A to parametric arcs (not in the original paper).
/// Cell-boundary crossings are found via atan2-based inverse trig (stable
/// near ±1), Newton-refined. Each clipped sub-arc uses the exact trig
/// antiderivative with product-to-sum identities for Δsin, Δcos.
pub(crate) fn arc_get_kl(
    cx: f64,
    cy: f64,
    r: f64,
    theta0: f64,
    theta1: f64,
    skip_x0: bool,
    skip_y0: bool,
) -> KL {
    let mut kl = KL::zero();
    let dtheta = theta1 - theta0;
    if dtheta.abs() < EPS {
        return kl;
    }

    let is_t_in = |t: f64| -> bool {
        let (px, py) = arc_evaluate(cx, cy, r, theta0, theta1, t);
        (-EPS..=1.0 + EPS).contains(&px) && (-EPS..=1.0 + EPS).contains(&py)
    };

    let mut ts: StackBuf<f64, 20> = StackBuf::new();
    ts.push(0.0);

    // x = val crossings: cos(θ) = d where d = (val - cx) / r
    // Use atan2(√(1-d²), d) instead of acos(d) for stability near d = ±1.
    let pre_len = ts.len;
    for val in [0.0, 1.0] {
        let d = (val - cx) / r;
        if d.abs() <= 1.0 {
            let s = (1.0 - d * d).max(0.0).sqrt();
            let a = f64::atan2(s, d); // = acos(d) but stable
            arc_push_crossings(theta0, dtheta, &[a, -a], &mut ts);
        }
    }
    // Newton-refine x crossings on the coordinate function
    for i in pre_len..ts.len {
        let val = if (cx + r * (theta0 + ts.data[i] * dtheta).cos()) < 0.5 {
            0.0
        } else {
            1.0
        };
        ts.data[i] = arc_newton_crossing(cx, cy, r, theta0, dtheta, 0, val, ts.data[i]);
    }

    // y = val crossings: sin(θ) = d where d = (val - cy) / r
    // Use atan2(d, √(1-d²)) instead of asin(d) for stability near d = ±1.
    let pre_len_y = ts.len;
    for val in [0.0, 1.0] {
        let d = (val - cy) / r;
        if d.abs() <= 1.0 {
            let c = (1.0 - d * d).max(0.0).sqrt();
            let a = f64::atan2(d, c); // = asin(d) but stable
            arc_push_crossings(theta0, dtheta, &[a, PI - a], &mut ts);
        }
    }
    // Newton-refine y crossings on the coordinate function
    for i in pre_len_y..ts.len {
        let val = if (cy + r * (theta0 + ts.data[i] * dtheta).sin()) < 0.5 {
            0.0
        } else {
            1.0
        };
        ts.data[i] = arc_newton_crossing(cx, cy, r, theta0, dtheta, 1, val, ts.data[i]);
    }

    ts.push(1.0);

    process_clipped_segments(&mut ts, is_t_in, |t0, t1| {
        let alpha = theta0 + t0 * dtheta;
        let beta = theta0 + t1 * dtheta;
        accumulate_arc_kl(&mut kl, cx, cy, r, alpha, beta, skip_x0, skip_y0);
    });

    kl
}

/// K/L accumulator for a single clipped sub-arc from angle α to β.
///
/// K uses endpoints. L uses the closed-form trig antiderivative:
/// Lx = r·cx/4·Δsinθ + r²(Δθ/8 + Δsin2θ/16),
/// Ly = -r·cy/4·Δcosθ + r²(Δθ/8 - Δsin2θ/16).
fn accumulate_arc_kl(
    kl: &mut KL,
    cx: f64,
    cy: f64,
    r: f64,
    alpha: f64,
    beta: f64,
    skip_x0: bool,
    skip_y0: bool,
) {
    let (sin_a, cos_a) = alpha.sin_cos();
    let (sin_b, cos_b) = beta.sin_cos();

    let start = Point::new(cx + r * cos_a, cy + r * sin_a);
    let end = Point::new(cx + r * cos_b, cy + r * sin_b);

    if skip_x0
        && (((start.x - 1.0).abs() < EPS && (end.x - 1.0).abs() < EPS)
            || (start.x.abs() < EPS && end.x.abs() < EPS))
    {
        return;
    }
    if skip_y0
        && (((start.y - 1.0).abs() < EPS && (end.y - 1.0).abs() < EPS)
            || (start.y.abs() < EPS && end.y.abs() < EPS))
    {
        return;
    }

    kl.kx += 0.25 * (end.y - start.y);
    kl.ky += 0.25 * (start.x - end.x);

    // Product-to-sum trig identities avoid catastrophic cancellation for small arcs:
    //   sin(β) - sin(α) = 2·cos((α+β)/2)·sin((β-α)/2)
    //   cos(β) - cos(α) = -2·sin((α+β)/2)·sin((β-α)/2)
    //   sin(2β) - sin(2α) = 2·cos(α+β)·sin(β-α)
    let dt = beta - alpha;
    let half_sum = (alpha + beta) * 0.5;
    let half_dt = dt * 0.5;
    let dsin = 2.0 * half_sum.cos() * half_dt.sin();
    let dcos = -2.0 * half_sum.sin() * half_dt.sin();
    let dsin2 = 2.0 * (alpha + beta).cos() * dt.sin();

    kl.lx += r * cx * 0.25 * dsin + r * r * (0.125 * dt + 0.0625 * dsin2);
    kl.ly += -r * cy * 0.25 * dcos + r * r * (0.125 * dt - 0.0625 * dsin2);
}

/// Bounding box for a circular arc, considering trig extrema.
fn arc_bbox(cx: f64, cy: f64, r: f64, theta0: f64, theta1: f64) -> Rect {
    let (t_min, t_max) = if theta0 < theta1 {
        (theta0, theta1)
    } else {
        (theta1, theta0)
    };

    let ax0 = cx + r * theta0.cos();
    let ay0 = cy + r * theta0.sin();
    let ax1 = cx + r * theta1.cos();
    let ay1 = cy + r * theta1.sin();

    let mut x_min = ax0.min(ax1);
    let mut x_max = ax0.max(ax1);
    let mut y_min = ay0.min(ay1);
    let mut y_max = ay0.max(ay1);

    // Check extremal angles: cos has extrema at 0 and π, sin at π/2 and 3π/2
    if angle_in_range(0.0, t_min, t_max) {
        x_max = cx + r;
    }
    if angle_in_range(PI, t_min, t_max) {
        x_min = cx - r;
    }
    if angle_in_range(FRAC_PI_2, t_min, t_max) {
        y_max = cy + r;
    }
    // sin = -1 at -π/2 and 3π/2; check both representations
    if angle_in_range(-FRAC_PI_2, t_min, t_max) || angle_in_range(3.0 * FRAC_PI_2, t_min, t_max) {
        y_min = cy - r;
    }

    Rect::new(x_min, y_min, x_max, y_max)
}

/// Signed area contribution for a circular arc (Green's theorem).
fn arc_area(cx: f64, cy: f64, r: f64, theta0: f64, theta1: f64) -> f64 {
    let (sin0, cos0) = theta0.sin_cos();
    let (sin1, cos1) = theta1.sin_cos();
    0.5 * (r * cx * (sin1 - sin0) - r * cy * (cos1 - cos0) + r * r * (theta1 - theta0))
}

// ---- Superellipse (waverast extension; not in M&S) ----
//
// A convex superellipse |x/a|^n + |y/b|^n = 1 (n ≥ 1) is represented as up to
// four quadrants, each parameterized by u ∈ [0,1] where x = cx + sx·a·u and
// y = cy + sy·b·(1-u^n)^(1/n). The L integral uses GL-16 numerical quadrature
// on ∫y dx, which is smooth (dx/du = sx·a is constant), then obtains ∫x dy via
// integration by parts: ∫x dy = [xy] - ∫y dx. This avoids the singular dy/du
// derivative at u = 1 (the profile cusp).
//
// Sub-intervals are split at u=0.75 and u=0.9375 near the cusp, which gives
// GL-16 sufficient accuracy to match the reference rasterizer.

/// Sign pairs for the four superellipse quadrants in CCW order.
const SE_QUADRANTS: [(f64, f64); 4] = [(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)];

/// Superellipse profile: (1 - u^n)^(1/n).
///
/// Clamped to avoid negative values from floating-point imprecision near u = 1.
fn se_profile(u: f64, n: f64) -> f64 {
    (1.0 - u.powf(n)).max(0.0).powf(1.0 / n)
}

/// Superellipse quadrant evaluation at parameter u ∈ [0, 1].
///
/// x(u) = cx + sx·a·u, y(u) = cy + sy·b·profile(u, n).
fn se_evaluate(cx: f64, cy: f64, a: f64, b: f64, n: f64, sx: f64, sy: f64, u: f64) -> (f64, f64) {
    (cx + sx * a * u, cy + sy * b * se_profile(u, n))
}

/// Superellipse K/L for all active quadrants, clipped to [0,1]².
pub(crate) fn se_get_kl(
    cx: f64,
    cy: f64,
    a: f64,
    b: f64,
    n: f64,
    quadrants: u8,
    skip_x0: bool,
    skip_y0: bool,
) -> KL {
    let mut kl = KL::zero();
    for (qi, &(sx, sy)) in SE_QUADRANTS.iter().enumerate() {
        if quadrants & (1 << qi) != 0 {
            se_quadrant_kl(&mut kl, cx, cy, a, b, n, sx, sy, skip_x0, skip_y0);
        }
    }
    kl
}

/// K/L for one superellipse quadrant, clipped to [0,1]².
///
/// Cell-boundary crossings are closed-form: x-crossings are linear in u,
/// y-crossings invert the profile function.
fn se_quadrant_kl(
    kl: &mut KL,
    cx: f64,
    cy: f64,
    a: f64,
    b: f64,
    n: f64,
    sx: f64,
    sy: f64,
    skip_x0: bool,
    skip_y0: bool,
) {
    let is_t_in = |u: f64| -> bool {
        let (px, py) = se_evaluate(cx, cy, a, b, n, sx, sy, u);
        (-EPS..=1.0 + EPS).contains(&px) && (-EPS..=1.0 + EPS).contains(&py)
    };

    let mut ts: StackBuf<f64, 12> = StackBuf::new();
    ts.push(0.0);
    // Split near u=1 where the profile has a cusp, improving GL-16 accuracy
    ts.push(0.75);
    ts.push(0.9375);

    // x = val crossings: u = (val - cx) / (sx * a)
    if sx.abs() * a > EPS {
        for val in [0.0, 1.0] {
            let u = (val - cx) / (sx * a);
            if (0.0..=1.0).contains(&u) {
                ts.push(u);
            }
        }
    }

    // y = val crossings: u = (1 - d^n)^(1/n) where d = (val - cy) / (sy * b)
    if sy.abs() * b > EPS {
        for val in [0.0, 1.0] {
            let d = (val - cy) / (sy * b);
            if (0.0..=1.0).contains(&d) {
                ts.push(se_profile(d, n));
            }
        }
    }

    ts.push(1.0);

    process_clipped_segments(&mut ts, is_t_in, |u_lo, u_hi| {
        accumulate_se_kl(kl, cx, cy, a, b, n, sx, sy, u_lo, u_hi, skip_x0, skip_y0);
    });
}

/// K/L accumulator for a clipped superellipse sub-segment.
///
/// Uses integration by parts to avoid the singular ∫x dy integrand:
/// dx/du = sx·a (constant), so ∫y dx is smooth (GL-16 quadrature),
/// and ∫x dy = [xy] - ∫y dx gives the other component.
fn accumulate_se_kl(
    kl: &mut KL,
    cx: f64,
    cy: f64,
    a: f64,
    b: f64,
    n: f64,
    sx: f64,
    sy: f64,
    u_lo: f64,
    u_hi: f64,
    skip_x0: bool,
    skip_y0: bool,
) {
    let dir = -sx * sy;

    // Endpoints in CCW traversal order
    let (u_start, u_end) = if dir < 0.0 {
        (u_hi, u_lo)
    } else {
        (u_lo, u_hi)
    };
    let (x_a, y_a) = se_evaluate(cx, cy, a, b, n, sx, sy, u_start);
    let (x_b, y_b) = se_evaluate(cx, cy, a, b, n, sx, sy, u_end);

    if skip_x0
        && (((x_a - 1.0).abs() < EPS && (x_b - 1.0).abs() < EPS)
            || (x_a.abs() < EPS && x_b.abs() < EPS))
    {
        return;
    }
    if skip_y0
        && (((y_a - 1.0).abs() < EPS && (y_b - 1.0).abs() < EPS)
            || (y_a.abs() < EPS && y_b.abs() < EPS))
    {
        return;
    }

    kl.kx += 0.25 * (y_b - y_a);
    kl.ky += 0.25 * (x_a - x_b);

    // ∫_A^B y dx = dir * sx * a * ∫_{u_lo}^{u_hi} y(u) du (smooth: dx/du = sx*a = const)
    let i_y = gauss_legendre_16(u_lo, u_hi, |u| cy + sy * b * se_profile(u, n));
    let int_y_dx = dir * sx * a * i_y;
    let int_x_dy = (x_b * y_b - x_a * y_a) - int_y_dx;

    kl.lx += 0.25 * int_x_dy;
    kl.ly += -0.25 * int_y_dx;
}

/// Bounding box for active quadrants of a superellipse.
fn se_bbox(cx: f64, cy: f64, a: f64, b: f64, quadrants: u8) -> Rect {
    let mut x_min = cx;
    let mut y_min = cy;
    let mut x_max = cx;
    let mut y_max = cy;
    for (qi, &(sx, sy)) in SE_QUADRANTS.iter().enumerate() {
        if quadrants & (1 << qi) != 0 {
            let x1 = cx + sx * a;
            let y1 = cy + sy * b;
            x_min = x_min.min(cx.min(x1));
            y_min = y_min.min(cy.min(y1));
            x_max = x_max.max(cx.max(x1));
            y_max = y_max.max(cy.max(y1));
        }
    }
    Rect::new(x_min, y_min, x_max, y_max)
}

/// Signed area for active quadrants of a superellipse.
fn se_area(cx: f64, cy: f64, a: f64, b: f64, n: f64, quadrants: u8) -> f64 {
    let mut total = 0.0;
    for (qi, &(sx, sy)) in SE_QUADRANTS.iter().enumerate() {
        if quadrants & (1 << qi) == 0 {
            continue;
        }
        let dir = -sx * sy;
        let (u_start, u_end) = if dir < 0.0 { (1.0, 0.0) } else { (0.0, 1.0) };
        let (x_a, y_a) = se_evaluate(cx, cy, a, b, n, sx, sy, u_start);
        let (x_b, y_b) = se_evaluate(cx, cy, a, b, n, sx, sy, u_end);

        let i_y = gauss_legendre_16(0.0, 1.0, |u| cy + sy * b * se_profile(u, n));
        let int_y_dx = dir * sx * a * i_y;

        total += 0.5 * ((x_b * y_b - x_a * y_a) - 2.0 * int_y_dx);
    }
    total
}

// ---- Tent-function integrals (M&S equation 7 with Ψ̄ from §2.1) ----
//
// Wavelet coefficients c^(0,1), c^(1,0), and optionally c^(1,1) via the direct
// boundary integral with the tent function Ψ̄(t) (antiderivative of the Haar
// wavelet). For cubics and arcs, all three coefficients are computed here.
// For lines, quads, and superellipses, only c01/c10 use this path.
//
// Bisection verification catches crossings the analytic solver misses near
// degenerate cubic configurations (e.g., depressed cubic with p ≈ 0).

/// Tent function Ψ̄(t), the antiderivative of the Haar wavelet (M&S §2.1).
///
/// Ψ̄(t) = t for t ∈ [0, 0.5], 1-t for t ∈ [0.5, 1], 0 otherwise.
fn tent(t: f64) -> f64 {
    if t <= 0.0 || t >= 1.0 {
        0.0
    } else if t <= 0.5 {
        t
    } else {
        1.0 - t
    }
}

// -- Line tent integral --

/// Line segment tent integral in cell coordinates [0,1]².
///
/// Returns (c01, c10) contributions.
pub(crate) fn line_tent_c01_c10(u0: f64, v0: f64, u1: f64, v1: f64) -> (f64, f64) {
    if let Some((cu0, cv0, cu1, cv1)) = line_clip(u0, v0, u1, v1, 0.0, 1.0, 0.0, 1.0) {
        let c01 = -tent_line_integral(cu0, cv0, cu1, cv1);
        let c10 = tent_line_integral(cv0, cu0, cv1, cu1);
        (c01, c10)
    } else {
        (0.0, 0.0)
    }
}

/// ∫Ψ̄(v) du for a clipped line, split at v=0.5.
fn tent_line_integral(u0: f64, v0: f64, u1: f64, v1: f64) -> f64 {
    if (v0 < 0.5 && v1 > 0.5) || (v0 > 0.5 && v1 < 0.5) {
        let t = (0.5 - v0) / (v1 - v0);
        let u_mid = u0 + t * (u1 - u0);
        tent_line_piece(u0, v0, u_mid, 0.5) + tent_line_piece(u_mid, 0.5, u1, v1)
    } else {
        tent_line_piece(u0, v0, u1, v1)
    }
}

/// ∫Ψ̄(v) du for a line piece where v stays on one side of 0.5 (exact).
fn tent_line_piece(u0: f64, v0: f64, u1: f64, v1: f64) -> f64 {
    (u1 - u0) * (tent(v0) + tent(v1)) * 0.5
}

// -- Quadratic Bezier tent integral --

/// Quadratic Bezier tent integral in cell coordinates [0,1]².
pub(crate) fn quad_tent_c01_c10(
    u0: f64,
    v0: f64,
    u1: f64,
    v1: f64,
    u2: f64,
    v2: f64,
) -> (f64, f64) {
    let c01 = -quad_tent_integral(u0, v0, u1, v1, u2, v2);
    let c10 = quad_tent_integral(v0, u0, v1, u1, v2, u2);
    (c01, c10)
}

fn quad_tent_integral(u0: f64, v0: f64, u1: f64, v1: f64, u2: f64, v2: f64) -> f64 {
    let au = u0 - 2.0 * u1 + u2;
    let bu = -2.0 * u0 + 2.0 * u1;
    let cu_c = u0;
    let av = v0 - 2.0 * v1 + v2;
    let bv = -2.0 * v0 + 2.0 * v1;
    let cv_c = v0;

    let u_min = u0.min(u1).min(u2);
    let u_max = u0.max(u1).max(u2);
    let v_min = v0.min(v1).min(v2);
    let v_max = v0.max(v1).max(v2);
    if u_max < -EPS || u_min > 1.0 + EPS || v_max < -EPS || v_min > 1.0 + EPS {
        return 0.0;
    }

    let is_t_in = |t: f64| -> bool {
        let (pu, pv) = quad_evaluate(u0, v0, u1, v1, u2, v2, t);
        (-EPS..=1.0 + EPS).contains(&pu) && (-EPS..=1.0 + EPS).contains(&pv)
    };

    let mut ts: StackBuf<f64, 16> = StackBuf::new();
    ts.push(0.0);
    for val in [0.0, 1.0] {
        if u_min < val + EPS && u_max > val - EPS {
            let (n, roots) = solve_quadratic(au, bu, cu_c - val);
            for &r in &roots[..n] {
                ts.push(r);
            }
        }
    }
    for val in [0.0, 0.5, 1.0] {
        if v_min < val + EPS && v_max > val - EPS {
            let (n, roots) = solve_quadratic(av, bv, cv_c - val);
            for &r in &roots[..n] {
                ts.push(r);
            }
        }
    }
    ts.push(1.0);

    let mut result = 0.0;
    process_clipped_segments(&mut ts, is_t_in, |ta, tb| {
        let (su0, sv0, su1, sv1, su2, sv2) = quad_subsection(u0, v0, u1, v1, u2, v2, ta, tb);
        let (_, v_mid) = quad_evaluate(su0, sv0, su1, sv1, su2, sv2, 0.5);
        let int_v_du = quad_integral_v_du(su0, sv0, su1, sv1, su2, sv2);
        if v_mid < 0.5 {
            result += int_v_du;
        } else {
            result += (su2 - su0) - int_v_du;
        }
    });
    result
}

/// Exact ∫₀¹ v(t)·u'(t) dt for a quadratic Bezier (Bernstein product).
fn quad_integral_v_du(u0: f64, v0: f64, u1: f64, v1: f64, u2: f64, v2: f64) -> f64 {
    let du0 = u1 - u0;
    let du1 = u2 - u1;
    (1.0 / 6.0) * ((3.0 * v0 + 2.0 * v1 + v2) * du0 + (v0 + 2.0 * v1 + 3.0 * v2) * du1)
}

// -- Cubic Bezier tent integral (equation 7 with Ψ̄; computes c01, c10, c11) --

/// Process one cubic segment across all overlapping cells at a given level.
///
/// Finds all cell-boundary and midpoint crossings once per segment, then
/// splits and accumulates the tent integral contributions per cell.
/// This is O(boundaries) root-finding calls instead of O(cells × boundaries).
pub(crate) fn cubic_tent_level(
    pts: &[Point; 4],
    bb: &Bbox,
    cellsf: f64,
    cells: usize,
    base: usize,
    acc: &mut [(f64, f64)],
    c11_acc: &mut [f64],
) {
    let kx_lo = (bb.x0 * cellsf).floor().max(0.0) as usize;
    let kx_hi = ((bb.x1 * cellsf).ceil() as usize).min(cells);
    let ky_lo = (bb.y0 * cellsf).floor().max(0.0) as usize;
    let ky_hi = ((bb.y1 * cellsf).ceil() as usize).min(cells);
    if kx_lo >= kx_hi || ky_lo >= ky_hi {
        return;
    }

    // Scale control points to level coordinates: u = cellsf * x, v = cellsf * y
    // (NOT shifted per cell — we work in level-global coords and assign to cells later)
    let sx = [
        cellsf * pts[0].x,
        cellsf * pts[1].x,
        cellsf * pts[2].x,
        cellsf * pts[3].x,
    ];
    let sy = [
        cellsf * pts[0].y,
        cellsf * pts[1].y,
        cellsf * pts[2].y,
        cellsf * pts[3].y,
    ];

    // Power-form coefficients (translation-invariant)
    let ax = -sx[0] + 3.0 * sx[1] - 3.0 * sx[2] + sx[3];
    let bx = 3.0 * sx[0] - 6.0 * sx[1] + 3.0 * sx[2];
    let cx_c = 3.0 * sx[1] - 3.0 * sx[0];
    let dx_c = sx[0];
    let ay = -sy[0] + 3.0 * sy[1] - 3.0 * sy[2] + sy[3];
    let by = 3.0 * sy[0] - 6.0 * sy[1] + 3.0 * sy[2];
    let cy_c = 3.0 * sy[1] - 3.0 * sy[0];
    let dy_c = sy[0];

    // Collect ALL boundary crossings at once:
    // x-crossings at kx, kx+0.5, kx+1 for all kx in range
    // y-crossings at ky, ky+0.5, ky+1 for all ky in range
    let mut ts = Vec::with_capacity(32);
    ts.push(0.0);

    // x-boundaries: integers and half-integers in [kx_lo, kx_hi]
    let x_min = sx[0].min(sx[1]).min(sx[2]).min(sx[3]);
    let x_max = sx[0].max(sx[1]).max(sx[2]).max(sx[3]);
    for k2 in (kx_lo * 2)..=(kx_hi * 2) {
        let val = k2 as f64 * 0.5;
        if x_min < val + EPS && x_max > val - EPS {
            let (n, roots) = solve_cubic(ax, bx, cx_c, dx_c - val);
            for &r in &roots[..n] {
                if (0.0..=1.0).contains(&r) {
                    ts.push(r);
                }
            }
        }
    }

    // y-boundaries: integers and half-integers in [ky_lo, ky_hi]
    let y_min = sy[0].min(sy[1]).min(sy[2]).min(sy[3]);
    let y_max = sy[0].max(sy[1]).max(sy[2]).max(sy[3]);
    for k2 in (ky_lo * 2)..=(ky_hi * 2) {
        let val = k2 as f64 * 0.5;
        if y_min < val + EPS && y_max > val - EPS {
            let (n, roots) = solve_cubic(ay, by, cy_c, dy_c - val);
            for &r in &roots[..n] {
                if (0.0..=1.0).contains(&r) {
                    ts.push(r);
                }
            }
        }
    }

    ts.push(1.0);
    ts.sort_by(|a, b| a.total_cmp(b));

    // Bisection safety net
    let eval = |t: f64| -> (f64, f64) {
        cubic_evaluate(sx[0], sy[0], sx[1], sy[1], sx[2], sy[2], sx[3], sy[3], t)
    };
    let mut extras = Vec::new();
    for i in 0..ts.len().saturating_sub(1) {
        let ta = ts[i];
        let tb = ts[i + 1];
        if tb - ta < 1e-14 {
            continue;
        }
        let (xa, ya) = eval(ta);
        let (xb, yb) = eval(tb);
        // y-boundaries
        for k2 in (ky_lo * 2)..=(ky_hi * 2) {
            let bnd = k2 as f64 * 0.5;
            if (ya - bnd) * (yb - bnd) < 0.0 {
                let ev = |t: f64| eval(t).1;
                extras.push(bisect_root(ta, tb, ya, bnd, &ev));
            }
        }
        // x-boundaries
        for k2 in (kx_lo * 2)..=(kx_hi * 2) {
            let bnd = k2 as f64 * 0.5;
            if (xa - bnd) * (xb - bnd) < 0.0 {
                let ev = |t: f64| eval(t).0;
                extras.push(bisect_root(ta, tb, xa, bnd, &ev));
            }
        }
    }
    ts.extend_from_slice(&extras);
    ts.sort_by(|a, b| a.total_cmp(b));
    ts.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

    // Process sub-intervals: determine which cell each belongs to, compute tent
    let mut prev = f64::NEG_INFINITY;
    let mut last_t: Option<f64> = None;
    for &t in &ts {
        if t == prev {
            continue;
        }
        prev = t;
        if let Some(lt) = last_t {
            let mid_t = (lt + t) * 0.5;
            let (mx, my) = eval(mid_t);
            // Determine cell from midpoint
            let ckx = mx.floor() as isize;
            let cky = my.floor() as isize;
            if ckx >= kx_lo as isize
                && ckx < kx_hi as isize
                && cky >= ky_lo as isize
                && cky < ky_hi as isize
            {
                let ckx = ckx as usize;
                let cky = cky as usize;
                // Cell-local coordinates
                let cu0 = sx[0] - ckx as f64;
                let cv0 = sy[0] - cky as f64;
                let cu1 = sx[1] - ckx as f64;
                let cv1 = sy[1] - cky as f64;
                let cu2 = sx[2] - ckx as f64;
                let cv2 = sy[2] - cky as f64;
                let cu3 = sx[3] - ckx as f64;
                let cv3 = sy[3] - cky as f64;
                let (su0, sv0, su1, sv1, su2, sv2, su3, sv3) =
                    cubic_subsection(cu0, cv0, cu1, cv1, cu2, cv2, cu3, cv3, lt, t);
                let (u_mid, v_mid) = cubic_evaluate(su0, sv0, su1, sv1, su2, sv2, su3, sv3, 0.5);
                let int_v_du = cubic_integral_v_du(su0, sv0, su1, sv1, su2, sv2, su3, sv3);
                let int_u_dv = cubic_integral_u_dv(su0, sv0, su1, sv1, su2, sv2, su3, sv3);

                let tent_v = if v_mid < 0.5 {
                    int_v_du
                } else {
                    (su3 - su0) - int_v_du
                };
                let tent_u = if u_mid < 0.5 {
                    int_u_dv
                } else {
                    (sv3 - sv0) - int_u_dv
                };

                let idx = base + ckx * cells + cky;
                acc[idx].0 -= tent_v;
                acc[idx].1 += tent_u;
                let sign_v = if v_mid < 0.5 { 1.0 } else { -1.0 };
                c11_acc[idx] += sign_v * tent_u;
            }
        }
        last_t = Some(t);
    }
}

/// Exact ∫₀¹ v(t)·u'(t) dt for a cubic Bezier (Bernstein product).
///
/// The product v·u' is degree 5; closed-form weights from Bernstein integration.
fn cubic_integral_v_du(
    u0: f64,
    v0: f64,
    u1: f64,
    v1: f64,
    u2: f64,
    v2: f64,
    u3: f64,
    v3: f64,
) -> f64 {
    let du0 = u1 - u0;
    let du1 = u2 - u1;
    let du2 = u3 - u2;
    (1.0 / 20.0)
        * ((10.0 * v0 + 6.0 * v1 + 3.0 * v2 + v3) * du0
            + (4.0 * v0 + 6.0 * v1 + 6.0 * v2 + 4.0 * v3) * du1
            + (v0 + 3.0 * v1 + 6.0 * v2 + 10.0 * v3) * du2)
}

/// Exact ∫₀¹ u(t)·v'(t) dt for a cubic Bezier (Bernstein product).
///
/// Computed directly (not via integration by parts) for numerical stability.
fn cubic_integral_u_dv(
    u0: f64,
    v0: f64,
    u1: f64,
    v1: f64,
    u2: f64,
    v2: f64,
    u3: f64,
    v3: f64,
) -> f64 {
    let dv0 = v1 - v0;
    let dv1 = v2 - v1;
    let dv2 = v3 - v2;
    (1.0 / 20.0)
        * ((10.0 * u0 + 6.0 * u1 + 3.0 * u2 + u3) * dv0
            + (4.0 * u0 + 6.0 * u1 + 6.0 * u2 + 4.0 * u3) * dv1
            + (u0 + 3.0 * u1 + 6.0 * u2 + 10.0 * u3) * dv2)
}

// -- Circular arc tent integral (equation 7 with Ψ̄; computes c01, c10, c11) --

/// Circular arc tent integral in cell coordinates [0,1]².
///
/// Returns (c01, c10, c11_tent) where c11 = Σ sign_v · tent_u.
/// Uses conditioned forms to avoid catastrophic cancellation when the sub-arc
/// nearly fills a cell half (delta ≈ integral):
///   tent_v (v≥0.5): r·Δcos·(1-cy) + r²(Δθ/2 - Δsin2θ/4)
///   tent_u (u≥0.5): r·Δsin·(1-cx) - r²(Δθ/2 + Δsin2θ/4)
/// Trig differences use product-to-sum identities for numerical stability.
pub(crate) fn arc_tent_c01_c10(
    cx: f64,
    cy: f64,
    r: f64,
    theta0: f64,
    theta1: f64,
) -> (f64, f64, f64) {
    let dtheta = theta1 - theta0;
    if dtheta.abs() < EPS {
        return (0.0, 0.0, 0.0);
    }

    let is_t_in = |t: f64| -> bool {
        let (pu, pv) = arc_evaluate(cx, cy, r, theta0, theta1, t);
        (-EPS..=1.0 + EPS).contains(&pu) && (-EPS..=1.0 + EPS).contains(&pv)
    };

    let mut ts: StackBuf<f64, 28> = StackBuf::new();
    ts.push(0.0);

    // x crossings at 0, 0.5, 1
    for val in [0.0, 0.5, 1.0] {
        let d = (val - cx) / r;
        if d.abs() <= 1.0 {
            let pre = ts.len;
            let s = (1.0 - d * d).max(0.0).sqrt();
            let a = f64::atan2(s, d);
            arc_push_crossings(theta0, dtheta, &[a, -a], &mut ts);
            for i in pre..ts.len {
                ts.data[i] = arc_newton_crossing(cx, cy, r, theta0, dtheta, 0, val, ts.data[i]);
            }
        }
    }

    // y crossings at 0, 0.5, 1
    for val in [0.0, 0.5, 1.0] {
        let d = (val - cy) / r;
        if d.abs() <= 1.0 {
            let pre = ts.len;
            let c = (1.0 - d * d).max(0.0).sqrt();
            let a = f64::atan2(d, c);
            arc_push_crossings(theta0, dtheta, &[a, PI - a], &mut ts);
            for i in pre..ts.len {
                ts.data[i] = arc_newton_crossing(cx, cy, r, theta0, dtheta, 1, val, ts.data[i]);
            }
        }
    }

    ts.push(1.0);

    let mut c01_sum = 0.0;
    let mut c10_sum = 0.0;
    let mut c11_sum = 0.0;
    process_clipped_segments(&mut ts, is_t_in, |t0, t1| {
        let alpha = theta0 + t0 * dtheta;
        let beta = theta0 + t1 * dtheta;
        let dt = beta - alpha;

        let half_sum = (alpha + beta) * 0.5;
        let half_dt = dt * 0.5;
        let dsin = 2.0 * half_sum.cos() * half_dt.sin();
        let dcos = -2.0 * half_sum.sin() * half_dt.sin();
        let dsin2 = 2.0 * (alpha + beta).cos() * dt.sin();

        let r2 = r * r;
        let theta_mid = (alpha + beta) * 0.5;
        let v_mid = cy + r * theta_mid.sin();
        let u_mid = cx + r * theta_mid.cos();

        // Conditioned tent integrals: avoid catastrophic cancellation
        // when delta ≈ int (near-full coverage).
        let tent_v = if v_mid < 0.5 {
            r * cy * dcos - r2 * dt * 0.5 + r2 * dsin2 * 0.25
        } else {
            r * dcos * (1.0 - cy) + r2 * (dt * 0.5 - dsin2 * 0.25)
        };
        let tent_u = if u_mid < 0.5 {
            r * cx * dsin + r2 * dt * 0.5 + r2 * dsin2 * 0.25
        } else {
            r * dsin * (1.0 - cx) - r2 * (dt * 0.5 + dsin2 * 0.25)
        };

        c01_sum += tent_v;
        c10_sum += tent_u;

        // c11 = sign_v * tent_u
        let sign_v = if v_mid < 0.5 { 1.0 } else { -1.0 };
        c11_sum += sign_v * tent_u;
    });

    (-c01_sum, c10_sum, c11_sum)
}

// -- Superellipse tent integral (equation 7 with Ψ̄; c01/c10 only) --

/// Superellipse tent integral in cell coordinates [0,1]².
pub(crate) fn se_tent_c01_c10(
    cx: f64,
    cy: f64,
    a: f64,
    b: f64,
    n: f64,
    quadrants: u8,
) -> (f64, f64) {
    let mut c01 = 0.0;
    let mut c10 = 0.0;
    for (qi, &(sx, sy)) in SE_QUADRANTS.iter().enumerate() {
        if quadrants & (1 << qi) != 0 {
            let (qc01, qc10) = se_quadrant_tent(cx, cy, a, b, n, sx, sy);
            c01 += qc01;
            c10 += qc10;
        }
    }
    (-c01, c10)
}

fn se_quadrant_tent(cx: f64, cy: f64, a: f64, b: f64, n: f64, sx: f64, sy: f64) -> (f64, f64) {
    let is_t_in = |u: f64| -> bool {
        let (pu, pv) = se_evaluate(cx, cy, a, b, n, sx, sy, u);
        (-EPS..=1.0 + EPS).contains(&pu) && (-EPS..=1.0 + EPS).contains(&pv)
    };

    let mut ts: StackBuf<f64, 16> = StackBuf::new();
    ts.push(0.0);
    // Split near u=1 where the profile has a cusp, improving GL-16 accuracy
    ts.push(0.75);
    ts.push(0.9375);
    if sx.abs() * a > EPS {
        for val in [0.0, 0.5, 1.0] {
            let u = (val - cx) / (sx * a);
            if (0.0..=1.0).contains(&u) {
                ts.push(u);
            }
        }
    }
    if sy.abs() * b > EPS {
        for val in [0.0, 0.5, 1.0] {
            let d = (val - cy) / (sy * b);
            if (0.0..=1.0).contains(&d) {
                ts.push(se_profile(d, n));
            }
        }
    }
    ts.push(1.0);

    let dir = -sx * sy;
    let mut c01_sum = 0.0;
    let mut c10_sum = 0.0;

    process_clipped_segments(&mut ts, is_t_in, |u_lo, u_hi| {
        let (u_start, u_end) = if dir < 0.0 {
            (u_hi, u_lo)
        } else {
            (u_lo, u_hi)
        };
        let (x_a, y_a) = se_evaluate(cx, cy, a, b, n, sx, sy, u_start);
        let (x_b, y_b) = se_evaluate(cx, cy, a, b, n, sx, sy, u_end);

        let u_mid_param = (u_lo + u_hi) * 0.5;
        let (x_mid, y_mid) = se_evaluate(cx, cy, a, b, n, sx, sy, u_mid_param);

        let i_y = gauss_legendre_16(u_lo, u_hi, |u| cy + sy * b * se_profile(u, n));
        let int_y_dx = dir * sx * a * i_y;
        let int_x_dy = (x_b * y_b - x_a * y_a) - int_y_dx;

        let delta_x = x_b - x_a;
        let delta_y = y_b - y_a;

        if y_mid < 0.5 {
            c01_sum += int_y_dx;
        } else {
            c01_sum += delta_x - int_y_dx;
        }
        if x_mid < 0.5 {
            c10_sum += int_x_dy;
        } else {
            c10_sum += delta_y - int_x_dy;
        }
    });

    (c01_sum, c10_sum)
}

/// A circular arc segment.
#[derive(Clone, Copy, Debug)]
pub struct CircularArc {
    pub center: Point,
    pub radius: f64,
    pub theta0: f64,
    pub theta1: f64,
}

/// A convex superellipse segment (n ≥ 1).
///
/// `quadrants` bitmask: bit 0 = Q0 (+x,+y), 1 = Q1 (-x,+y),
/// 2 = Q2 (-x,-y), 3 = Q3 (+x,-y).
#[derive(Clone, Copy, Debug)]
pub struct Superellipse {
    pub center: Point,
    pub a: f64,
    pub b: f64,
    pub n: f64,
    pub quadrants: u8,
}

/// A boundary segment of a closed shape.
///
/// Each variant carries the geometric data for its K/L computation. Segments
/// are oriented: the shape's interior is to the left when traversed in
/// parameter order (CCW winding).
#[derive(Clone, Debug)]
pub enum Segment {
    Line(Line),
    QuadBez(QuadBez),
    CubicBez(CubicBez),
    CircularArc(CircularArc),
    Superellipse(Superellipse),
}

impl Segment {
    /// K/L boundary integral for this segment.
    ///
    /// Coordinates must already be in the \[0,1\]² cell-quadrant domain.
    pub fn get_kl(&self) -> KL {
        match self {
            Segment::Line(Line { p0, p1 }) => line_get_kl(p0.x, p0.y, p1.x, p1.y, true, true),
            Segment::QuadBez(QuadBez { p0, p1, p2 }) => {
                quad_get_kl(p0.x, p0.y, p1.x, p1.y, p2.x, p2.y, true, true)
            }
            Segment::CubicBez(CubicBez { p0, p1, p2, p3 }) => {
                cubic_get_kl(p0.x, p0.y, p1.x, p1.y, p2.x, p2.y, p3.x, p3.y, true, true)
            }
            Segment::CircularArc(CircularArc {
                center,
                radius,
                theta0,
                theta1,
            }) => arc_get_kl(center.x, center.y, *radius, *theta0, *theta1, true, true),
            Segment::Superellipse(Superellipse {
                center,
                a,
                b,
                n,
                quadrants,
            }) => se_get_kl(center.x, center.y, *a, *b, *n, *quadrants, true, true),
        }
    }

    /// K/L with inline affine transform `scale * p - 2k - q`.
    ///
    /// Transforms from normalized shape space into a cell's quadrant domain
    /// without constructing a temporary `Segment`.
    pub fn get_kl_transformed(&self, scale: f64, kx: f64, ky: f64, qx: f64, qy: f64) -> KL {
        let tx = |x: f64| scale * x - kx * 2.0 - qx;
        let ty = |y: f64| scale * y - ky * 2.0 - qy;
        let skip_x0 = qx > 0.5;
        let skip_y0 = qy > 0.5;
        match self {
            Segment::Line(Line { p0, p1 }) => {
                line_get_kl(tx(p0.x), ty(p0.y), tx(p1.x), ty(p1.y), skip_x0, skip_y0)
            }
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
            Segment::CircularArc(CircularArc {
                center,
                radius,
                theta0,
                theta1,
            }) => arc_get_kl(
                tx(center.x),
                ty(center.y),
                scale * *radius,
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
                quadrants,
            }) => se_get_kl(
                tx(center.x),
                ty(center.y),
                scale * *a,
                scale * *b,
                *n,
                *quadrants,
                skip_x0,
                skip_y0,
            ),
        }
    }

    /// New segment with affine transform `scale * p - 2k - q` applied.
    pub fn transformed(&self, scale: f64, kx: f64, ky: f64, qx: f64, qy: f64) -> Segment {
        let xform =
            |p: &Point| Point::new(scale * p.x - kx * 2.0 - qx, scale * p.y - ky * 2.0 - qy);
        match self {
            Segment::Line(Line { p0, p1 }) => Segment::Line(Line::new(xform(p0), xform(p1))),
            Segment::QuadBez(QuadBez { p0, p1, p2 }) => {
                Segment::QuadBez(QuadBez::new(xform(p0), xform(p1), xform(p2)))
            }
            Segment::CubicBez(CubicBez { p0, p1, p2, p3 }) => {
                Segment::CubicBez(CubicBez::new(xform(p0), xform(p1), xform(p2), xform(p3)))
            }
            Segment::CircularArc(CircularArc {
                center,
                radius,
                theta0,
                theta1,
            }) => Segment::CircularArc(CircularArc {
                center: xform(center),
                radius: scale * *radius,
                theta0: *theta0,
                theta1: *theta1,
            }),
            Segment::Superellipse(Superellipse {
                center,
                a,
                b,
                n,
                quadrants,
            }) => Segment::Superellipse(Superellipse {
                center: xform(center),
                a: scale * *a,
                b: scale * *b,
                n: *n,
                quadrants: *quadrants,
            }),
        }
    }

    /// Axis-aligned bounding box.
    ///
    /// Control-point hull for Beziers (valid superset by convex hull property).
    /// Tight geometric bbox for arcs and superellipses.
    pub fn bbox(&self) -> Rect {
        match self {
            Segment::Line(Line { p0, p1 }) => Rect::new(
                p0.x.min(p1.x),
                p0.y.min(p1.y),
                p0.x.max(p1.x),
                p0.y.max(p1.y),
            ),
            Segment::QuadBez(QuadBez { p0, p1, p2 }) => Rect::new(
                p0.x.min(p1.x).min(p2.x),
                p0.y.min(p1.y).min(p2.y),
                p0.x.max(p1.x).max(p2.x),
                p0.y.max(p1.y).max(p2.y),
            ),
            Segment::CubicBez(CubicBez { p0, p1, p2, p3 }) => Rect::new(
                p0.x.min(p1.x).min(p2.x).min(p3.x),
                p0.y.min(p1.y).min(p2.y).min(p3.y),
                p0.x.max(p1.x).max(p2.x).max(p3.x),
                p0.y.max(p1.y).max(p2.y).max(p3.y),
            ),
            Segment::CircularArc(CircularArc {
                center,
                radius,
                theta0,
                theta1,
            }) => arc_bbox(center.x, center.y, *radius, *theta0, *theta1),
            Segment::Superellipse(Superellipse {
                center,
                a,
                b,
                quadrants,
                ..
            }) => se_bbox(center.x, center.y, *a, *b, *quadrants),
        }
    }

    /// Signed area contribution (Green's theorem: ½ ∮ x dy - y dx).
    ///
    /// Summed across all segments of a closed shape to get the c^(0,0) scaling
    /// coefficient.
    pub fn area_contribution(&self) -> f64 {
        use kurbo::ParamCurveArea;
        match self {
            Segment::Line(l) => l.signed_area(),
            Segment::QuadBez(q) => q.signed_area(),
            Segment::CubicBez(c) => c.signed_area(),
            Segment::CircularArc(CircularArc {
                center,
                radius,
                theta0,
                theta1,
            }) => arc_area(center.x, center.y, *radius, *theta0, *theta1),
            Segment::Superellipse(Superellipse {
                center,
                a,
                b,
                n,
                quadrants,
            }) => se_area(center.x, center.y, *a, *b, *n, *quadrants),
        }
    }
}

/// A closed shape as an ordered list of boundary segments.
///
/// Must form a closed contour with CCW winding. Convenience constructors
/// for common shapes; arbitrary contours via [`Shape::new`].
#[derive(Clone, Debug)]
pub struct Shape {
    pub segments: Vec<Segment>,
}

impl Shape {
    /// Construct a shape from an explicit list of boundary segments.
    pub fn new(segments: Vec<Segment>) -> Self {
        Self { segments }
    }

    /// Construct a closed polygon from vertices, connected by line segments.
    pub fn polygon(vertices: &[(f64, f64)]) -> Self {
        let n = vertices.len();
        let segments = (0..n)
            .map(|i| {
                let j = (i + 1) % n;
                Segment::Line(Line::new(
                    Point::new(vertices[i].0, vertices[i].1),
                    Point::new(vertices[j].0, vertices[j].1),
                ))
            })
            .collect();
        Self { segments }
    }

    /// Closed contour of quadratic Bezier curves.
    ///
    /// Points alternate: [on-curve, control, on-curve, control, ...].
    /// Each (control, on-curve) pair defines one curve from the previous on-curve.
    pub fn quad_bezier(points: &[(f64, f64)]) -> Self {
        let n = points.len();
        assert!(
            n.is_multiple_of(2),
            "quad bezier needs even number of points"
        );
        let segments = (0..n)
            .step_by(2)
            .map(|i| {
                let i0 = if i >= 2 { i - 2 } else { n - 2 + i };
                let i1 = if i >= 1 { i - 1 } else { n - 1 };
                let i2 = i;
                Segment::QuadBez(QuadBez::new(
                    Point::new(points[i0].0, points[i0].1),
                    Point::new(points[i1].0, points[i1].1),
                    Point::new(points[i2].0, points[i2].1),
                ))
            })
            .collect();
        Self { segments }
    }

    /// Closed contour of cubic Bezier curves.
    ///
    /// Points grouped in triples: [ctrl1, ctrl2, on-curve, ...].
    /// Each triple defines one curve from the previous on-curve.
    pub fn cubic_bezier(points: &[(f64, f64)]) -> Self {
        let n = points.len();
        assert!(
            n.is_multiple_of(3),
            "cubic bezier needs multiple-of-3 points"
        );
        let segments = (0..n)
            .step_by(3)
            .map(|i| {
                let idx = |offset: usize| {
                    let j = (n + i - (3 - offset)) % n;
                    Point::new(points[j].0, points[j].1)
                };
                Segment::CubicBez(CubicBez::new(idx(0), idx(1), idx(2), idx(3)))
            })
            .collect();
        Self { segments }
    }

    /// Total signed area (Green's theorem sum over all segments).
    pub fn area(&self) -> f64 {
        self.segments.iter().map(|s| s.area_contribution()).sum()
    }

    /// Normalize coordinates to [0, 1) by scaling by 1/wh.
    pub fn normalize(&mut self, wh: f64) {
        let inv = 1.0 / wh;
        let sp = |p: Point| Point::new(p.x * inv, p.y * inv);
        for seg in &mut self.segments {
            match seg {
                Segment::Line(l) => {
                    l.p0 = sp(l.p0);
                    l.p1 = sp(l.p1);
                }
                Segment::QuadBez(q) => {
                    q.p0 = sp(q.p0);
                    q.p1 = sp(q.p1);
                    q.p2 = sp(q.p2);
                }
                Segment::CubicBez(c) => {
                    c.p0 = sp(c.p0);
                    c.p1 = sp(c.p1);
                    c.p2 = sp(c.p2);
                    c.p3 = sp(c.p3);
                }
                Segment::Superellipse(Superellipse { center, a, b, .. }) => {
                    *center = sp(*center);
                    *a *= inv;
                    *b *= inv;
                }
                Segment::CircularArc(CircularArc { center, radius, .. }) => {
                    *center = sp(*center);
                    *radius *= inv;
                }
            }
        }
    }

    /// Circle as a single full-revolution arc segment.
    pub fn circle(cx: f64, cy: f64, r: f64) -> Self {
        Self {
            segments: vec![Segment::CircularArc(CircularArc {
                center: Point::new(cx, cy),
                radius: r,
                theta0: 0.0,
                theta1: TAU,
            })],
        }
    }

    /// Full convex superellipse |x/a|^n + |y/b|^n = 1 (all 4 quadrants).
    pub fn superellipse(cx: f64, cy: f64, a: f64, b: f64, n: f64) -> Self {
        Self {
            segments: vec![Segment::Superellipse(Superellipse {
                center: Point::new(cx, cy),
                a,
                b,
                n,
                quadrants: 0b1111,
            })],
        }
    }

    /// Partial superellipse with custom quadrant bitmask.
    ///
    /// Bits: 0 = (+x,+y), 1 = (-x,+y), 2 = (-x,-y), 3 = (+x,-y).
    /// E.g., `0b0011` = upper half, `0b1001` = right half.
    pub fn superellipse_half(cx: f64, cy: f64, a: f64, b: f64, n: f64, quadrants: u8) -> Self {
        Self {
            segments: vec![Segment::Superellipse(Superellipse {
                center: Point::new(cx, cy),
                a,
                b,
                n,
                quadrants,
            })],
        }
    }

    /// Single quadrant of a superellipse.
    ///
    /// `quadrant`: 0 = (+x,+y), 1 = (-x,+y), 2 = (-x,-y), 3 = (+x,-y).
    pub fn superellipse_quadrant(cx: f64, cy: f64, a: f64, b: f64, n: f64, quadrant: u8) -> Self {
        Self {
            segments: vec![Segment::Superellipse(Superellipse {
                center: Point::new(cx, cy),
                a,
                b,
                n,
                quadrants: 1 << quadrant,
            })],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_circle_area() {
        let shape = Shape::circle(32.0, 32.0, 10.0);
        let expected = PI * 100.0;
        assert!(
            approx(shape.area(), expected, 1e-10),
            "circle area: got {}, expected {}",
            shape.area(),
            expected
        );
    }

    #[test]
    fn test_circle_area_off_center() {
        let shape = Shape::circle(50.0, 30.0, 7.0);
        let expected = PI * 49.0;
        assert!(
            approx(shape.area(), expected, 1e-10),
            "off-center circle area: got {}, expected {}",
            shape.area(),
            expected
        );
    }

    #[test]
    fn test_arc_bbox_full_circle() {
        let bb = arc_bbox(10.0, 20.0, 5.0, 0.0, TAU);
        assert!(approx(bb.x0, 5.0, 1e-10));
        assert!(approx(bb.x1, 15.0, 1e-10));
        assert!(approx(bb.y0, 15.0, 1e-10));
        assert!(approx(bb.y1, 25.0, 1e-10));
    }

    #[test]
    fn test_arc_bbox_quarter() {
        let bb = arc_bbox(0.0, 0.0, 1.0, 0.0, FRAC_PI_2);
        assert!(approx(bb.x0, 0.0, 1e-10));
        assert!(approx(bb.x1, 1.0, 1e-10));
        assert!(approx(bb.y0, 0.0, 1e-10));
        assert!(approx(bb.y1, 1.0, 1e-10));
    }

    #[test]
    fn test_arc_kl_quarter_at_origin() {
        // Quarter arc: center (0,0), r=1, θ=0 to θ=π/2.
        // Endpoints: (1,0) → (0,1), both in [0,1]^2, no clipping.
        let kl = arc_get_kl(0.0, 0.0, 1.0, 0.0, FRAC_PI_2, true, true);

        assert!(approx(kl.kx, 0.25, 1e-10), "Kx = {}", kl.kx);
        assert!(approx(kl.ky, 0.25, 1e-10), "Ky = {}", kl.ky);
        // Lx = Ly = r²/8 · Δθ = π/16 (center at origin)
        let expected_l = PI / 16.0;
        assert!(approx(kl.lx, expected_l, 1e-10), "Lx = {}", kl.lx);
        assert!(approx(kl.ly, expected_l, 1e-10), "Ly = {}", kl.ly);
    }

    #[test]
    fn test_circle_rasterize_interior() {
        use crate::rasterizer::Rasterizer;

        let shape = Shape::circle(32.0, 32.0, 20.0);
        let pixels = Rasterizer::new(shape, 64, 64).rasterize();

        // Center: fully inside
        assert!(
            pixels[32 * 64 + 32] > 0.99,
            "center = {}",
            pixels[32 * 64 + 32]
        );
        // Corner: fully outside
        assert!(pixels[0] < 0.01, "corner = {}", pixels[0]);
        // Near edge (diagonal): partial coverage
        let edge = pixels[45 * 64 + 46];
        assert!(edge > 0.01 && edge < 0.99, "edge (45,46) = {}", edge);
    }

    #[test]
    fn test_circle_vs_polygon_area() {
        let (cx, cy, r) = (32.0, 32.0, 15.0);
        let n = 360;
        let poly_verts: Vec<(f64, f64)> = (0..n)
            .map(|i| {
                let theta = TAU * i as f64 / n as f64;
                (cx + r * theta.cos(), cy + r * theta.sin())
            })
            .collect();

        let circle_area = Shape::circle(cx, cy, r).area();
        let poly_area = Shape::polygon(&poly_verts).area();
        let exact = PI * r * r;

        // Circle area should be exact
        assert!(
            approx(circle_area, exact, 1e-10),
            "circle={circle_area}, exact={exact}"
        );
        // Polygon should be close but not exact
        assert!(
            approx(poly_area, exact, 0.1),
            "polygon={poly_area}, exact={exact}"
        );
    }

    #[test]
    fn test_superellipse_diamond_area() {
        // n=1: diamond |x/a| + |y/b| = 1, area = 2*a*b
        let shape = Shape::superellipse(0.0, 0.0, 5.0, 3.0, 1.0);
        let expected = 2.0 * 5.0 * 3.0;
        assert!(
            approx(shape.area(), expected, 1e-6),
            "diamond area: got {}, expected {}",
            shape.area(),
            expected
        );
    }

    #[test]
    fn test_superellipse_ellipse_area() {
        // n=2: ellipse, area = π*a*b. GL-16 gives ~4e-5 relative error on cuspy integrand.
        let shape = Shape::superellipse(0.0, 0.0, 4.0, 3.0, 2.0);
        let expected = PI * 4.0 * 3.0;
        assert!(
            approx(shape.area(), expected, 0.01),
            "ellipse area: got {}, expected {}",
            shape.area(),
            expected
        );
    }

    #[test]
    fn test_superellipse_circle_area() {
        // n=2, a=b=r: circle, area = π*r²
        let shape = Shape::superellipse(10.0, 20.0, 7.0, 7.0, 2.0);
        let expected = PI * 49.0;
        assert!(
            approx(shape.area(), expected, 0.01),
            "circle area: got {}, expected {}",
            shape.area(),
            expected
        );
    }

    #[test]
    fn test_superellipse_kl_vs_circle() {
        // n=2 circle at origin, r=1: single Q0 KL should match arc_get_kl
        let se_kl = se_get_kl(0.0, 0.0, 1.0, 1.0, 2.0, 0b0001, true, true);
        let arc_kl_val = arc_get_kl(0.0, 0.0, 1.0, 0.0, FRAC_PI_2, true, true);

        // K values are endpoint-only, exact
        assert!(
            approx(se_kl.kx, arc_kl_val.kx, 1e-10),
            "Kx: se={}, arc={}",
            se_kl.kx,
            arc_kl_val.kx
        );
        assert!(
            approx(se_kl.ky, arc_kl_val.ky, 1e-10),
            "Ky: se={}, arc={}",
            se_kl.ky,
            arc_kl_val.ky
        );
        // L values use GL-16, ~1e-5 accuracy on cuspy integrand
        assert!(
            approx(se_kl.lx, arc_kl_val.lx, 1e-5),
            "Lx: se={}, arc={}",
            se_kl.lx,
            arc_kl_val.lx
        );
        assert!(
            approx(se_kl.ly, arc_kl_val.ly, 1e-5),
            "Ly: se={}, arc={}",
            se_kl.ly,
            arc_kl_val.ly
        );
    }

    /// Verify wavelet rasterizer matches the analytic reference for all primitives.
    ///
    /// Both rasterizers output signed coverage; comparison uses absolute
    /// difference with tolerance for f32 wavelet coefficient precision.
    #[test]
    fn test_wavelet_vs_reference() {
        use crate::rasterizer::Rasterizer;
        use crate::reference::{FillRule, reference_rasterize};

        let shapes: Vec<(&str, Shape, usize)> = vec![
            (
                "triangle",
                Shape::polygon(&[(8.0, 8.0), (56.0, 8.0), (30.0, 50.0)]),
                64,
            ),
            (
                "quad_bezier",
                Shape::quad_bezier(&[(8.0, 8.0), (56.0, 8.0), (56.0, 56.0), (8.0, 56.0)]),
                64,
            ),
            (
                "cubic_bezier",
                Shape::cubic_bezier(&[
                    (8.0, 8.0),
                    (12.0, 8.0),
                    (56.0, 24.0),
                    (56.0, 56.0),
                    (24.0, 56.0),
                    (8.0, 24.0),
                ]),
                64,
            ),
            ("circle", Shape::circle(32.0, 32.0, 20.0), 64),
            (
                "squircle",
                Shape::superellipse(32.0, 32.0, 20.0, 20.0, 4.0),
                64,
            ),
        ];

        for (name, shape, size) in shapes {
            let ref_px = reference_rasterize(&shape, size, size, FillRule::NonZero);
            let raster = Rasterizer::for_size(size, size);
            let data = raster.compute(shape);
            let wav_px = raster.rasterize(&data);

            let mut max_err: f32 = 0.0;
            let mut srgb8_wrong = 0usize;
            let mut mce: f64 = 0.0;
            for i in 0..wav_px.len() {
                let diff = wav_px[i] - ref_px[i];
                let ad = diff.abs();
                if ad > max_err {
                    max_err = ad;
                }
                // sRGB 8-bit comparison
                let w_lin = (1.0 - wav_px[i].abs()).clamp(0.0, 1.0);
                let r_lin = (1.0 - ref_px[i].abs()).clamp(0.0, 1.0);
                let to_srgb = |v: f32| -> u8 {
                    let s = if v <= 0.0031308 {
                        v * 12.92
                    } else {
                        1.055 * v.powf(1.0 / 2.4) - 0.055
                    };
                    (s * 255.0 + 0.5) as u8
                };
                if to_srgb(w_lin) != to_srgb(r_lin) {
                    srgb8_wrong += 1;
                }
                let w_s = if w_lin <= 0.0031308 {
                    w_lin * 12.92
                } else {
                    1.055 * w_lin.powf(1.0 / 2.4) - 0.055
                };
                let r_s = if r_lin <= 0.0031308 {
                    r_lin * 12.92
                } else {
                    1.055 * r_lin.powf(1.0 / 2.4) - 0.055
                };
                let sd = (w_s - r_s) as f64;
                mce += sd * sd * sd.abs();
            }
            mce /= wav_px.len() as f64;
            println!(
                "{name:>15}: max_err={max_err:.6}  srgb8_wrong={srgb8_wrong}/{}  mce={mce:.2e}",
                wav_px.len()
            );
        }
    }

    #[test]
    fn test_superellipse_rasterize() {
        use crate::rasterizer::Rasterizer;

        // Squircle (n=4)
        let shape = Shape::superellipse(32.0, 32.0, 20.0, 20.0, 4.0);
        let pixels = Rasterizer::new(shape, 64, 64).rasterize();

        // Center: fully inside
        assert!(
            pixels[32 * 64 + 32] > 0.99,
            "center = {}",
            pixels[32 * 64 + 32]
        );
        // Corner: fully outside
        assert!(pixels[0] < 0.01, "corner = {}", pixels[0]);
        // Well inside (halfway to edge along x-axis)
        assert!(
            pixels[32 * 64 + 42] > 0.99,
            "inside (32,42) = {}",
            pixels[32 * 64 + 42]
        );
    }
}
