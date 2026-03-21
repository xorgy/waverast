//! Real root solvers for polynomials up to degree 3.
//!
//! All solvers return `(count, roots)` where only `roots[0..count]` are valid.
//! Roots are returned in fixed-size arrays to avoid heap allocation in the
//! inner loops of the clipping routines that call these.
//!
//! The quadratic and cubic solvers follow Blinn, "How to Solve a Cubic Equation"
//! (Parts 1–5, IEEE CG&A 2006–2007):
//!   - Safe addition via σ(b) to avoid cancellation (Blinn P5 §3)
//!   - Vieta's formula for the second quadratic root
//!   - Depressed cubic via t = x - b/(3a) (Blinn P1 §2)
//!   - Trigonometric method with atan2 for three real roots (Blinn P4/P5)
//!   - Cardano with safe addition for one real root (Blinn P5 Eq 26)

const EPS: f64 = 9.1e-13;

/// Linear solver for `ax + b = 0`.
pub fn solve_linear(a: f64, b: f64) -> (usize, [f64; 1]) {
    if a.abs() < EPS {
        return (0, [0.0]);
    }
    (1, [-b / a])
}

/// Quadratic solver for `ax² + bx + c = 0`.
///
/// Uses safe addition (Blinn P5 §3): the first root uses -b - σ(b)√Δ to
/// pick the sum with larger magnitude, then gets the other via Vieta's
/// formula r₂ = c/(a·r₁).
pub fn solve_quadratic(a: f64, b: f64, c: f64) -> (usize, [f64; 2]) {
    if a.abs() < EPS {
        let (n, r) = solve_linear(b, c);
        return (n, [r[0], 0.0]);
    }

    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return (0, [0.0; 2]);
    }

    let sqrt_disc = disc.sqrt();
    // Safe addition: -b and ∓√disc — pick the combination with larger magnitude
    let sigma_b = if b < 0.0 { -1.0 } else { 1.0 };
    let r1 = (-b - sigma_b * sqrt_disc) / (2.0 * a);
    let r2 = c / (a * r1);
    (2, [r1, r2])
}

/// Cubic solver for `ax³ + bx² + cx + d = 0`.
///
/// Depresses to t³ + a₂t + b₂ = 0 via t = x - b/(3a) (Blinn P1 §2).
/// Three real roots: atan2 trig method (Blinn P4). One real root: Cardano
/// with safe addition (Blinn P5 §3, Eq 26). Double root (disc = 0): cbrt.
pub fn solve_cubic(a: f64, b: f64, c: f64, d: f64) -> (usize, [f64; 3]) {
    if a.abs() < EPS {
        let (n, r) = solve_quadratic(b, c, d);
        return (n, [r[0], r[1], 0.0]);
    }

    // Normalize: t^3 + pt^2 + qt + r = 0
    let inv_a = 1.0 / a;
    let p = b * inv_a;
    let q = c * inv_a;
    let r = d * inv_a;

    // Depress: substitute t = x - p/3
    // x^3 + a2·x + b2 = 0
    let a2 = q - p * p / 3.0;
    let b2 = (2.0 * p * p * p - 9.0 * p * q + 27.0 * r) / 27.0;

    let shift = -p / 3.0;

    if a2.abs() < EPS {
        let x = -b2.cbrt();
        return (1, [x + shift, 0.0, 0.0]);
    }

    let disc = b2 * b2 / 4.0 + a2 * a2 * a2 / 27.0;

    if disc > EPS {
        // One real root — Cardano with safe addition (Blinn P5 §3, Eq 26).
        let sqrt_disc = disc.sqrt();
        let sigma_b2 = if b2 < 0.0 { -1.0 } else { 1.0 };
        let t0 = -sigma_b2 * sqrt_disc;
        let t1 = -b2 / 2.0 + t0;
        let pv = t1.cbrt();
        let qv = if t1 == t0 * 2.0 {
            -pv
        } else {
            -a2 / (3.0 * pv)
        };
        (1, [pv + qv + shift, 0.0, 0.0])
    } else if disc < -EPS {
        // Three real roots — atan2 trig method (Blinn P4/P5).
        let m = (-a2 / 3.0).sqrt();
        let theta = f64::atan2((-disc).sqrt() * 2.0, -b2) / 3.0;
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let two_m = 2.0 * m;
        let x0 = two_m * cos_t + shift;
        let x1 = two_m * (-0.5 * cos_t + 0.866_025_403_784_438_6 * sin_t) + shift;
        let x2 = two_m * (-0.5 * cos_t - 0.866_025_403_784_438_6 * sin_t) + shift;
        (3, [x0, x1, x2])
    } else {
        // Double root
        let u = if b2 > 0.0 {
            -(b2 / 2.0).cbrt()
        } else {
            (-b2 / 2.0).cbrt()
        };
        (2, [2.0 * u + shift, -u + shift, 0.0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    fn verify_roots_linear(a: f64, b: f64) {
        let (n, roots) = solve_linear(a, b);
        for (i, &r) in roots[..n].iter().enumerate() {
            assert!(
                approx_eq(a * r + b, 0.0),
                "linear root {i} = {r} doesn't satisfy {a}x + {b} = 0 (got {})",
                a * r + b
            );
        }
    }

    fn verify_roots_quadratic(a: f64, b: f64, c: f64) {
        let (n, roots) = solve_quadratic(a, b, c);
        for (i, &x) in roots[..n].iter().enumerate() {
            let val = a * x * x + b * x + c;
            assert!(
                approx_eq(val, 0.0),
                "quadratic root {i} = {x} doesn't satisfy (got {val})"
            );
        }
    }

    fn verify_roots_cubic(a: f64, b: f64, c: f64, d: f64) {
        let (n, roots) = solve_cubic(a, b, c, d);
        for (i, &x) in roots[..n].iter().enumerate() {
            let val = a * x * x * x + b * x * x + c * x + d;
            assert!(
                approx_eq(val, 0.0),
                "cubic root {i} = {x} doesn't satisfy (got {val})"
            );
        }
    }

    #[test]
    fn test_linear() {
        verify_roots_linear(2.0, -4.0);
        verify_roots_linear(0.0, 5.0); // no solution
        verify_roots_linear(1.0, 0.0); // x = 0
    }

    #[test]
    fn test_quadratic() {
        verify_roots_quadratic(1.0, -3.0, 2.0); // x=1, x=2
        verify_roots_quadratic(1.0, 0.0, 1.0); // no real roots
        verify_roots_quadratic(1.0, -2.0, 1.0); // double root x=1
    }

    #[test]
    fn test_cubic_known() {
        // 2x^3 - 4x^2 - 22x + 24 = 0 => roots: 4, -3, 1
        let (n, _) = solve_cubic(2.0, -4.0, -22.0, 24.0);
        assert_eq!(n, 3);
        verify_roots_cubic(2.0, -4.0, -22.0, 24.0);

        // 3x^3 - 10x^2 + 14x + 27 = 0 => one real root: -1
        let (n, _) = solve_cubic(3.0, -10.0, 14.0, 27.0);
        assert!(n >= 1);
        verify_roots_cubic(3.0, -10.0, 14.0, 27.0);

        // x^3 + 6x^2 + 12x + 8 = 0 => triple root: -2
        verify_roots_cubic(1.0, 6.0, 12.0, 8.0);
    }

    #[test]
    fn test_cubic_degenerate() {
        // Degenerates to quadratic: 0x^3 + x^2 - 5x + 6 = 0
        verify_roots_cubic(0.0, 1.0, -5.0, 6.0);
        // Degenerates to linear
        verify_roots_cubic(0.0, 0.0, 3.0, -9.0);
    }
}
