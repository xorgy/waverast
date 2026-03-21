use crate::contour::{
    Bbox, CircularArc, CubicBez, KL, Line, Point, QuadBez, Rect, Segment, Shape, Superellipse,
    arc_tent_c01_c10, line_get_kl, line_tent_c01_c10, quad_get_kl, quad_tent_c01_c10, se_get_kl,
    se_tent_c01_c10,
};

/// Haar wavelet detail coefficients for one quadtree cell (M&S equation 4).
///
/// Contains c^(0,1), c^(1,0), c^(1,1). Computed in f64, stored as f32 (sufficient
/// for pixel synthesis at 8–16 bit quantization). `#[repr(C)]` for GPU upload.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
#[cfg_attr(feature = "gpu", derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct Coeffs {
    pub c01: f32,
    pub c10: f32,
    pub c11: f32,
}

/// Half-precision coefficient storage for GPU upload.
///
/// Same layout as WGSL `vec4<f16>`: three f16 coefficients plus one padding
/// half, totaling 8 bytes per cell.
#[cfg(feature = "gpu")]
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct CoeffsF16 {
    pub c01: f16,
    pub c10: f16,
    pub c11: f16,
    pub _pad: f16,
}

#[cfg(feature = "gpu")]
unsafe impl bytemuck::Pod for CoeffsF16 {}
#[cfg(feature = "gpu")]
unsafe impl bytemuck::Zeroable for CoeffsF16 {}

/// Output of the coefficient computation stage.
///
/// Contains the area (c^(0,0) scaling coefficient) and the dense coefficient
/// buffer. Use with [`Rasterizer::rasterize`] for CPU synthesis, or upload to
/// GPU via `GpuSynthesizer`.
pub struct CoeffData<C> {
    pub area: f32,
    pub coeffs: Vec<C>,
}

/// Wavelet rasterizer for 2D shapes (M&S "Wavelet Rasterization", 2006).
///
/// Computes Haar wavelet coefficients of a shape's indicator function via
/// boundary integrals (M&S equations 5–8), then synthesizes per-pixel coverage
/// values by inverse wavelet transform. The coefficient array is dense,
/// indexed by quadtree level and cell position.
pub struct Rasterizer {
    w: usize,
    h: usize,
    /// Maximum wavelet decomposition level.
    ///
    /// An image of width/height N uses levels 0..=max_j where 2^(max_j+1) ≥ N.
    max_j: u32,
    /// Grid dimension as float: 2^(max_j+1).
    ///
    /// Pixel coordinates are divided by this to map into [0, 1).
    wh: f32,
}

/// Base index into the dense coefficient array for level `j`.
///
/// Equals Σ_{i=0}^{j−1} 4^i = (4^j − 1) / 3.
pub fn level_offset(j: u32) -> usize {
    ((1usize << (2 * j)) - 1) / 3
}

impl Rasterizer {
    /// Create a rasterizer, compute f32 coefficients, and synthesize in one call.
    #[allow(clippy::new_ret_no_self)]
    pub fn new(shape: Shape, w: usize, h: usize) -> RasterizerWithCoeffs {
        let r = Self::for_size(w, h);
        let data = r.compute(shape);
        RasterizerWithCoeffs { r, data }
    }

    /// Create a rasterizer for the given output dimensions.
    pub fn for_size(w: usize, h: usize) -> Self {
        let max_j = (w.max(h) as f64).log2().ceil() as u32;
        let wh = (1u64 << (max_j + 1)) as f32;
        Self { w, h, max_j, wh }
    }

    /// Create a rasterizer with extra superresolution levels beyond pixel resolution.
    pub fn for_size_with_extra_levels(w: usize, h: usize, extra: u32) -> Self {
        let max_j = ((w.max(h) as f64).log2().ceil() as u32).saturating_sub(1) + extra;
        let wh = (1u64 << (max_j + 1)) as f32;
        Self { w, h, max_j, wh }
    }

    /// Maximum wavelet decomposition level.
    pub fn max_level(&self) -> u32 {
        self.max_j
    }

    /// Grid dimension (2^(max_j+1)), used to normalize pixel coordinates.
    pub fn grid_size(&self) -> f32 {
        self.wh
    }

    /// Output width in pixels.
    pub fn width(&self) -> usize {
        self.w
    }

    /// Output height in pixels.
    pub fn height(&self) -> usize {
        self.h
    }

    /// Compute f32 wavelet coefficients (M&S equations 5–8).
    ///
    /// Two-pass approach, all accumulation in f64:
    ///   1. c11 via equation-8 K/L decomposition (lines, quads, SEs)
    ///   2. c01/c10 via tent for all types; c11 via tent for cubics and arcs
    pub fn compute(&self, mut shape: Shape) -> CoeffData<Coeffs> {
        let wh = self.wh as f64;
        shape.normalize(wh);
        let area = shape.area() as f32;
        let batches = SegmentBatches::from_shape(&shape);
        let total = level_offset(self.max_j + 1);

        let mut coeffs = vec![
            Coeffs {
                c01: 0.0,
                c10: 0.0,
                c11: 0.0
            };
            total
        ];

        // c11 accumulator (f64). Equation 8: c11 = L00_x - L01_x + K10_x - L10_x - K11_x + L11_x
        let mut c11_acc = vec![0.0f64; total];

        // Pass 1: c11 via equation-8 (lines, quads, SEs; not cubics/arcs)
        self.run_coefficient_loop(&batches, |base, cells, bb, get_kl| {
            for_overlapping_cells(cells, bb, get_kl, |idx, [q00, q01, q10, q11]| {
                c11_acc[base + idx] += q00.lx - q01.lx + q10.kx - q10.lx - q11.kx + q11.lx;
            });
        });

        // Pass 2: c01/c10 via tent (all types); c11 via tent (cubics, arcs)
        self.run_tent_coefficient_loop(&batches, &mut coeffs, &mut c11_acc);

        for i in 0..total {
            coeffs[i].c11 = c11_acc[i] as f32;
        }

        CoeffData { area, coeffs }
    }

    /// Compute f16 wavelet coefficients.
    ///
    /// Accumulates in f32 (via [`compute`]) then converts to f16 in a single
    /// pass. f16 has only 11 bits of mantissa — not enough for incremental
    /// accumulation across many segments.
    #[cfg(feature = "gpu")]
    pub fn compute_f16(&self, shape: Shape) -> CoeffData<CoeffsF16> {
        let f32_data = self.compute(shape);
        let coeffs = f32_data
            .coeffs
            .iter()
            .map(|c| CoeffsF16 {
                c01: c.c01 as f16,
                c10: c.c10 as f16,
                c11: c.c11 as f16,
                _pad: 0.0,
            })
            .collect();
        CoeffData {
            area: f32_data.area,
            coeffs,
        }
    }

    /// Convert pre-computed f32 coefficients to f16.
    ///
    /// For GPU-computed coefficients (f32 atomic accumulation) when the
    /// synthesis shader uses f16.
    #[cfg(feature = "gpu")]
    pub fn compute_f16_from_f32(&self, f32_data: &CoeffData<Coeffs>) -> CoeffData<CoeffsF16> {
        let coeffs = f32_data
            .coeffs
            .iter()
            .map(|c| CoeffsF16 {
                c01: c.c01 as f16,
                c10: c.c10 as f16,
                c11: c.c11 as f16,
                _pad: 0.0,
            })
            .collect();
        CoeffData {
            area: f32_data.area,
            coeffs,
        }
    }

    /// The equation-8 coefficient loop for c11.
    ///
    /// For each wavelet level and each cell overlapping the segment bbox, computes
    /// K/L in all four quadrants (M&S equation 8) and combines them into c11.
    fn run_coefficient_loop(
        &self,
        batches: &SegmentBatches,
        mut accum: impl FnMut(usize, usize, &Bbox, &dyn Fn(f64, f64, f64, f64) -> KL),
    ) {
        for j in 0..=self.max_j {
            let cells = 1usize << j;
            let scale = (1u64 << (j + 1)) as f64;
            let base = level_offset(j);

            // skip_x1/skip_y1: skip boundary segments at x=1/y=1 only when
            // that boundary is the OUTER cell edge (qx=1/qy=1), not when it's
            // the inner midpoint (qx=0/qy=0). This ensures correct cancellation
            // of midpoint boundary contributions between adjacent quadrants.
            for (pts, bb) in &batches.lines {
                accum(base, cells, bb, &|kxf, kyf, qx, qy| {
                    let tx = |x: f64| scale * x - kxf * 2.0 - qx;
                    let ty = |y: f64| scale * y - kyf * 2.0 - qy;
                    line_get_kl(
                        tx(pts[0].x),
                        ty(pts[0].y),
                        tx(pts[1].x),
                        ty(pts[1].y),
                        qx > 0.5,
                        qy > 0.5,
                    )
                });
            }

            for (pts, bb) in &batches.quads {
                accum(base, cells, bb, &|kxf, kyf, qx, qy| {
                    let tx = |x: f64| scale * x - kxf * 2.0 - qx;
                    let ty = |y: f64| scale * y - kyf * 2.0 - qy;
                    quad_get_kl(
                        tx(pts[0].x),
                        ty(pts[0].y),
                        tx(pts[1].x),
                        ty(pts[1].y),
                        tx(pts[2].x),
                        ty(pts[2].y),
                        qx > 0.5,
                        qy > 0.5,
                    )
                });
            }

            // Cubics, arcs: c11 computed via tent in pass 2

            for &(center, a, sb, n, quadrants, ref bb) in &batches.superellipses {
                accum(base, cells, bb, &|kxf, kyf, qx, qy| {
                    se_get_kl(
                        scale * center.x - kxf * 2.0 - qx,
                        scale * center.y - kyf * 2.0 - qy,
                        scale * a,
                        scale * sb,
                        n,
                        quadrants,
                        qx > 0.5,
                        qy > 0.5,
                    )
                });
            }
        }
    }

    /// Tent-function integral loop (equation 7 with Ψ̄).
    ///
    /// Computes c01/c10 for all segment types. For cubics and arcs, also
    /// computes c11 via tent (sign_v · tent_u), correcting the equation-8
    /// value already in c11_acc.
    fn run_tent_coefficient_loop(
        &self,
        batches: &SegmentBatches,
        coeffs: &mut [Coeffs],
        c11_acc: &mut [f64],
    ) {
        let total = coeffs.len();
        let mut acc = vec![(0.0f64, 0.0f64); total];

        for j in 0..=self.max_j {
            let cells = 1usize << j;
            let cellsf = cells as f64;
            let base = level_offset(j);

            for (pts, bb) in &batches.lines {
                for_overlapping_cells_tent(
                    cells,
                    bb,
                    &|kxf, kyf| {
                        line_tent_c01_c10(
                            cellsf * pts[0].x - kxf,
                            cellsf * pts[0].y - kyf,
                            cellsf * pts[1].x - kxf,
                            cellsf * pts[1].y - kyf,
                        )
                    },
                    |idx, c01, c10| {
                        let a = &mut acc[base + idx];
                        a.0 += c01;
                        a.1 += c10;
                    },
                );
            }

            for (pts, bb) in &batches.quads {
                for_overlapping_cells_tent(
                    cells,
                    bb,
                    &|kxf, kyf| {
                        let tu = |x: f64| cellsf * x - kxf;
                        let tv = |y: f64| cellsf * y - kyf;
                        quad_tent_c01_c10(
                            tu(pts[0].x),
                            tv(pts[0].y),
                            tu(pts[1].x),
                            tv(pts[1].y),
                            tu(pts[2].x),
                            tv(pts[2].y),
                        )
                    },
                    |idx, c01, c10| {
                        let a = &mut acc[base + idx];
                        a.0 += c01;
                        a.1 += c10;
                    },
                );
            }

            for (pts, bb) in &batches.cubics {
                crate::contour::cubic_tent_level(pts, bb, cellsf, cells, base, &mut acc, c11_acc);
            }

            for &(center, r, t0, t1, ref bb) in &batches.arcs {
                let kx_lo = (bb.x0 * cellsf).floor().max(0.0) as usize;
                let kx_hi = ((bb.x1 * cellsf).ceil() as usize).min(cells);
                let ky_lo = (bb.y0 * cellsf).floor().max(0.0) as usize;
                let ky_hi = ((bb.y1 * cellsf).ceil() as usize).min(cells);
                for kx in kx_lo..kx_hi {
                    let kxf = kx as f64;
                    for ky in ky_lo..ky_hi {
                        let kyf = ky as f64;
                        let (c01, c10, c11_tent) = arc_tent_c01_c10(
                            cellsf * center.x - kxf,
                            cellsf * center.y - kyf,
                            cellsf * r,
                            t0,
                            t1,
                        );
                        let idx = base + kx * cells + ky;
                        acc[idx].0 += c01;
                        acc[idx].1 += c10;
                        c11_acc[idx] += c11_tent;
                    }
                }
            }

            for &(center, a, sb, n, quadrants, ref bb) in &batches.superellipses {
                for_overlapping_cells_tent(
                    cells,
                    bb,
                    &|kxf, kyf| {
                        se_tent_c01_c10(
                            cellsf * center.x - kxf,
                            cellsf * center.y - kyf,
                            cellsf * a,
                            cellsf * sb,
                            n,
                            quadrants,
                        )
                    },
                    |idx, c01, c10| {
                        let a = &mut acc[base + idx];
                        a.0 += c01;
                        a.1 += c10;
                    },
                );
            }
        }

        for i in 0..total {
            coeffs[i].c01 = acc[i].0 as f32;
            coeffs[i].c10 = acc[i].1 as f32;
        }
    }

    /// Synthesize pixel coverage from f32 coefficients.
    ///
    /// Returns a row-major `Vec<f32>` of size `h × w`, each value in [0, 1].
    pub fn rasterize(&self, data: &CoeffData<Coeffs>) -> Vec<f32> {
        let mut pixels = vec![0.0f32; self.h * self.w];

        for row in 0..self.h {
            let py = row as f32 / self.wh;
            let row_off = row * self.w;
            for col in 0..self.w {
                let px = col as f32 / self.wh;
                pixels[row_off + col] = synthesize(&data.coeffs, self.max_j, data.area, px, py);
            }
        }

        pixels
    }
}

/// Convenience wrapper for the compute-then-synthesize pattern.
///
/// Returned by [`Rasterizer::new`].
pub struct RasterizerWithCoeffs {
    r: Rasterizer,
    data: CoeffData<Coeffs>,
}

impl RasterizerWithCoeffs {
    pub fn rasterize(&self) -> Vec<f32> {
        self.r.rasterize(&self.data)
    }
}

/// Inverse Haar wavelet transform for a single pixel (M&S equation 3).
///
/// Evaluates g(p) = area + Σ_j Σ_e sign_e(p) · c^e_{j,k(p)} by walking one
/// cell per level (O(log N) per pixel). Accumulates detail from finest level
/// first to avoid adding small corrections to a large area value.
fn synthesize(all_c: &[Coeffs], max_j: u32, area: f32, px: f32, py: f32) -> f32 {
    let mut detail: f32 = 0.0;

    for j in (0..=max_j).rev() {
        let scale = (1u32 << j) as f32;
        let tx = scale * px;
        let ty = scale * py;

        let kx = tx as usize;
        let ky = ty as usize;
        let cells = 1usize << j;

        if kx >= cells || ky >= cells {
            continue;
        }

        let fx = tx - kx as f32;
        let fy = ty - ky as f32;

        let sign_x = 1.0 - 2.0 * (fx >= 0.5) as u32 as f32;
        let sign_y = 1.0 - 2.0 * (fy >= 0.5) as u32 as f32;

        let c = &all_c[level_offset(j) + kx * cells + ky];

        detail += sign_y * c.c01 + sign_x * (c.c10 + sign_y * c.c11);
    }

    area + detail
}

// ---- Segment batches and accumulation ----

/// Segments classified by type for cache-friendly iteration.
///
/// Each entry pairs geometric data with its precomputed bounding box.
struct SegmentBatches {
    lines: Vec<([Point; 2], Rect)>,
    quads: Vec<([Point; 3], Rect)>,
    cubics: Vec<([Point; 4], Rect)>,
    arcs: Vec<(Point, f64, f64, f64, Rect)>,
    superellipses: Vec<(Point, f64, f64, f64, u8, Rect)>,
}

impl SegmentBatches {
    fn from_shape(shape: &Shape) -> Self {
        let mut b = Self {
            lines: Vec::new(),
            quads: Vec::new(),
            cubics: Vec::new(),
            arcs: Vec::new(),
            superellipses: Vec::new(),
        };
        for seg in &shape.segments {
            let bb = seg.bbox();
            match seg {
                Segment::Line(Line { p0, p1 }) => b.lines.push(([*p0, *p1], bb)),
                Segment::QuadBez(QuadBez { p0, p1, p2 }) => b.quads.push(([*p0, *p1, *p2], bb)),
                Segment::CubicBez(CubicBez { p0, p1, p2, p3 }) => {
                    b.cubics.push(([*p0, *p1, *p2, *p3], bb))
                }
                Segment::CircularArc(CircularArc {
                    center,
                    radius,
                    theta0,
                    theta1,
                }) => b.arcs.push((*center, *radius, *theta0, *theta1, bb)),
                Segment::Superellipse(Superellipse {
                    center,
                    a,
                    b: sb,
                    n,
                    quadrants,
                }) => b.superellipses.push((*center, *a, *sb, *n, *quadrants, bb)),
            }
        }
        b
    }
}

const QUADRANTS: [(f64, f64); 4] = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)];

/// Cell iterator for equation-8 K/L computation.
///
/// For each cell overlapping `bb`, evaluates K/L in all four quadrants and
/// calls `combine` with the resulting `[KL; 4]`.
#[inline]
fn for_overlapping_cells(
    cells: usize,
    bb: &Bbox,
    get_kl: &dyn Fn(f64, f64, f64, f64) -> KL,
    mut combine: impl FnMut(usize, [KL; 4]),
) {
    let cellsf = cells as f64;
    let kx_lo = (bb.x0 * cellsf).floor().max(0.0) as usize;
    let kx_hi = ((bb.x1 * cellsf).ceil() as usize).min(cells);
    let ky_lo = (bb.y0 * cellsf).floor().max(0.0) as usize;
    let ky_hi = ((bb.y1 * cellsf).ceil() as usize).min(cells);

    for kx in kx_lo..kx_hi {
        let kxf = kx as f64;
        for ky in ky_lo..ky_hi {
            let kyf = ky as f64;
            let mut kls = [KL::zero(); 4];
            for (qi, &(qx, qy)) in QUADRANTS.iter().enumerate() {
                kls[qi] = get_kl(kxf, kyf, qx, qy);
            }
            combine(kx * cells + ky, kls);
        }
    }
}

/// Cell iterator for tent-function integral computation.
///
/// For each cell overlapping `bb`, calls `combine` with the (c01, c10) tent values.
#[inline]
fn for_overlapping_cells_tent(
    cells: usize,
    bb: &Bbox,
    get_tent: &dyn Fn(f64, f64) -> (f64, f64),
    mut combine: impl FnMut(usize, f64, f64),
) {
    let cellsf = cells as f64;
    let kx_lo = (bb.x0 * cellsf).floor().max(0.0) as usize;
    let kx_hi = ((bb.x1 * cellsf).ceil() as usize).min(cells);
    let ky_lo = (bb.y0 * cellsf).floor().max(0.0) as usize;
    let ky_hi = ((bb.y1 * cellsf).ceil() as usize).min(cells);

    for kx in kx_lo..kx_hi {
        let kxf = kx as f64;
        for ky in ky_lo..ky_hi {
            let kyf = ky as f64;
            let (c01, c10) = get_tent(kxf, kyf);
            combine(kx * cells + ky, c01, c10);
        }
    }
}
