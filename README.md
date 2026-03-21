# waverast

A GPU-accelerated 2D vector graphics rasterizer. It computes exact anti-aliased pixel coverage using Haar wavelet decomposition of the shape's indicator function, based on the method from Manson and Schaefer [1].

This implementation adds stability improvements for f32 GPU execution and supports circular arcs and superellipses beyond the polynomial segments described in the paper.

## Method

The shape's indicator function $\chi_M$ (1 inside, 0 outside) is expanded in the 2D Haar wavelet basis. The Haar basis functions are piecewise constant over a quadtree of cells: one scaling function (constant over the whole cell) and three detail functions (constant $\pm 1$ on each quadrant). A cell's scaling coefficient is its average coverage; the detail coefficients encode how coverage differs across the cell's midpoint boundaries in $x$, $y$, and diagonally.

For an indicator function, the scaling coefficient at the root is just the shape's area, and a cell's detail coefficients are nonzero only if the boundary passes through it. This gives a sparse representation — only $O(\text{boundary length} \times \text{tree depth})$ cells need coefficients, not $O(\text{pixels})$.

### Boundary integrals

Computing coefficients by integrating $\chi_M$ over each cell would require clipping the shape to every cell. Instead, the divergence theorem converts the area integral into a line integral over the boundary:

$$c^e_{j,k} = \oint_{\partial M} F^e_{j,k} \cdot n \; d\sigma$$

where $F^e$ is a vector field satisfying $\nabla \cdot F^e = \psi^e$ (the wavelet basis function). Manson and Schaefer choose $F^e$ based on $\bar\Psi$, the piecewise-linear antiderivative of the Haar wavelet (the "tent function"), giving $F^e$ compact support so only segments intersecting the cell contribute.

To evaluate the integral, the paper splits each cell into four quadrants and computes per-quadrant $K$ (endpoint) and $L$ (path integral) terms, then combines them via equation 8. For polynomial segments (lines, quadratics, cubics), the $L$ integrals have closed-form solutions expressed in terms of control-point coordinates (Appendix A and B of [1]).

Each boundary segment can be processed independently for each cell it overlaps — no global state, no ordering. On the GPU, one thread handles one segment across all levels.

### Synthesis

Pixel coverage is recovered by the inverse wavelet transform:

$$g(p) = \text{area} + \sum_j \mathrm{sign}_y(p) \cdot c^{(0,1)}_{j,k(p)} + \mathrm{sign}_x(p) \cdot \left( c^{(1,0)}_{j,k(p)} + \mathrm{sign}_y(p) \cdot c^{(1,1)}_{j,k(p)} \right)$$

For each pixel, this walks one cell per quadtree level — $O(\log N)$ work per pixel.

## Extensions

### Problem: midpoint cancellation in equation 8

The paper's four-quadrant $K$/$L$ decomposition (eq. 8) combines contributions from the four quadrants of each cell. When a segment lies exactly on a cell's midpoint boundary ($x = 0.5$ or $y = 0.5$ in cell coordinates), two adjacent quadrants produce $K$/$L$ values that are equal and opposite — they should cancel to zero, but in finite precision they leave a residual. This creates wrong coefficients at every level where the segment aligns with a midpoint, producing visible horizontal or vertical line artifacts.

**Fix: direct tent-function integrals.** The paper's $F^e$ fields are already defined in terms of $\bar\Psi$, but the evaluation goes through the four-quadrant $K$/$L$ split. We skip the quadrant decomposition and integrate $\bar\Psi$ directly along the boundary:

$$c^{(0,1)} = -\int \bar\Psi(v)\, du, \qquad c^{(1,0)} = \int \bar\Psi(u)\, dv$$

where $(u, v)$ are cell-local coordinates in $[0, 1]^2$. Since $\bar\Psi$ is continuous at the midpoint ($\bar\Psi(t) = t$ for $t \in [0, \tfrac{1}{2}]$, $1-t$ for $t \in [\tfrac{1}{2}, 1]$), there is no cancellation. The integral is split at $v = 0.5$ (where $\bar\Psi$ has a slope discontinuity) and evaluated in closed form per sub-curve. For cubics and arcs, $c^{(1,1)}$ also uses the tent integral (as $\sum \mathrm{sign}_v \cdot \text{tent}_u$) rather than equation 8.

### Problem: integration by parts on GPU

The paper's $L$ integrals provide $\int v \cdot u'\, dt$ (equivalently $\int v\, du$) per sub-curve. To get $\int u\, dv$, the standard approach is integration by parts: $\int u\, dv = uv\big|_a^b - \int v\, du$. When a sub-curve nearly fills a cell half, the boundary term $uv\big|_a^b$ and the integral $\int v\, du$ are nearly equal, and the subtraction loses most of the significant bits. On GPU hardware with fused multiply-add, the rounding in the boundary term differs from the rounding in the integral, making the cancellation worse than a naive f32 analysis predicts.

**Fix: independent Bernstein products.** The paper gives closed-form $L$ expressions for each integral direction (Appendix B). We compute both $\int v \cdot u'\, dt$ and $\int u \cdot v'\, dt$ as independent Bernstein product formulas — the same type of closed form as the paper's $L$ terms, but expressed so neither integral is derived from the other via subtraction. For a cubic:

$$\int_0^1 v \cdot u'\, dt = \frac{1}{20}\left[(10v_0{+}6v_1{+}3v_2{+}v_3)\Delta u_0 + (4v_0{+}6v_1{+}6v_2{+}4v_3)\Delta u_1 + (v_0{+}3v_1{+}6v_2{+}10v_3)\Delta u_2\right]$$

$$\int_0^1 u \cdot v'\, dt = \frac{1}{20}\left[(10u_0{+}6u_1{+}3u_2{+}u_3)\Delta v_0 + (4u_0{+}6u_1{+}6u_2{+}4u_3)\Delta v_1 + (u_0{+}3u_1{+}6u_2{+}10u_3)\Delta v_2\right]$$

### Problem: missed roots in the cubic solver

The paper clips Bézier curves to quadtree cells using recursive subdivision [SP86]. We use analytic root-finding instead (solving $B(t) = \text{val}$ for each cell boundary), which is faster but can miss roots when the cubic is nearly tangent to a boundary. In the depressed form, this happens when $p \approx 0$ (i.e., $3ac \approx b^2$): the discriminant is near zero and the solver may return the wrong number of roots. A missed root means a sub-interval spans two cells instead of being split at the boundary, producing a wrong coefficient visible as a horizontal line error across many pixels.

**Fix: bisection verification.** After the analytic solver returns roots, we sort them and check each adjacent pair for sign changes at all boundary values. If a pair straddles a boundary that has no root between them, we bisect to find it. This adds a few curve evaluations per cell but catches every case the analytic solver misses.

### Circular arcs and superellipses

The paper handles polygons and Bézier curves (quadratic and cubic). We add circular arcs and convex superellipses.

**Arcs** use closed-form trigonometric antiderivatives for the $K$ and $L$ terms. For the tent integral, a near-full-coverage sub-arc produces terms that nearly cancel. We use conditioned forms that group terms so the leading quantities are small:

$$\text{tent}_v \;(v \ge \tfrac{1}{2}):\quad r \cdot \Delta\cos \cdot (1 - c_y) + r^2\!\left(\frac{\Delta\theta}{2} - \frac{\Delta\sin 2\theta}{4}\right)$$

Trigonometric differences use product-to-sum identities ($\sin\beta - \sin\alpha = 2\cos\!\tfrac{\alpha+\beta}{2}\sin\!\tfrac{\beta-\alpha}{2}$) throughout.

**Superellipses** $|x/a|^n + |y/b|^n = 1$ are parameterized per-quadrant so $dx/du$ is constant. This makes $\int y\, dx$ smooth and suitable for Gauss-Legendre quadrature, while $\int x\, dy$ is obtained via integration by parts to avoid the singular $dy/du$ at $u = 1$. The integration interval is split near the cusp ($u = 0.75$, $0.9375$) for quadrature accuracy.

### Cubic solver

The paper does not describe a specific polynomial solver. Our f32 cubic solver follows Blinn [2]: discriminant from Hessian invariants ($s_1 = 3ac - b^2$, $s_2 = 9ad - bc$, $s_3 = 3bd - c^2$) computed without dividing by $a$; atan2-based trigonometric method for three real roots; Cardano with safe addition for one real root; double roots handled by the trigonometric path ($\Delta \le 0$, not $< 0$). Each root is Newton-refined in Bernstein form.

## Usage

```bash
cargo build --release
```

### Render to PNG

```bash
cargo run --release --bin render-svg -- <path-file> <width> <height> [output.png] [flags]
```

Flags: `--gpu-coeff`, `--gpu-synth`, `--reference`, `--compare`, `--gpu-compare`, `--gpu-bench`, `--sparse`, `--evenodd`, `--16bit`

### Viewer

```bash
cargo run --release --bin viewer -- <path-file> <width> <height> [--async-coeff] [--sparse] [--f32] [--evenodd]
```

Mouse wheel zooms toward cursor. Drag to pan.

## References

[1] J. Manson and S. Schaefer. "Wavelet Rasterization." *Computer Graphics Forum* (Eurographics 2011) 30(2):395–404.

[2] J. Blinn. "How to Solve a Cubic Equation." *IEEE Computer Graphics and Applications*, Parts 1–5, 2006–2007.
