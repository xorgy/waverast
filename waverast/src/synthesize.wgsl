//! Inverse Haar wavelet transform — one thread per pixel.
//!
//! Reads from a dense coefficient buffer and writes per-pixel coverage to
//! an output buffer.

struct Params {
    width: u32,
    height: u32,
    max_j: u32,
    wh: f32,
    area: f32,
}

struct Coeffs {
    c01: f32,
    c10: f32,
    c11: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> all_c: array<Coeffs>;
@group(0) @binding(2) var<storage, read_write> pixels: array<f32>;

// level_offset(j) = (4^j - 1) / 3
fn level_offset(j: u32) -> u32 {
    return ((1u << (2u * j)) - 1u) / 3u;
}

@compute @workgroup_size(256)
fn synthesize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.width * params.height;
    if idx >= total {
        return;
    }

    let row = idx / params.width;
    let col = idx % params.width;
    let py = f32(row) / params.wh;
    let px = f32(col) / params.wh;

    var detail: f32 = 0.0;

    // Walk levels from finest to coarsest to accumulate small values first.
    for (var j: i32 = i32(params.max_j); j >= 0; j = j - 1) {
        let ju = u32(j);
        let scale = f32(1u << ju);
        let tx = scale * px;
        let ty = scale * py;

        let kx = u32(tx);
        let ky = u32(ty);
        let cells = 1u << ju;

        if kx >= cells || ky >= cells {
            continue;
        }

        let fx = tx - f32(kx);
        let fy = ty - f32(ky);

        let sign_x = select(-1.0, 1.0, fx < 0.5);
        let sign_y = select(-1.0, 1.0, fy < 0.5);

        let c = all_c[level_offset(ju) + kx * cells + ky];

        detail += sign_y * c.c01 + sign_x * (c.c10 + sign_y * c.c11);
    }

    pixels[idx] = params.area + detail;
}
