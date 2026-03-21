//! Inverse Haar wavelet transform — fp16 variant.
//!
//! Coefficients are stored as packed f16×3 (6 bytes per cell, padded to 8 with
//! a u32 + u16 layout: two f16 in the first u32, one f16 in a second u16 padded
//! to u32). We use vec2<f16> loads and f16 arithmetic throughout.

enable f16;

struct Params {
    width: u32,
    height: u32,
    max_j: u32,
    wh: f32,
    area: f32,
}

// Coefficients packed as [c01: f16, c10: f16, c11: f16, _pad: f16] = 8 bytes = 2 × u32.
// We read as array<vec4<f16>> where each element holds (c01, c10, c11, pad).
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> all_c: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read_write> pixels: array<f32>;

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

    var detail: f16 = 0.0h;

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

        let sign_x: f16 = select(-1.0h, 1.0h, fx < 0.5);
        let sign_y: f16 = select(-1.0h, 1.0h, fy < 0.5);

        let c = all_c[level_offset(ju) + kx * cells + ky];
        let c01 = c.x;
        let c10 = c.y;
        let c11 = c.z;

        detail += sign_y * c01 + sign_x * (c10 + sign_y * c11);
    }

    pixels[idx] = f32(detail) + params.area;
}
