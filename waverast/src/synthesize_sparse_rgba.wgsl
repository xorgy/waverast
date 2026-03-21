//! Sparse inverse Haar wavelet transform — writes sRGB RGBA8 to a storage texture.
//!
//! Coefficients are stored only for active cells (those overlapping the boundary).
//! Per-level binary search maps (level, cell_key) to the compact buffer index.
//! Inactive cells contribute zero detail and are skipped.

struct Params {
    width: u32,
    height: u32,
    max_j: u32,
    wh: f32,
    area: f32,
    fill_rule: u32,
    view_scale: f32,
    view_offset_x: f32,
    view_offset_y: f32,
    _pad: f32,
}

struct Coeffs {
    c01: f32,
    c10: f32,
    c11: f32,
}

struct LevelInfo {
    start: u32,
    count: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> all_c: array<Coeffs>;
@group(0) @binding(2) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<storage, read> level_infos: array<LevelInfo>;
@group(0) @binding(4) var<storage, read> cell_indices: array<u32>;

/// Binary search for cell_key in the sorted cell_indices slice for level j.
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

// Linear f32 -> sRGB u8
fn linear_to_srgb(v: f32) -> u32 {
    let c = clamp(v, 0.0, 1.0);
    var s: f32;
    if c <= 0.0031308 {
        s = c * 12.92;
    } else {
        s = 1.055 * pow(c, 1.0 / 2.4) - 0.055;
    }
    return u32(s * 255.0 + 0.5);
}

@compute @workgroup_size(16, 16)
fn synthesize_sparse_rgba(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if col >= params.width || row >= params.height {
        return;
    }
    let py = f32(row) * params.view_scale / params.wh + params.view_offset_y;
    let px = f32(col) * params.view_scale / params.wh + params.view_offset_x;

    var detail: f32 = 0.0;

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

        let cell_key = kx * cells + ky;
        let compact = find_compact_index(ju, cell_key);
        if compact >= 0 {
            let c = all_c[compact];
            detail += sign_y * c.c01 + sign_x * (c.c10 + sign_y * c.c11);
        }
    }

    let coverage = params.area + detail;

    // Black path on white background — apply sRGB encoding in shader.
    var filled: f32;
    if params.fill_rule == 1u {
        // Even-odd fill rule
        let w = abs(coverage);
        let frac = w - floor(w);
        if u32(floor(w)) % 2u == 0u {
            filled = frac;
        } else {
            filled = 1.0 - frac;
        }
    } else {
        // Nonzero fill rule (default)
        filled = abs(coverage);
    }
    let linear = clamp(1.0 - filled, 0.0, 1.0);
    let srgb = f32(linear_to_srgb(linear)) / 255.0;
    textureStore(output, vec2<i32>(i32(col), i32(row)), vec4(srgb, srgb, srgb, 1.0));
}
