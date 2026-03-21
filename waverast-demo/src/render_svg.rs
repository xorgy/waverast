use std::env;
use std::fs;
use std::io::BufWriter;
use std::time::Instant;

use waverast::contour::{CubicBez, Line, Segment, Shape};
use waverast::rasterizer::Rasterizer;

#[path = "svg_path.rs"]
mod svg_path;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: render-svg <path-file> <width> <height> [output.png] [flags]");
        eprintln!("  --gpu-coeff    Use GPU for coefficient computation");
        eprintln!("  --gpu-synth    Use GPU for synthesis");
        eprintln!("  --reference    Use reference analytic rasterizer (exact, slow)");
        eprintln!("  --compare      Compare wavelet vs reference and report errors");
        eprintln!("  --evenodd      Use even-odd fill rule (default: nonzero)");
        eprintln!("  --16bit        Output 16-bit sRGB PNG");
        eprintln!("  --sparse       Use sparse wavelet coefficients (GPU only)");
        eprintln!("  Flags can be combined. Without flags, both stages run on CPU.");
        std::process::exit(1);
    }

    let path_file = &args[1];
    let w: usize = args[2].parse().expect("invalid width");
    let h: usize = args[3].parse().expect("invalid height");
    let gpu_coeff_flag = args.iter().any(|a| a == "--gpu-coeff");
    let gpu_synth_flag = args.iter().any(|a| a == "--gpu-synth");
    let reference_flag = args.iter().any(|a| a == "--reference");
    let compare_flag = args.iter().any(|a| a == "--compare");
    let evenodd = args.iter().any(|a| a == "--evenodd");
    let depth_16 = args.iter().any(|a| a == "--16bit");
    let gpu_avg_flag = args.iter().any(|a| a == "--gpu-avg");
    let sparse_flag = args.iter().any(|a| a == "--sparse");
    let extra_levels: u32 = args
        .iter()
        .position(|a| a == "--extra-levels")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let out_file = args
        .iter()
        .skip(4)
        .enumerate()
        .find(|(i, a)| {
            !a.starts_with("--") && args.get(3 + *i).is_none_or(|p| p != "--extra-levels")
        })
        .map_or("output.png", |(_, s)| s.as_str());

    // ---- Parse SVG path ----
    let t_parse = Instant::now();
    let path_data = fs::read_to_string(path_file).expect("failed to read path file");
    let segments = svg_path::parse_svg_path(&path_data);
    println!(
        "Parse:       {:>10.3?}  ({} segments from {} bytes)",
        t_parse.elapsed(),
        segments.len(),
        path_data.len(),
    );

    // ---- Reference renderer shortcut ----
    if reference_flag {
        let t_ref = Instant::now();
        let fill_rule = if evenodd {
            waverast::reference::FillRule::EvenOdd
        } else {
            waverast::reference::FillRule::NonZero
        };
        let pixels =
            waverast::reference::reference_rasterize(&Shape::new(segments), w, h, fill_rule);
        println!(
            "Reference:   {:>10.3?}  ({}x{}, {} pixels)",
            t_ref.elapsed(),
            w,
            h,
            pixels.len()
        );
        write_png(out_file, &pixels, w, h, evenodd, depth_16).expect("failed to write PNG");
        println!("PNG output:  {:>10.3?}  ({})", t_ref.elapsed(), out_file);
        return;
    }

    // ---- GPU averaging mode: run 256 times and average coverage ----
    if gpu_avg_flag {
        let shape = Shape::new(segments);
        let raster = Rasterizer::for_size(w, h);
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
            .expect("no GPU adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("waverast"),
            required_limits: wgpu::Limits {
                max_storage_buffer_binding_size: adapter.limits().max_storage_buffer_binding_size,
                max_buffer_size: adapter.limits().max_buffer_size,
                ..Default::default()
            },
            ..Default::default()
        }))
        .expect("no GPU device");

        let comp = waverast::gpu::GpuCoeffComputer::new(&device);
        let npx = w * h;
        let mut accum = vec![0.0f64; npx];
        let runs = 256usize;

        let t_start = Instant::now();
        for run in 0..runs {
            let gpu_data = comp.compute_on_gpu(&device, &queue, &raster, shape.clone());
            let total = gpu_data.total_cells;
            let size = (total * 3 * 4) as u64;
            let rb = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rb"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut enc = device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(&gpu_data.buffer, 0, &rb, 0, size);
            queue.submit([enc.finish()]);
            let slice = rb.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::PollType::wait_indefinitely()).ok();
            let map = slice.get_mapped_range();
            let raw: &[u32] = bytemuck::cast_slice(&map);
            let coeffs: Vec<waverast::rasterizer::Coeffs> = (0..total)
                .map(|i| waverast::rasterizer::Coeffs {
                    c01: f32::from_bits(raw[i * 3]),
                    c10: f32::from_bits(raw[i * 3 + 1]),
                    c11: f32::from_bits(raw[i * 3 + 2]),
                })
                .collect();
            drop(map);
            rb.unmap();
            let data = waverast::rasterizer::CoeffData {
                area: gpu_data.area,
                coeffs,
            };
            let pixels = raster.rasterize(&data);
            for i in 0..npx {
                accum[i] += pixels[i] as f64;
            }
            if (run + 1) % 64 == 0 {
                println!("  run {}/{runs}...", run + 1);
            }
        }
        println!("GPU avg:     {:>10.3?}  ({runs} runs)", t_start.elapsed());

        let avg: Vec<f32> = accum.iter().map(|&v| (v / runs as f64) as f32).collect();
        write_png(out_file, &avg, w, h, evenodd, true).expect("failed to write PNG");
        println!("PNG output:  ({})", out_file);
        return;
    }

    // ---- Compare wavelet vs reference ----
    if compare_flag {
        let shape = Shape::new(segments);
        let fill_rule = if evenodd {
            waverast::reference::FillRule::EvenOdd
        } else {
            waverast::reference::FillRule::NonZero
        };

        let t_ref = Instant::now();
        let ref_px = waverast::reference::reference_rasterize(&shape, w, h, fill_rule);
        println!("Reference:   {:>10.3?}", t_ref.elapsed());

        let raster = Rasterizer::for_size(w, h);
        let t_wav = Instant::now();
        let data = raster.compute(shape.clone());
        let wav_px = raster.rasterize(&data);
        println!("Wavelet:     {:>10.3?}", t_wav.elapsed());

        // Enumerate ALL error cells across ALL levels
        let max_j = (w.max(h) as f64).log2().ceil() as u32;
        let wh = (1u64 << (max_j + 1)) as f64;

        println!("\nError cells (any coefficient with error > 1e-4):");
        #[allow(clippy::type_complexity)]
        let mut error_cells: Vec<(u32, usize, usize, f32, f32, f32, usize, bool)> = Vec::new();
        for j in 0..=max_j {
            let cells = 1usize << j;
            for kx in 0..cells.min(w) {
                for ky in 0..cells.min(h) {
                    let x_lo = kx as f64 / cells as f64 * wh;
                    let y_lo = ky as f64 / cells as f64 * wh;
                    if x_lo >= w as f64 || y_lo >= h as f64 {
                        continue;
                    }
                    let x_hi = (kx + 1) as f64 / cells as f64 * wh;
                    let y_hi = (ky + 1) as f64 / cells as f64 * wh;
                    let idx = waverast::rasterizer::level_offset(j) + kx * cells + ky;
                    let c = &data.coeffs[idx];
                    let (ec01, ec10, ec11) =
                        waverast::reference::expected_coefficient_from_reference(
                            &ref_px, w, h, j, kx, ky, wh as f32, data.area,
                        );
                    let e01 = (c.c01 - ec01).abs();
                    let e10 = (c.c10 - ec10).abs();
                    let e11 = (c.c11 - ec11).abs();
                    if e01 > 1e-4 || e10 > 1e-4 || e11 > 1e-4 {
                        let segs =
                            waverast::reference::segments_in_cell(&shape, x_lo, x_hi, y_lo, y_hi);
                        let mid_y = (y_lo + y_hi) * 0.5;
                        let mid_x = (x_lo + x_hi) * 0.5;
                        let has_midpt =
                            segs.iter()
                                .any(|&(si, _, _, _, _)| match &shape.segments[si] {
                                    Segment::Line(Line { p0, p1 }) => {
                                        ((p0.y - mid_y).abs() < 0.01 && (p1.y - mid_y).abs() < 0.01)
                                            || ((p0.x - mid_x).abs() < 0.01
                                                && (p1.x - mid_x).abs() < 0.01)
                                    }
                                    _ => false,
                                });
                        // Compare equation-8 vs direct formula
                        if e01 > 0.1 && has_midpt {
                            let all_si: Vec<usize> =
                                segs.iter().map(|&(si, _, _, _, _)| si).collect();
                            let direct =
                                waverast::reference::direct_c01(&shape, j, kx, ky, &all_si, wh);
                            let eq8 = waverast::reference::coefficient_at_cell(
                                &shape, j, kx, ky, &all_si, wh,
                            )
                            .0;
                            eprintln!(
                                "j={j} ({kx},{ky}): eq8_c01={eq8:.6} direct_c01={direct:.6} stored={:.6} expected={:.6} diff_eq8_direct={:.6}",
                                c.c01,
                                ec01,
                                eq8 as f64 - direct,
                            );
                        }
                        error_cells.push((j, kx, ky, e01, e10, e11, segs.len(), has_midpt));
                    }
                }
            }
        }
        error_cells.sort_by(|a, b| b.3.max(b.4).max(b.5).total_cmp(&a.3.max(a.4).max(a.5)));
        for &(j, kx, ky, e01, e10, e11, nsegs, midpt) in &error_cells {
            let cells = 1usize << j;
            let x_lo = kx as f64 / cells as f64 * wh;
            let x_hi = (kx + 1) as f64 / cells as f64 * wh;
            let y_lo = ky as f64 / cells as f64 * wh;
            let y_hi = (ky + 1) as f64 / cells as f64 * wh;
            let m = if midpt { " MIDPT" } else { "" };
            println!(
                "  j={j:2} ({kx:4},{ky:4}) [{x_lo:7.1}-{x_hi:7.1}]×[{y_lo:7.1}-{y_hi:7.1}] err=({e01:.4},{e10:.4},{e11:.4}) segs={nsegs}{m}",
            );
        }
        println!("\nTotal error cells: {}", error_cells.len());

        // Spot-check: compare eq8 vs direct on a few NON-error cells with significant coefficients
        println!("\nSpot-check eq8 vs direct on non-error cells:");
        let mut checked = 0;
        for j in [8u32, 9, 10, 11] {
            let cells = 1usize << j;
            for kx in 0..cells.min(w) {
                for ky in 0..cells.min(h) {
                    let x_lo = kx as f64 / cells as f64 * wh;
                    let y_lo = ky as f64 / cells as f64 * wh;
                    if x_lo >= w as f64 || y_lo >= h as f64 {
                        continue;
                    }
                    let x_hi = (kx + 1) as f64 / cells as f64 * wh;
                    let y_hi = (ky + 1) as f64 / cells as f64 * wh;
                    let idx = waverast::rasterizer::level_offset(j) + kx * cells + ky;
                    let c = &data.coeffs[idx];
                    if c.c01.abs() < 0.01 {
                        continue;
                    } // skip tiny coefficients
                    let (ec01, _, _) = waverast::reference::expected_coefficient_from_reference(
                        &ref_px, w, h, j, kx, ky, wh as f32, data.area,
                    );
                    if (c.c01 - ec01).abs() > 1e-4 {
                        continue;
                    } // skip error cells
                    let segs =
                        waverast::reference::segments_in_cell(&shape, x_lo, x_hi, y_lo, y_hi);
                    if segs.is_empty() {
                        continue;
                    }
                    let all_si: Vec<usize> = segs.iter().map(|&(si, _, _, _, _)| si).collect();
                    let direct = waverast::reference::direct_c01(&shape, j, kx, ky, &all_si, wh);
                    let eq8 =
                        waverast::reference::coefficient_at_cell(&shape, j, kx, ky, &all_si, wh).0;
                    let diff = eq8 as f64 - direct;
                    if diff.abs() > 1e-4 {
                        println!(
                            "  j={j} ({kx},{ky}) eq8={eq8:.6} direct={direct:.6} diff={diff:.6} stored={:.6}",
                            c.c01
                        );
                        checked += 1;
                        if checked >= 5 {
                            break;
                        }
                    }
                }
                if checked >= 5 {
                    break;
                }
            }
            if checked >= 5 {
                break;
            }
        }
        if checked == 0 {
            println!("  All checked cells: eq8 ≈ direct (no discrepancies > 1e-4)");
        }

        // Dump the sub-path containing a specific error segment
        if let Some(&(_, _, _, _, _, _, _, true)) = error_cells.first() {
            // Find a MIDPT error cell with large error
            for &(j, kx, ky, e01, e10, e11, _, midpt) in &error_cells {
                if !midpt || e01.max(e10).max(e11) < 0.4 {
                    continue;
                }
                let cells = 1usize << j;
                let x_lo = kx as f64 / cells as f64 * wh;
                let x_hi = (kx + 1) as f64 / cells as f64 * wh;
                let y_lo = ky as f64 / cells as f64 * wh;
                let y_hi = (ky + 1) as f64 / cells as f64 * wh;
                let segs = waverast::reference::segments_in_cell(&shape, x_lo, x_hi, y_lo, y_hi);
                if let Some(&(si, _, _, _, _)) = segs
                    .iter()
                    .find(|&&(si, _, _, _, _)| matches!(&shape.segments[si], Segment::Line(..)))
                {
                    // Find the full sub-path containing this segment
                    // Walk backwards to find sub-path start (a closing line or start of segments)
                    let start = si.saturating_sub(10);
                    let end = (si + 5).min(shape.segments.len());
                    println!("\nSub-path around seg[{si}] (cell j={j} ({kx},{ky})):");
                    for i in start..end {
                        let s = &shape.segments[i];
                        let desc = match s {
                            Segment::Line(Line { p0, p1 }) => {
                                format!("Line ({:.3},{:.3})→({:.3},{:.3})", p0.x, p0.y, p1.x, p1.y)
                            }
                            Segment::CubicBez(CubicBez { p0, p1, p2, p3 }) => format!(
                                "Cubic ({:.3},{:.3})→({:.3},{:.3})→({:.3},{:.3})→({:.3},{:.3})",
                                p0.x, p0.y, p1.x, p1.y, p2.x, p2.y, p3.x, p3.y
                            ),
                            _ => "Other".to_string(),
                        };
                        let marker = if i == si { " <--" } else { "" };
                        println!("    seg[{i}] {desc}{marker}");
                    }
                }
                break;
            }
        }

        write_png(out_file, &wav_px, w, h, evenodd, depth_16).expect("failed to write PNG");
        return;
    }

    // ---- GPU frame benchmark ----
    let gpu_bench_flag = args.iter().any(|a| a == "--gpu-bench");
    if gpu_bench_flag {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
            .expect("no GPU adapter");
        eprintln!("GPU: {}", adapter.get_info().name);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("waverast"),
            required_limits: wgpu::Limits {
                max_storage_buffer_binding_size: adapter.limits().max_storage_buffer_binding_size,
                max_buffer_size: adapter.limits().max_buffer_size,
                ..Default::default()
            },
            ..Default::default()
        }))
        .expect("no GPU device");
        let gpu_synth = waverast::gpu::GpuSynthesizer::new(&device, false);
        let gpu_coeff = waverast::gpu::GpuCoeffComputer::new(&device);
        let raster = Rasterizer::for_size(w, h);

        // Warmup
        let shape = Shape::new(segments.clone());
        let (_data, _tex) =
            gpu_synth.compute_and_rasterize(&gpu_coeff, &device, &queue, &raster, shape, evenodd);
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        // Benchmark: coefficients only
        let n = 20;
        let t0 = Instant::now();
        for _ in 0..n {
            let shape = Shape::new(segments.clone());
            let _data = gpu_coeff.compute_on_gpu(&device, &queue, &raster, shape);
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
        }
        let t_coeff = t0.elapsed() / n as u32;

        // Benchmark: synthesis only (reuse last coefficients)
        let shape = Shape::new(segments.clone());
        let data = gpu_coeff.compute_on_gpu(&device, &queue, &raster, shape);
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let t0 = Instant::now();
        for _ in 0..n {
            let _tex = gpu_synth.rasterize_to_texture(&device, &queue, &raster, &data, evenodd);
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
        }
        let t_synth = t0.elapsed() / n as u32;

        // Benchmark: combined
        let t0 = Instant::now();
        for _ in 0..n {
            let shape = Shape::new(segments.clone());
            let (_data, _tex) = gpu_synth
                .compute_and_rasterize(&gpu_coeff, &device, &queue, &raster, shape, evenodd);
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
        }
        let t_combined = t0.elapsed() / n as u32;

        // Per-type benchmark
        let count_by_type = |segs: &[Segment]| {
            let (mut nl, mut nq, mut nc, mut na, mut ns) = (0, 0, 0, 0, 0);
            for s in segs {
                match s {
                    Segment::Line(..) => nl += 1,
                    Segment::QuadBez(..) => nq += 1,
                    Segment::CubicBez(..) => nc += 1,
                    Segment::CircularArc(..) => na += 1,
                    Segment::Superellipse(..) => ns += 1,
                }
            }
            (nl, nq, nc, na, ns)
        };
        let (nl, nq, nc, na, ns) = count_by_type(&segments);

        // Cubics only
        let cubics_only: Vec<_> = segments
            .iter()
            .filter(|s| matches!(s, Segment::CubicBez(..)))
            .cloned()
            .collect();
        let t0 = Instant::now();
        for _ in 0..n {
            let shape = Shape::new(cubics_only.clone());
            let _data = gpu_coeff.compute_on_gpu(&device, &queue, &raster, shape);
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
        }
        let t_cubics = t0.elapsed() / n as u32;

        // Lines only
        let lines_only: Vec<_> = segments
            .iter()
            .filter(|s| matches!(s, Segment::Line(..)))
            .cloned()
            .collect();
        let t0 = Instant::now();
        for _ in 0..n {
            let shape = Shape::new(lines_only.clone());
            let _data = gpu_coeff.compute_on_gpu(&device, &queue, &raster, shape);
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
        }
        let t_lines = t0.elapsed() / n as u32;

        // Arcs only
        let arcs_only: Vec<_> = segments
            .iter()
            .filter(|s| matches!(s, Segment::CircularArc(..)))
            .cloned()
            .collect();
        let t0 = Instant::now();
        for _ in 0..n {
            let shape = Shape::new(arcs_only.clone());
            let _data = gpu_coeff.compute_on_gpu(&device, &queue, &raster, shape);
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
        }
        let t_arcs = t0.elapsed() / n as u32;

        // Sparse benchmarks
        // Warmup
        let shape = Shape::new(segments.clone());
        let (_sdata, _stex) = gpu_synth
            .compute_and_rasterize_sparse(&gpu_coeff, &device, &queue, &raster, shape, evenodd);
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        // Sparse: coefficients only
        let t0 = Instant::now();
        for _ in 0..n {
            let shape = Shape::new(segments.clone());
            let _data = gpu_coeff.compute_on_gpu_sparse(&device, &queue, &raster, shape);
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
        }
        let t_sparse_coeff = t0.elapsed() / n as u32;

        // Sparse: synthesis only
        let shape = Shape::new(segments.clone());
        let sparse_data = gpu_coeff.compute_on_gpu_sparse(&device, &queue, &raster, shape);
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let t0 = Instant::now();
        for _ in 0..n {
            let _tex = gpu_synth.rasterize_sparse_to_texture(
                &device,
                &queue,
                &raster,
                &sparse_data,
                evenodd,
            );
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
        }
        let t_sparse_synth = t0.elapsed() / n as u32;

        // Sparse: combined
        let t0 = Instant::now();
        for _ in 0..n {
            let shape = Shape::new(segments.clone());
            let (_sdata, _stex) = gpu_synth
                .compute_and_rasterize_sparse(&gpu_coeff, &device, &queue, &raster, shape, evenodd);
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
        }
        let t_sparse_combined = t0.elapsed() / n as u32;

        println!("GPU benchmark ({n} frames, {w}x{h}):");
        println!("  segments: {nl} lines, {nq} quads, {nc} cubics, {na} arcs, {ns} SEs");
        println!("  Dense:");
        println!("    coefficients only: {t_coeff:.2?}");
        println!("    synthesis only:    {t_synth:.2?}");
        println!("    combined:          {t_combined:.2?}");
        println!("    cubics only:       {t_cubics:.2?}");
        println!("    lines only:        {t_lines:.2?}");
        println!("    arcs only:         {t_arcs:.2?}");
        println!("  Sparse ({} active cells):", sparse_data.n_active);
        println!("    coefficients only: {t_sparse_coeff:.2?}");
        println!("    synthesis only:    {t_sparse_synth:.2?}");
        println!("    combined:          {t_sparse_combined:.2?}");
        return;
    }

    // ---- Compare GPU coefficients vs CPU coefficients ----
    let gpu_compare_flag = args.iter().any(|a| a == "--gpu-compare");
    if gpu_compare_flag {
        let shape = Shape::new(segments);
        let raster = Rasterizer::for_size(w, h);

        // CPU coefficients (ground truth with direct tent formula)
        let t_cpu = Instant::now();
        let cpu_data = raster.compute(shape.clone());
        println!("CPU coeffs:  {:>10.3?}", t_cpu.elapsed());

        // GPU coefficients (equation-8 in f32)
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
            .expect("no GPU adapter");
        eprintln!("GPU: {}", adapter.get_info().name);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("waverast"),
            required_limits: wgpu::Limits {
                max_storage_buffer_binding_size: adapter.limits().max_storage_buffer_binding_size,
                max_buffer_size: adapter.limits().max_buffer_size,
                ..Default::default()
            },
            ..Default::default()
        }))
        .expect("no GPU device");

        let comp = waverast::gpu::GpuCoeffComputer::new(&device);
        let t_gpu = Instant::now();
        let gpu_data = comp.compute_on_gpu(&device, &queue, &raster, shape.clone());
        println!("GPU coeffs:  {:>10.3?}", t_gpu.elapsed());

        // Read back GPU coefficients
        let total = gpu_data.total_cells;
        let size = (total * 3 * 4) as u64;
        let rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rb"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&gpu_data.buffer, 0, &rb, 0, size);
        queue.submit([enc.finish()]);
        let slice = rb.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let map = slice.get_mapped_range();
        let raw: &[u32] = bytemuck::cast_slice(&map);
        let gpu_coeffs: Vec<waverast::rasterizer::Coeffs> = (0..total)
            .map(|i| waverast::rasterizer::Coeffs {
                c01: f32::from_bits(raw[i * 3]),
                c10: f32::from_bits(raw[i * 3 + 1]),
                c11: f32::from_bits(raw[i * 3 + 2]),
            })
            .collect();
        drop(map);
        rb.unmap();

        // Compare coefficient by coefficient
        let max_j = raster.max_level();
        let wh = raster.grid_size() as f64;
        println!("\nGPU vs CPU coefficient differences (|diff| > 1e-4):");

        struct CellErr {
            j: u32,
            kx: usize,
            ky: usize,
            d01: f32,
            d10: f32,
            d11: f32,
            cpu_c01: f32,
            gpu_c01: f32,
            cpu_c10: f32,
            gpu_c10: f32,
            cpu_c11: f32,
            gpu_c11: f32,
            midpt: bool,
        }
        let mut errs: Vec<CellErr> = Vec::new();

        for j in 0..=max_j {
            let cells = 1usize << j;
            let base = waverast::rasterizer::level_offset(j);
            for kx in 0..cells {
                for ky in 0..cells {
                    let idx = base + kx * cells + ky;
                    let cc = &cpu_data.coeffs[idx];
                    let gc = &gpu_coeffs[idx];
                    let d01 = (gc.c01 - cc.c01).abs();
                    let d10 = (gc.c10 - cc.c10).abs();
                    let d11 = (gc.c11 - cc.c11).abs();
                    if d01 > 1e-4 || d10 > 1e-4 || d11 > 1e-4 {
                        let x_lo = kx as f64 / cells as f64 * wh;
                        let x_hi = (kx + 1) as f64 / cells as f64 * wh;
                        let y_lo = ky as f64 / cells as f64 * wh;
                        let y_hi = (ky + 1) as f64 / cells as f64 * wh;
                        let segs =
                            waverast::reference::segments_in_cell(&shape, x_lo, x_hi, y_lo, y_hi);
                        let mid_y = (y_lo + y_hi) * 0.5;
                        let mid_x = (x_lo + x_hi) * 0.5;
                        let has_midpt =
                            segs.iter()
                                .any(|&(si, _, _, _, _)| match &shape.segments[si] {
                                    Segment::Line(Line { p0, p1 }) => {
                                        ((p0.y - mid_y).abs() < 0.01 && (p1.y - mid_y).abs() < 0.01)
                                            || ((p0.x - mid_x).abs() < 0.01
                                                && (p1.x - mid_x).abs() < 0.01)
                                    }
                                    _ => false,
                                });
                        errs.push(CellErr {
                            j,
                            kx,
                            ky,
                            d01,
                            d10,
                            d11,
                            cpu_c01: cc.c01,
                            gpu_c01: gc.c01,
                            cpu_c10: cc.c10,
                            gpu_c10: gc.c10,
                            cpu_c11: cc.c11,
                            gpu_c11: gc.c11,
                            midpt: has_midpt,
                        });
                    }
                }
            }
        }

        errs.sort_by(|a, b| {
            b.d01
                .max(b.d10)
                .max(b.d11)
                .total_cmp(&a.d01.max(a.d10).max(a.d11))
        });

        // Categorize errors
        let mut midpt_only_c01c10 = 0usize;
        let mut midpt_with_c11 = 0usize;
        let mut precision_only = 0usize;
        let mut other = 0usize;

        for e in &errs {
            let max_d = e.d01.max(e.d10).max(e.d11);
            if e.midpt && e.d11 < 1e-3 && (e.d01 > 1e-3 || e.d10 > 1e-3) {
                midpt_only_c01c10 += 1;
            } else if e.midpt {
                midpt_with_c11 += 1;
            } else if max_d < 0.01 {
                precision_only += 1;
            } else {
                other += 1;
            }
        }

        // Print top errors
        for (i, e) in errs.iter().take(30).enumerate() {
            let cells = 1usize << e.j;
            let x_lo = e.kx as f64 / cells as f64 * wh;
            let y_lo = e.ky as f64 / cells as f64 * wh;
            let m = if e.midpt { " MIDPT" } else { "" };
            println!(
                "  {:2}. j={:2} ({:4},{:4}) px≈({:.0},{:.0}) d=({:.5},{:.5},{:.5}) cpu=({:.5},{:.5},{:.5}) gpu=({:.5},{:.5},{:.5}){}",
                i + 1,
                e.j,
                e.kx,
                e.ky,
                x_lo,
                y_lo,
                e.d01,
                e.d10,
                e.d11,
                e.cpu_c01,
                e.cpu_c10,
                e.cpu_c11,
                e.gpu_c01,
                e.gpu_c10,
                e.gpu_c11,
                m,
            );
        }

        println!("\nTotal cells with GPU≠CPU: {}", errs.len());
        println!("  Midpoint c01/c10 errors (c11 ok): {midpt_only_c01c10}");
        println!("  Midpoint errors (all coeffs):     {midpt_with_c11}");
        println!("  Precision-only (max < 0.01):      {precision_only}");
        println!("  Other:                            {other}");

        // Histogram by level
        println!("\nErrors by level:");
        for j in 0..=max_j {
            let count = errs.iter().filter(|e| e.j == j).count();
            if count > 0 {
                let max_d: f32 = errs
                    .iter()
                    .filter(|e| e.j == j)
                    .map(|e| e.d01.max(e.d10).max(e.d11))
                    .fold(0.0f32, f32::max);
                println!("  j={j:2}: {count:5} cells  max_diff={max_d:.5}");
            }
        }

        // Pixel-level comparison: synthesize and compare
        let gpu_px_data = waverast::rasterizer::CoeffData {
            area: gpu_data.area,
            coeffs: gpu_coeffs.clone(),
        };
        let gpu_px = raster.rasterize(&gpu_px_data);
        // Compare GPU against reference (exact analytic rasterizer)
        let fill_rule = if evenodd {
            waverast::reference::FillRule::EvenOdd
        } else {
            waverast::reference::FillRule::NonZero
        };
        let ref_px = waverast::reference::reference_rasterize(&shape, w, h, fill_rule);
        let mut srgb8_wrong = 0usize;
        let mut srgb16_wrong = 0usize;
        let mut mean_cubed_err: f64 = 0.0;
        for i in 0..ref_px.len() {
            if linear_to_srgb_u8(gpu_px[i], evenodd) != linear_to_srgb_u8(ref_px[i], evenodd) {
                srgb8_wrong += 1;
            }
            if linear_to_srgb_u16(gpu_px[i], evenodd) != linear_to_srgb_u16(ref_px[i], evenodd) {
                srgb16_wrong += 1;
            }
            let gpu_srgb = linear_to_srgb(coverage_to_linear(gpu_px[i], evenodd)) as f64;
            let ref_srgb = linear_to_srgb(coverage_to_linear(ref_px[i], evenodd)) as f64;
            let diff = gpu_srgb - ref_srgb;
            mean_cubed_err += diff * diff * diff.abs();
        }
        mean_cubed_err /= ref_px.len() as f64;
        println!(
            "\nGPU vs reference: srgb8={srgb8_wrong}  srgb16={srgb16_wrong}  mce={mean_cubed_err:.10e}"
        );

        // CPU vs reference
        let cpu_px = raster.rasterize(&cpu_data);
        let mut cpu_srgb8_wrong = 0usize;
        let mut cpu_srgb16_wrong = 0usize;
        let mut cpu_mce: f64 = 0.0;
        for i in 0..ref_px.len() {
            if linear_to_srgb_u8(cpu_px[i], evenodd) != linear_to_srgb_u8(ref_px[i], evenodd) {
                cpu_srgb8_wrong += 1;
            }
            if linear_to_srgb_u16(cpu_px[i], evenodd) != linear_to_srgb_u16(ref_px[i], evenodd) {
                cpu_srgb16_wrong += 1;
            }
            let cpu_srgb = linear_to_srgb(coverage_to_linear(cpu_px[i], evenodd)) as f64;
            let ref_srgb = linear_to_srgb(coverage_to_linear(ref_px[i], evenodd)) as f64;
            let diff = cpu_srgb - ref_srgb;
            cpu_mce += diff * diff * diff.abs();
        }
        cpu_mce /= ref_px.len() as f64;
        println!(
            "CPU vs reference: srgb8={cpu_srgb8_wrong}  srgb16={cpu_srgb16_wrong}  mce={cpu_mce:.10e}"
        );

        // ---- Sparse GPU comparison ----
        let synth = waverast::gpu::GpuSynthesizer::new(&device, false);
        let t_sparse = Instant::now();
        let sparse_data = comp.compute_on_gpu_sparse(&device, &queue, &raster, shape.clone());
        println!(
            "Sparse GPU coeffs: {:>7.3?}  ({} active cells, dense={})",
            t_sparse.elapsed(),
            sparse_data.n_active,
            total,
        );

        // Read back sparse coefficients and expand to dense for comparison
        let sparse_n = sparse_data.n_active;
        let sparse_size = (sparse_n.max(1) * 3 * 4) as u64;
        let sparse_rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sparse_rb"),
            size: sparse_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let li_size = (sparse_data.max_j as usize + 1) * 2 * 4;
        let li_rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("li_rb"),
            size: li_size.max(4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let ci_size = sparse_n.max(1) * 4;
        let ci_rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ci_rb"),
            size: ci_size as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc2 = device.create_command_encoder(&Default::default());
        enc2.copy_buffer_to_buffer(&sparse_data.buffer, 0, &sparse_rb, 0, sparse_size);
        enc2.copy_buffer_to_buffer(
            &sparse_data.level_info_buf,
            0,
            &li_rb,
            0,
            li_size.max(4) as u64,
        );
        enc2.copy_buffer_to_buffer(&sparse_data.cell_indices_buf, 0, &ci_rb, 0, ci_size as u64);
        queue.submit([enc2.finish()]);

        // Map all three
        sparse_rb.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        li_rb.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        ci_rb.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).ok();

        let sparse_raw: Vec<u32> =
            bytemuck::cast_slice(&sparse_rb.slice(..).get_mapped_range()).to_vec();
        let li_raw: Vec<u32> = bytemuck::cast_slice(&li_rb.slice(..).get_mapped_range()).to_vec();
        let ci_raw: Vec<u32> = bytemuck::cast_slice(&ci_rb.slice(..).get_mapped_range()).to_vec();

        // Expand sparse coefficients to dense for pixel-level comparison
        let mut sparse_dense = vec![
            waverast::rasterizer::Coeffs {
                c01: 0.0,
                c10: 0.0,
                c11: 0.0
            };
            total
        ];
        for j in 0..=max_j {
            let li_start = li_raw[j as usize * 2] as usize;
            let li_count = li_raw[j as usize * 2 + 1] as usize;
            let base = waverast::rasterizer::level_offset(j);
            for i in 0..li_count {
                let compact_idx = li_start + i;
                let cell_key = ci_raw[compact_idx] as usize;
                let dense_idx = base + cell_key;
                if dense_idx < total && compact_idx < sparse_n {
                    sparse_dense[dense_idx] = waverast::rasterizer::Coeffs {
                        c01: f32::from_bits(sparse_raw[compact_idx * 3]),
                        c10: f32::from_bits(sparse_raw[compact_idx * 3 + 1]),
                        c11: f32::from_bits(sparse_raw[compact_idx * 3 + 2]),
                    };
                }
            }
        }

        // Compare sparse vs dense GPU
        let mut sparse_diff_count = 0usize;
        let mut sparse_max_diff: f32 = 0.0;
        for i in 0..total {
            let d01 = (sparse_dense[i].c01 - gpu_coeffs[i].c01).abs();
            let d10 = (sparse_dense[i].c10 - gpu_coeffs[i].c10).abs();
            let d11 = (sparse_dense[i].c11 - gpu_coeffs[i].c11).abs();
            let d = d01.max(d10).max(d11);
            if d > 1e-7 {
                sparse_diff_count += 1;
                sparse_max_diff = sparse_max_diff.max(d);
            }
        }
        println!(
            "Sparse vs Dense GPU: {} cells differ, max_diff={:.10e}",
            sparse_diff_count, sparse_max_diff
        );

        // Pixel-level comparison for sparse
        let sparse_px_data = waverast::rasterizer::CoeffData {
            area: sparse_data.area,
            coeffs: sparse_dense,
        };
        let sparse_px = raster.rasterize(&sparse_px_data);
        let mut sparse_srgb8_wrong = 0usize;
        let mut sparse_mce: f64 = 0.0;
        for i in 0..ref_px.len() {
            if linear_to_srgb_u8(sparse_px[i], evenodd) != linear_to_srgb_u8(ref_px[i], evenodd) {
                sparse_srgb8_wrong += 1;
            }
            let sp_srgb = linear_to_srgb(coverage_to_linear(sparse_px[i], evenodd)) as f64;
            let ref_srgb = linear_to_srgb(coverage_to_linear(ref_px[i], evenodd)) as f64;
            let diff = sp_srgb - ref_srgb;
            sparse_mce += diff * diff * diff.abs();
        }
        sparse_mce /= ref_px.len() as f64;
        println!("Sparse vs reference: srgb8={sparse_srgb8_wrong}  mce={sparse_mce:.10e}");

        // Also test the sparse GPU synthesis path end-to-end
        let sparse_tex =
            synth.rasterize_sparse_to_texture(&device, &queue, &raster, &sparse_data, evenodd);
        // Read back the texture
        let tex_w = raster.width() as u32;
        let tex_h = raster.height() as u32;
        let bytes_per_row = (tex_w * 4).div_ceil(256) * 256;
        let tex_rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tex_rb"),
            size: (bytes_per_row * tex_h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc3 = device.create_command_encoder(&Default::default());
        enc3.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &sparse_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &tex_rb,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: tex_w,
                height: tex_h,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([enc3.finish()]);
        tex_rb.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let tex_data = tex_rb.slice(..).get_mapped_range();

        // Compare sparse GPU-synthesized texture vs reference
        let mut sparse_gpu_srgb8_wrong = 0usize;
        for row in 0..tex_h {
            for col in 0..tex_w {
                let idx = (row * bytes_per_row + col * 4) as usize;
                let r = tex_data[idx];
                let i = (row * tex_w + col) as usize;
                let ref_val = linear_to_srgb_u8(ref_px[i], evenodd);
                if r != ref_val {
                    sparse_gpu_srgb8_wrong += 1;
                }
            }
        }
        drop(tex_data);
        tex_rb.unmap();

        println!("Sparse GPU synthesis vs reference: srgb8={sparse_gpu_srgb8_wrong}");

        return;
    }

    let raster = if extra_levels > 0 {
        Rasterizer::for_size_with_extra_levels(w, h, extra_levels)
    } else {
        Rasterizer::for_size(w, h)
    };

    // Set up GPU if any GPU stage is requested
    let gpu = if gpu_coeff_flag || gpu_synth_flag || sparse_flag {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
            .expect("no GPU adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("waverast"),
            required_limits: wgpu::Limits {
                max_storage_buffer_binding_size: adapter.limits().max_storage_buffer_binding_size,
                max_buffer_size: adapter.limits().max_buffer_size,
                ..Default::default()
            },
            ..Default::default()
        }))
        .expect("no GPU device");
        Some((device, queue))
    } else {
        None
    };

    // ---- Sparse combined GPU path (compute + rasterize to texture) ----
    if sparse_flag {
        let (device, queue) = gpu.as_ref().unwrap();
        let synth = waverast::gpu::GpuSynthesizer::new(device, false);
        let coeff = waverast::gpu::GpuCoeffComputer::new(device);

        let t0 = Instant::now();
        let (_sparse_data, texture) = synth.compute_and_rasterize_sparse(
            &coeff,
            device,
            queue,
            &raster,
            Shape::new(segments),
            evenodd,
        );
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        println!(
            "Sparse GPU:  {:>10.3?}  ({}x{}, {} active cells)",
            t0.elapsed(),
            w,
            h,
            _sparse_data.n_active,
        );

        // Read back texture and write PNG
        let tex_w = w as u32;
        let tex_h = h as u32;
        let bytes_per_row = (tex_w * 4).div_ceil(256) * 256;
        let rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sparse_tex_rb"),
            size: (bytes_per_row * tex_h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &rb,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: tex_w,
                height: tex_h,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([enc.finish()]);
        rb.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let tex_data = rb.slice(..).get_mapped_range();

        // Build pixel array from texture readback
        let mut pixels = vec![0.0f32; (w * h) as usize];
        for row in 0..tex_h {
            for col in 0..tex_w {
                let idx = (row * bytes_per_row + col * 4) as usize;
                // Texture is sRGB-encoded Rgba8Unorm; extract the R channel
                // as a raw u8 value and convert back to coverage for PNG output
                let srgb_byte = tex_data[idx];
                pixels[(row * tex_w + col) as usize] = srgb_byte as f32 / 255.0;
            }
        }
        drop(tex_data);
        rb.unmap();

        // Write as-is (already sRGB)
        write_png_srgb(out_file, &pixels, w, h, depth_16).expect("failed to write PNG");
        println!("Output:      {:>10.3?}  ({})", t0.elapsed(), out_file);
        return;
    }

    // ---- Compute wavelet coefficients ----
    let t_compute = Instant::now();
    let data = if gpu_coeff_flag {
        let (device, queue) = gpu.as_ref().unwrap();
        let comp = waverast::gpu::GpuCoeffComputer::new(device);
        let gpu_data = comp.compute_on_gpu(device, queue, &raster, Shape::new(segments));

        // Read back coefficients for CPU synthesis / PNG output
        let total = gpu_data.total_cells;
        let size = (total * 3 * 4) as u64;
        let rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rb"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&gpu_data.buffer, 0, &rb, 0, size);
        queue.submit([enc.finish()]);
        let slice = rb.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let map = slice.get_mapped_range();
        let raw: &[u32] = bytemuck::cast_slice(&map);
        let coeffs = (0..total)
            .map(|i| waverast::rasterizer::Coeffs {
                c01: f32::from_bits(raw[i * 3]),
                c10: f32::from_bits(raw[i * 3 + 1]),
                c11: f32::from_bits(raw[i * 3 + 2]),
            })
            .collect();
        drop(map);
        rb.unmap();
        let data = waverast::rasterizer::CoeffData {
            area: gpu_data.area,
            coeffs,
        };
        println!(
            "Compute:     {:>10.3?}  ({}x{}, GPU)",
            t_compute.elapsed(),
            w,
            h
        );
        data
    } else {
        let data = raster.compute(Shape::new(segments));
        println!(
            "Compute:     {:>10.3?}  ({}x{}, CPU)",
            t_compute.elapsed(),
            w,
            h
        );
        data
    };

    // ---- Synthesize pixel coverage ----
    let t_synth = Instant::now();
    let pixels = if gpu_synth_flag {
        let (device, queue) = gpu.as_ref().unwrap();
        let synth = waverast::gpu::GpuSynthesizer::new(device, true);
        let px = synth.rasterize_f32(device, queue, &raster, &data);
        println!(
            "Synthesize:  {:>10.3?}  ({} pixels, GPU f32)",
            t_synth.elapsed(),
            px.len()
        );
        px
    } else {
        let px = raster.rasterize(&data);
        println!(
            "Synthesize:  {:>10.3?}  ({} pixels, CPU)",
            t_synth.elapsed(),
            px.len()
        );
        px
    };
    let synth_dur = t_synth.elapsed();

    // ---- Write PNG ----
    let t_png = Instant::now();
    write_png(out_file, &pixels, w, h, evenodd, depth_16).expect("failed to write PNG");
    println!("PNG output:  {:>10.3?}  ({})", t_png.elapsed(), out_file);

    // ---- Summary ----
    let compute_dur = t_compute.elapsed() - synth_dur;
    println!("─────────────────────────");
    println!(
        "Rasterize:   {:>10.3?}  (compute + synthesize)",
        compute_dur + synth_dur
    );
    println!("Total:       {:>10.3?}", t_parse.elapsed());
}

/// Coverage → linear grayscale (path=black on white), applying fill rule.
fn coverage_to_linear(v: f32, evenodd: bool) -> f32 {
    let filled = if evenodd {
        let w = v.abs();
        let frac = w - w.floor();
        if (w.floor() as i32) % 2 == 0 {
            frac
        } else {
            1.0 - frac
        }
    } else {
        v.abs()
    };
    (1.0 - filled).clamp(0.0, 1.0)
}

/// Linear → sRGB transfer function.
fn linear_to_srgb(linear: f32) -> f32 {
    if linear <= 0.0031308 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

fn linear_to_srgb_u8(v: f32, evenodd: bool) -> u8 {
    (linear_to_srgb(coverage_to_linear(v, evenodd)) * 255.0 + 0.5) as u8
}

fn linear_to_srgb_u16(v: f32, evenodd: bool) -> u16 {
    (linear_to_srgb(coverage_to_linear(v, evenodd)) * 65535.0 + 0.5) as u16
}

fn write_png(
    path: &str,
    pixels: &[f32],
    w: usize,
    h: usize,
    evenodd: bool,
    depth_16: bool,
) -> std::io::Result<()> {
    let file = fs::File::create(path)?;
    let buf = BufWriter::new(file);
    let mut encoder = png::Encoder::new(buf, w as u32, h as u32);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
    if depth_16 {
        encoder.set_depth(png::BitDepth::Sixteen);
        let mut writer = encoder.write_header().map_err(std::io::Error::other)?;
        let bytes: Vec<u8> = pixels
            .iter()
            .flat_map(|&v| linear_to_srgb_u16(v, evenodd).to_be_bytes())
            .collect();
        writer
            .write_image_data(&bytes)
            .map_err(std::io::Error::other)?;
    } else {
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().map_err(std::io::Error::other)?;
        let bytes: Vec<u8> = pixels
            .iter()
            .map(|&v| linear_to_srgb_u8(v, evenodd))
            .collect();
        writer
            .write_image_data(&bytes)
            .map_err(std::io::Error::other)?;
    }
    Ok(())
}

/// Write a PNG from already-sRGB-encoded pixel values (0.0..1.0 range).
fn write_png_srgb(
    path: &str,
    pixels: &[f32],
    w: usize,
    h: usize,
    _depth_16: bool,
) -> std::io::Result<()> {
    let file = fs::File::create(path)?;
    let buf = BufWriter::new(file);
    let mut encoder = png::Encoder::new(buf, w as u32, h as u32);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().map_err(std::io::Error::other)?;
    let bytes: Vec<u8> = pixels
        .iter()
        .map(|&v| (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8)
        .collect();
    writer
        .write_image_data(&bytes)
        .map_err(std::io::Error::other)?;
    Ok(())
}
