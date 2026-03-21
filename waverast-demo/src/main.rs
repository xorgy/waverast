use std::fs::File;
use std::io::Write;

use waverast::contour::Shape;
use waverast::gpu::GpuCoeffComputer;
use waverast::rasterizer::{CoeffData, Coeffs, Rasterizer};

fn write_pgm(path: &str, pixels: &[f32], w: usize, h: usize) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    write!(f, "P5\n{w} {h}\n255\n")?;
    let bytes: Vec<u8> = pixels
        .iter()
        .map(|&v| (v.abs() * 255.0 + 0.5).clamp(0.0, 255.0) as u8)
        .collect();
    f.write_all(&bytes)?;
    Ok(())
}

fn readback_coeffs(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    gpu_data: &waverast::gpu::GpuCoeffData,
) -> CoeffData<Coeffs> {
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
        .map(|i| Coeffs {
            c01: f32::from_bits(raw[i * 3]),
            c10: f32::from_bits(raw[i * 3 + 1]),
            c11: f32::from_bits(raw[i * 3 + 2]),
        })
        .collect();
    drop(map);
    rb.unmap();
    CoeffData {
        area: gpu_data.area,
        coeffs,
    }
}

fn max_coeff_error(cpu: &CoeffData<Coeffs>, gpu: &CoeffData<Coeffs>) -> f32 {
    assert_eq!(cpu.coeffs.len(), gpu.coeffs.len());
    let mut max_err: f32 = 0.0;
    for (c, g) in cpu.coeffs.iter().zip(gpu.coeffs.iter()) {
        max_err = max_err
            .max((c.c01 - g.c01).abs())
            .max((c.c10 - g.c10).abs())
            .max((c.c11 - g.c11).abs());
    }
    max_err = max_err.max((cpu.area - gpu.area).abs());
    max_err
}

fn test_shape(
    name: &str,
    shape: Shape,
    w: usize,
    h: usize,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    comp: &GpuCoeffComputer,
) {
    let raster = Rasterizer::for_size(w, h);

    // CPU path
    let cpu_data = raster.compute(shape.clone());
    let cpu_pixels = raster.rasterize(&cpu_data);
    write_pgm(&format!("{name}_cpu.pgm"), &cpu_pixels, w, h).unwrap();

    // GPU path
    let gpu_raw = comp.compute_on_gpu(device, queue, &raster, shape);
    let gpu_data = readback_coeffs(device, queue, &gpu_raw);
    let gpu_pixels = raster.rasterize(&gpu_data);
    write_pgm(&format!("{name}_gpu.pgm"), &gpu_pixels, w, h).unwrap();

    let max_err = max_coeff_error(&cpu_data, &gpu_data);

    // Per-pixel coverage error
    let mut max_px_err: f32 = 0.0;
    let mut err_count = 0usize;
    for (c, g) in cpu_pixels.iter().zip(gpu_pixels.iter()) {
        let e = (c - g).abs();
        max_px_err = max_px_err.max(e);
        if e > 0.01 {
            err_count += 1;
        }
    }

    println!(
        "{name:>15}: max_coeff_err={max_err:.2e}  max_pixel_err={max_px_err:.4}  bad_pixels={err_count}",
    );
}

fn main() {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter =
        pollster::block_on(instance.request_adapter(&Default::default())).expect("no GPU adapter");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("waverast-demo"),
        required_limits: wgpu::Limits {
            max_storage_buffer_binding_size: adapter.limits().max_storage_buffer_binding_size,
            max_buffer_size: adapter.limits().max_buffer_size,
            ..Default::default()
        },
        ..Default::default()
    }))
    .expect("no GPU device");

    let comp = GpuCoeffComputer::new(&device);

    println!("Testing CPU vs GPU coefficient computation for all segment types:");
    println!();

    // Lines (triangle)
    test_shape(
        "triangle",
        Shape::polygon(&[(8.0, 8.0), (60.0, 12.0), (20.0, 28.0)]),
        64,
        64,
        &device,
        &queue,
        &comp,
    );

    // Quadratic Bezier
    test_shape(
        "quad_bezier",
        Shape::quad_bezier(&[(8.0, 8.0), (56.0, 8.0), (56.0, 56.0), (8.0, 56.0)]),
        64,
        64,
        &device,
        &queue,
        &comp,
    );

    // Cubic Bezier
    test_shape(
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
        64,
        &device,
        &queue,
        &comp,
    );

    // Circle (circular arcs)
    test_shape(
        "circle",
        Shape::circle(32.0, 32.0, 20.0),
        64,
        64,
        &device,
        &queue,
        &comp,
    );

    // Superellipse (squircle, n=4)
    test_shape(
        "squircle",
        Shape::superellipse(32.0, 32.0, 20.0, 20.0, 4.0),
        64,
        64,
        &device,
        &queue,
        &comp,
    );

    // Larger: 256x256 triangle
    test_shape(
        "triangle_256",
        Shape::polygon(&[(32.0, 32.0), (240.0, 48.0), (80.0, 200.0)]),
        256,
        256,
        &device,
        &queue,
        &comp,
    );
}
