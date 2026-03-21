//! GPU-accelerated wavelet coefficient computation and synthesis via wgpu.
//!
//! Provides `GpuCoeffComputer` for computing wavelet coefficients on the GPU
//! and `GpuSynthesizer` for the inverse wavelet transform (pixel synthesis).
//! When both stages run on GPU, the coefficient buffer stays device-resident.

use crate::contour::{CircularArc, CubicBez, Line, QuadBez, Segment, Shape, Superellipse};
use crate::rasterizer::{CoeffData, Coeffs, CoeffsF16, Rasterizer, level_offset};

// ---- Shared types ----

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct SynthParams {
    width: u32,
    height: u32,
    max_j: u32,
    wh: f32,
    area: f32,
    fill_rule: u32,
    // View transform for progressive rendering: maps display pixels to
    // coefficient space. When (1.0, 0.0, 0.0), synthesis is identity.
    view_scale: f32,    // zoom_coeff / zoom_display
    view_offset_x: f32, // pan delta in normalized coeff coords
    view_offset_y: f32,
    _pad: f32,
}
unsafe impl bytemuck::Pod for SynthParams {}
unsafe impl bytemuck::Zeroable for SynthParams {}

/// GPU-resident coefficient data.
///
/// Wraps a `wgpu::Buffer` containing f32 coefficients in the 3×u32 atomic
/// layout (bit-identical to 3×f32). Can be passed directly to
/// [`GpuSynthesizer::rasterize_buffer`] without readback.
pub struct GpuCoeffData {
    pub area: f32,
    /// f32 coefficients as 3 × u32 per cell (atomic accumulation layout).
    pub buffer: wgpu::Buffer,
    /// Total number of coefficient cells.
    pub total_cells: usize,
}

/// GPU-resident sparse coefficient data.
pub struct GpuSparseCoeffData {
    pub area: f32,
    pub buffer: wgpu::Buffer,
    pub n_active: usize,
    pub level_info_buf: wgpu::Buffer,
    pub cell_indices_buf: wgpu::Buffer,
    pub max_j: u32,
}

/// CPU-side sparse index for readback/comparison.
#[derive(Clone)]
pub struct SparseIndexMap {
    pub level_info: Vec<(u32, u32)>,
    pub cell_indices: Vec<u32>,
    pub n_active: usize,
}

// ---- GPU Synthesizer ----

/// GPU synthesis context for the inverse wavelet transform.
///
/// Holds the compute pipeline and bind group layout. Created with a `force_f32`
/// flag that controls whether the f16 shader variant is used.
pub struct GpuSynthesizer {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    rgba_pipeline: wgpu::ComputePipeline,
    rgba_bgl: wgpu::BindGroupLayout,
    sparse_rgba_pipeline: wgpu::ComputePipeline,
    sparse_rgba_bgl: wgpu::BindGroupLayout,
    use_f16: bool,
}

impl GpuSynthesizer {
    /// Create a GPU synthesizer.
    ///
    /// When `force_f32` is false and the device supports `ShaderF16`, the fp16
    /// shader variant is selected automatically.
    pub fn new(device: &wgpu::Device, force_f32: bool) -> Self {
        let use_f16 = !force_f32 && device.features().contains(wgpu::Features::SHADER_F16);

        let shader_src = if use_f16 {
            include_str!("synthesize_f16.wgsl")
        } else {
            include_str!("synthesize.wgsl")
        };

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("synthesize"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("synthesize_bgl"),
            entries: &[bgl_uniform(0), bgl_storage_ro(1), bgl_storage_rw(2)],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("synthesize_pl"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("synthesize"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("synthesize"),
            compilation_options: Default::default(),
            cache: None,
        });

        // RGBA output pipeline (synthesize_to_rgba.wgsl)
        let rgba_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("synth_rgba"),
            source: wgpu::ShaderSource::Wgsl(include_str!("synthesize_to_rgba.wgsl").into()),
        });
        let rgba_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("synth_rgba_bgl"),
            entries: &[
                bgl_uniform(0),
                bgl_storage_ro(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let rgba_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("synth_rgba_pl"),
            bind_group_layouts: &[Some(&rgba_bgl)],
            immediate_size: 0,
        });
        let rgba_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("synth_rgba"),
            layout: Some(&rgba_pl),
            module: &rgba_shader,
            entry_point: Some("synthesize_rgba"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Sparse RGBA synthesis pipeline
        let sparse_rgba_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("synth_sparse_rgba"),
            source: wgpu::ShaderSource::Wgsl(include_str!("synthesize_sparse_rgba.wgsl").into()),
        });
        let sparse_rgba_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("synth_sparse_rgba_bgl"),
            entries: &[
                bgl_uniform(0),
                bgl_storage_ro(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                bgl_storage_ro(3),
                bgl_storage_ro(4),
            ],
        });
        let sparse_rgba_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("synth_sparse_rgba_pl"),
            bind_group_layouts: &[Some(&sparse_rgba_bgl)],
            immediate_size: 0,
        });
        let sparse_rgba_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("synth_sparse_rgba"),
                layout: Some(&sparse_rgba_pl),
                module: &sparse_rgba_shader,
                entry_point: Some("synthesize_sparse_rgba"),
                compilation_options: Default::default(),
                cache: None,
            });

        eprintln!("GPU synthesis: {}", if use_f16 { "fp16" } else { "fp32" });

        Self {
            pipeline,
            bind_group_layout,
            rgba_pipeline,
            rgba_bgl,
            sparse_rgba_pipeline,
            sparse_rgba_bgl,
            use_f16,
        }
    }

    pub fn uses_f16(&self) -> bool {
        self.use_f16
    }

    /// Compute coefficients and synthesize to texture in a single GPU submission.
    ///
    /// Avoids the pipeline bubble of separate compute_on_gpu + rasterize_to_texture.
    pub fn compute_and_rasterize(
        &self,
        coeff_computer: &GpuCoeffComputer,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        raster: &Rasterizer,
        shape: Shape,
        evenodd: bool,
    ) -> (GpuCoeffData, wgpu::Texture) {
        let mut encoder = device.create_command_encoder(&Default::default());
        let gpu_data = coeff_computer.encode_compute(device, queue, raster, shape, &mut encoder);
        let texture = self.encode_rasterize_to_texture(
            device,
            queue,
            raster,
            &gpu_data,
            evenodd,
            &mut encoder,
        );
        queue.submit([encoder.finish()]);
        (gpu_data, texture)
    }

    /// Synthesize directly to an Rgba8Unorm storage texture with sRGB encoding.
    ///
    /// `evenodd` selects the even-odd fill rule; `false` uses the default nonzero rule.
    pub fn rasterize_to_texture(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        raster: &Rasterizer,
        gpu_data: &GpuCoeffData,
        evenodd: bool,
    ) -> wgpu::Texture {
        let mut encoder = device.create_command_encoder(&Default::default());
        let texture = self.encode_rasterize_to_texture(
            device,
            queue,
            raster,
            gpu_data,
            evenodd,
            &mut encoder,
        );
        queue.submit([encoder.finish()]);
        texture
    }

    /// Synthesize with a view transform applied in the shader.
    ///
    /// `view_scale`: ratio of coefficient zoom to display zoom (1.0 = identity).
    /// `view_offset`: pan delta in normalized coefficient coordinates (0.0 = identity).
    /// Synthesize with a view transform, outputting to a texture of the given
    /// display dimensions (which may differ from the rasterizer's grid size).
    #[allow(clippy::too_many_arguments)]
    pub fn rasterize_to_texture_with_view(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        raster: &Rasterizer,
        gpu_data: &GpuCoeffData,
        evenodd: bool,
        view_scale: f32,
        view_offset: (f32, f32),
        display_w: u32,
        display_h: u32,
    ) -> wgpu::Texture {
        let mut encoder = device.create_command_encoder(&Default::default());
        let texture = self.encode_rasterize_to_texture_view(
            device,
            queue,
            raster,
            gpu_data,
            evenodd,
            view_scale,
            view_offset,
            display_w,
            display_h,
            &mut encoder,
        );
        queue.submit([encoder.finish()]);
        texture
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_rasterize_to_texture_view(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        raster: &Rasterizer,
        gpu_data: &GpuCoeffData,
        evenodd: bool,
        view_scale: f32,
        view_offset: (f32, f32),
        display_w: u32,
        display_h: u32,
        encoder: &mut wgpu::CommandEncoder,
    ) -> wgpu::Texture {
        let w = display_w;
        let h = display_h;

        let params = SynthParams {
            width: w,
            height: h,
            max_j: raster.max_level(),
            wh: raster.grid_size(),
            area: gpu_data.area,
            fill_rule: if evenodd { 1 } else { 0 },
            view_scale,
            view_offset_x: view_offset.0,
            view_offset_y: view_offset.1,
            _pad: 0.0,
        };

        let params_buf = upload_buffer(
            device,
            queue,
            "synth_params",
            wgpu::BufferUsages::UNIFORM,
            bytemuck::bytes_of(&params),
        );
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("synth_output"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let texture_view = texture.create_view(&Default::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("synth_rgba_bg"),
            layout: &self.rgba_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gpu_data.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.rgba_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(w.div_ceil(16), h.div_ceil(16), 1);
        }
        texture
    }

    fn encode_rasterize_to_texture(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        raster: &Rasterizer,
        gpu_data: &GpuCoeffData,
        evenodd: bool,
        encoder: &mut wgpu::CommandEncoder,
    ) -> wgpu::Texture {
        let w = raster.width() as u32;
        let h = raster.height() as u32;
        self.encode_rasterize_to_texture_view(
            device,
            queue,
            raster,
            gpu_data,
            evenodd,
            1.0,
            (0.0, 0.0),
            w,
            h,
            encoder,
        )
    }

    /// Synthesize from a GPU-resident coefficient buffer (from `GpuCoeffComputer`).
    ///
    /// If this synthesizer uses f16, the coefficients are converted to f16
    /// before synthesis.
    /// Compute sparse coefficients and synthesize in one submit.
    pub fn compute_and_rasterize_sparse(
        &self,
        coeff_computer: &GpuCoeffComputer,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        raster: &Rasterizer,
        shape: Shape,
        evenodd: bool,
    ) -> (GpuSparseCoeffData, wgpu::Texture) {
        let mut encoder = device.create_command_encoder(&Default::default());
        let gpu_data =
            coeff_computer.encode_compute_sparse(device, queue, raster, shape, &mut encoder);
        let texture = self.encode_rasterize_sparse_to_texture(
            device,
            queue,
            raster,
            &gpu_data,
            evenodd,
            &mut encoder,
        );
        queue.submit([encoder.finish()]);
        (gpu_data, texture)
    }

    /// Synthesize sparse coefficients to texture.
    pub fn rasterize_sparse_to_texture(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        raster: &Rasterizer,
        gpu_data: &GpuSparseCoeffData,
        evenodd: bool,
    ) -> wgpu::Texture {
        let mut encoder = device.create_command_encoder(&Default::default());
        let texture = self.encode_rasterize_sparse_to_texture(
            device,
            queue,
            raster,
            gpu_data,
            evenodd,
            &mut encoder,
        );
        queue.submit([encoder.finish()]);
        texture
    }

    fn encode_rasterize_sparse_to_texture(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        raster: &Rasterizer,
        gpu_data: &GpuSparseCoeffData,
        evenodd: bool,
        encoder: &mut wgpu::CommandEncoder,
    ) -> wgpu::Texture {
        let w = raster.width() as u32;
        let h = raster.height() as u32;
        let params = SynthParams {
            width: w,
            height: h,
            max_j: raster.max_level(),
            wh: raster.grid_size(),
            area: gpu_data.area,
            fill_rule: if evenodd { 1 } else { 0 },
            view_scale: 1.0,
            view_offset_x: 0.0,
            view_offset_y: 0.0,
            _pad: 0.0,
        };
        let params_buf = upload_buffer(
            device,
            queue,
            "synth_sparse_params",
            wgpu::BufferUsages::UNIFORM,
            bytemuck::bytes_of(&params),
        );
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("synth_sparse_output"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let texture_view = texture.create_view(&Default::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("synth_sparse_rgba_bg"),
            layout: &self.sparse_rgba_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gpu_data.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: gpu_data.level_info_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: gpu_data.cell_indices_buf.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.sparse_rgba_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(w.div_ceil(16), h.div_ceil(16), 1);
        }
        texture
    }

    pub fn rasterize_gpu(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        raster: &Rasterizer,
        gpu_data: &GpuCoeffData,
    ) -> Vec<f32> {
        if self.use_f16 {
            let f16_buf = Self::convert_f32_to_f16_gpu(device, queue, gpu_data);
            self.rasterize_inner(device, queue, raster, gpu_data.area, &f16_buf)
        } else {
            self.rasterize_inner(device, queue, raster, gpu_data.area, &gpu_data.buffer)
        }
    }

    /// Synthesize from CPU-side f32 coefficient data.
    pub fn rasterize_f32(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        raster: &Rasterizer,
        data: &CoeffData<Coeffs>,
    ) -> Vec<f32> {
        assert!(!self.use_f16, "f32 data passed to f16 pipeline");
        let buf = upload_buffer(
            device,
            queue,
            "coeffs_f32",
            wgpu::BufferUsages::STORAGE,
            bytemuck::cast_slice(&data.coeffs),
        );
        self.rasterize_inner(device, queue, raster, data.area, &buf)
    }

    /// Synthesize from CPU-side f16 coefficient data.
    pub fn rasterize_f16(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        raster: &Rasterizer,
        data: &CoeffData<CoeffsF16>,
    ) -> Vec<f32> {
        assert!(self.use_f16, "f16 data passed to f32 pipeline");
        let buf = upload_buffer(
            device,
            queue,
            "coeffs_f16",
            wgpu::BufferUsages::STORAGE,
            bytemuck::cast_slice(&data.coeffs),
        );
        self.rasterize_inner(device, queue, raster, data.area, &buf)
    }

    fn rasterize_inner(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        raster: &Rasterizer,
        area: f32,
        coeffs_buf: &wgpu::Buffer,
    ) -> Vec<f32> {
        let w = raster.width() as u32;
        let h = raster.height() as u32;
        let pixel_count = (w * h) as usize;

        let params = SynthParams {
            width: w,
            height: h,
            max_j: raster.max_level(),
            wh: raster.grid_size(),
            area,
            fill_rule: 0,
            view_scale: 1.0,
            view_offset_x: 0.0,
            view_offset_y: 0.0,
            _pad: 0.0,
        };

        let params_buf = upload_buffer(
            device,
            queue,
            "synth_params",
            wgpu::BufferUsages::UNIFORM,
            bytemuck::bytes_of(&params),
        );

        let pixel_size = (pixel_count * 4) as u64;
        let pixel_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pixels"),
            size: pixel_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("synthesize_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: coeffs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: pixel_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((pixel_count as u32).div_ceil(256), 1, 1);
        }
        queue.submit([encoder.finish()]);

        let bytes = readback_bytes(device, queue, &pixel_buf, pixel_size);
        bytemuck::cast_slice(&bytes).to_vec()
    }

    /// Convert f32 coefficient buffer to f16 on GPU.
    ///
    /// Reads 3×f32 per cell, writes vec4<f16> (with padding) per cell.
    fn convert_f32_to_f16_gpu(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        gpu_data: &GpuCoeffData,
    ) -> wgpu::Buffer {
        // Read back f32, convert to f16 on CPU, re-upload.
        let total = gpu_data.total_cells;
        let src_size = (total * 3 * 4) as u64;

        let bytes = readback_bytes(device, queue, &gpu_data.buffer, src_size);
        let raw: &[u32] = bytemuck::cast_slice(&bytes);

        let mut packed = Vec::with_capacity(total * 8);
        for i in 0..total {
            let c01 = f32::from_bits(raw[i * 3]) as f16;
            let c10 = f32::from_bits(raw[i * 3 + 1]) as f16;
            let c11 = f32::from_bits(raw[i * 3 + 2]) as f16;
            packed.extend_from_slice(&c01.to_le_bytes());
            packed.extend_from_slice(&c10.to_le_bytes());
            packed.extend_from_slice(&c11.to_le_bytes());
            packed.extend_from_slice(&0u16.to_le_bytes());
        }

        upload_buffer(
            device,
            queue,
            "coeffs_f16_converted",
            wgpu::BufferUsages::STORAGE,
            &packed,
        )
    }
}

// ---- GPU Coefficient Computer ----

/// GPU coefficient computation via per-segment bisection solvers.
pub struct GpuCoeffComputer {
    line_pipeline: wgpu::ComputePipeline,
    quad_pipeline: wgpu::ComputePipeline,
    cubic_pipeline: wgpu::ComputePipeline,
    arc_pipeline: wgpu::ComputePipeline,
    se_pipeline: wgpu::ComputePipeline,
    segment_bgl: wgpu::BindGroupLayout,
    sparse_line_pipeline: wgpu::ComputePipeline,
    sparse_quad_pipeline: wgpu::ComputePipeline,
    sparse_cubic_pipeline: wgpu::ComputePipeline,
    sparse_arc_pipeline: wgpu::ComputePipeline,
    sparse_se_pipeline: wgpu::ComputePipeline,
    sparse_bgl: wgpu::BindGroupLayout,
}

impl GpuCoeffComputer {
    pub fn new(device: &wgpu::Device) -> Self {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("coeff_bgl"),
            entries: &[bgl_uniform(0), bgl_storage_ro(1), bgl_storage_rw(2)],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("coeff_pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let seg_shader_src = include_str!("compute_coeffs.wgsl");

        let make = |label: &str, entry: &str| {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(seg_shader_src.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let line_pipeline = make("compute_coeffs", "compute_lines");
        let quad_pipeline = make("compute_coeffs_q", "compute_quads");
        let cubic_pipeline = make("compute_coeffs_c", "compute_cubics");
        let arc_pipeline = make("compute_coeffs_a", "compute_arcs");
        let se_pipeline = make("compute_coeffs_se", "compute_superellipses");

        let sparse_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("coeff_sparse_bgl"),
            entries: &[
                bgl_uniform(0),
                bgl_storage_ro(1),
                bgl_storage_rw(2),
                bgl_storage_ro(3),
                bgl_storage_ro(4),
            ],
        });
        let sparse_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("coeff_sparse_pl"),
            bind_group_layouts: &[Some(&sparse_bgl)],
            immediate_size: 0,
        });
        let make_sparse = |label: &str, entry: &str| {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(seg_shader_src.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&sparse_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let sparse_line_pipeline = make_sparse("sc_l", "compute_lines_sparse");
        let sparse_quad_pipeline = make_sparse("sc_q", "compute_quads_sparse");
        let sparse_cubic_pipeline = make_sparse("sc_c", "compute_cubics_sparse");
        let sparse_arc_pipeline = make_sparse("sc_a", "compute_arcs_sparse");
        let sparse_se_pipeline = make_sparse("sc_se", "compute_superellipses_sparse");

        Self {
            line_pipeline,
            quad_pipeline,
            cubic_pipeline,
            arc_pipeline,
            se_pipeline,
            segment_bgl: bgl,
            sparse_line_pipeline,
            sparse_quad_pipeline,
            sparse_cubic_pipeline,
            sparse_arc_pipeline,
            sparse_se_pipeline,
            sparse_bgl,
        }
    }

    /// Compute coefficients on GPU and submit immediately.
    pub fn compute_on_gpu(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        raster: &Rasterizer,
        shape: Shape,
    ) -> GpuCoeffData {
        let mut encoder = device.create_command_encoder(&Default::default());
        let data = self.encode_compute(device, queue, raster, shape, &mut encoder);
        queue.submit([encoder.finish()]);
        data
    }

    /// Encode coefficient computation into an existing command encoder.
    ///
    /// The caller is responsible for submitting the encoder. This allows
    /// batching coefficient computation with synthesis in a single submit.
    fn encode_compute(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        raster: &Rasterizer,
        shape: Shape,
        encoder: &mut wgpu::CommandEncoder,
    ) -> GpuCoeffData {
        let wh = raster.grid_size() as f64;
        let inv = 1.0 / wh;

        let total = level_offset(raster.max_level() + 1);
        let coeff_size = (total * 3 * 4) as u64;

        let coeff_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("coeffs_atomic"),
            size: coeff_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.clear_buffer(&coeff_buf, 0, None);

        // Single pass: normalize + compute area + pack into GPU buffers.
        // Pre-count segment types to avoid reallocation.
        let (mut n_line, mut n_quad, mut n_cubic, mut n_arc, mut n_se) = (0, 0, 0, 0, 0);
        for seg in &shape.segments {
            match seg {
                Segment::Line(..) => n_line += 1,
                Segment::QuadBez(..) => n_quad += 1,
                Segment::CubicBez(..) => n_cubic += 1,
                Segment::CircularArc(..) => n_arc += 1,
                Segment::Superellipse(..) => n_se += 1,
            }
        }
        let mut lines: Vec<f32> = Vec::with_capacity(n_line * 4);
        let mut quads: Vec<f32> = Vec::with_capacity(n_quad * 8);
        let mut cubics: Vec<f32> = Vec::with_capacity(n_cubic * 8);
        let mut arcs: Vec<f32> = Vec::with_capacity(n_arc * 12);
        let mut superellipses: Vec<f32> = Vec::with_capacity(n_se * 12);
        let mut area = 0.0f64;
        let det = |ax: f64, ay: f64, bx: f64, by: f64| ax * by - ay * bx;

        for seg in &shape.segments {
            match seg {
                Segment::Line(Line { p0, p1 }) => {
                    let (x0, y0) = (p0.x * inv, p0.y * inv);
                    let (x1, y1) = (p1.x * inv, p1.y * inv);
                    area += 0.5 * det(x0, y0, x1, y1);
                    lines.extend_from_slice(&[x0 as f32, y0 as f32, x1 as f32, y1 as f32]);
                }
                Segment::QuadBez(QuadBez { p0, p1, p2 }) => {
                    let (x0, y0) = (p0.x * inv, p0.y * inv);
                    let (x1, y1) = (p1.x * inv, p1.y * inv);
                    let (x2, y2) = (p2.x * inv, p2.y * inv);
                    area += (1.0 / 3.0) * det(x0, y0, x1, y1)
                        + (1.0 / 3.0) * det(x1, y1, x2, y2)
                        + (1.0 / 6.0) * det(x0, y0, x2, y2);
                    quads.extend_from_slice(&[
                        x0 as f32, y0 as f32, x1 as f32, y1 as f32, x2 as f32, y2 as f32, 0.0, 0.0,
                    ]);
                }
                Segment::CubicBez(CubicBez { p0, p1, p2, p3 }) => {
                    let (x0, y0) = (p0.x * inv, p0.y * inv);
                    let (x1, y1) = (p1.x * inv, p1.y * inv);
                    let (x2, y2) = (p2.x * inv, p2.y * inv);
                    let (x3, y3) = (p3.x * inv, p3.y * inv);
                    area += 0.5
                        * (0.6 * det(x0, y0, x1, y1)
                            + 0.3 * det(x0, y0, x2, y2)
                            + 0.1 * det(x0, y0, x3, y3)
                            + 0.3 * det(x1, y1, x2, y2)
                            + 0.3 * det(x1, y1, x3, y3)
                            + 0.6 * det(x2, y2, x3, y3));
                    cubics.extend_from_slice(&[
                        x0 as f32, y0 as f32, x1 as f32, y1 as f32, x2 as f32, y2 as f32,
                        x3 as f32, y3 as f32,
                    ]);
                }
                Segment::CircularArc(CircularArc {
                    center,
                    radius,
                    theta0,
                    theta1,
                }) => {
                    let (cx, cy) = (center.x * inv, center.y * inv);
                    let r = *radius * inv;
                    let (sin0, cos0) = theta0.sin_cos();
                    let (sin1, cos1) = theta1.sin_cos();
                    area += 0.5
                        * (r * cx * (sin1 - sin0) - r * cy * (cos1 - cos0)
                            + r * r * (theta1 - theta0));
                    // Bbox in normalized coords
                    let bb = seg.bbox();
                    arcs.extend_from_slice(&[
                        cx as f32,
                        cy as f32,
                        r as f32,
                        0.0,
                        *theta0 as f32,
                        *theta1 as f32,
                        (bb.x0 * inv) as f32,
                        (bb.y0 * inv) as f32,
                        (bb.x1 * inv) as f32,
                        (bb.y1 * inv) as f32,
                        0.0,
                        0.0,
                    ]);
                }
                Segment::Superellipse(Superellipse {
                    center,
                    a,
                    b,
                    n,
                    quadrants,
                }) => {
                    let (cx, cy) = (center.x * inv, center.y * inv);
                    let sa = *a * inv;
                    let sb = *b * inv;
                    // Area: delegate to segment method (GL-16 quadrature, not hot path)
                    let mut norm_seg = seg.clone();
                    if let Segment::Superellipse(Superellipse {
                        center: c,
                        a: ra,
                        b: rb,
                        ..
                    }) = &mut norm_seg
                    {
                        c.x = cx;
                        c.y = cy;
                        *ra = sa;
                        *rb = sb;
                    }
                    area += norm_seg.area_contribution();
                    let bb = seg.bbox();
                    superellipses.extend_from_slice(&[
                        cx as f32,
                        cy as f32,
                        sa as f32,
                        sb as f32,
                        *n as f32,
                        *quadrants as f32,
                        (bb.x0 * inv) as f32,
                        (bb.y0 * inv) as f32,
                        (bb.x1 * inv) as f32,
                        (bb.y1 * inv) as f32,
                        0.0,
                        0.0,
                    ]);
                }
            }
        }
        let area = area as f32;

        let max_j = raster.max_level();

        #[derive(Clone, Copy)]
        #[repr(C)]
        struct SegParams {
            max_j: u32,
            num_segments: u32,
        }
        unsafe impl bytemuck::Pod for SegParams {}
        unsafe impl bytemuck::Zeroable for SegParams {}

        let dispatch_type = |encoder: &mut wgpu::CommandEncoder,
                             pipeline: &wgpu::ComputePipeline,
                             n: u32,
                             wg_size: u32,
                             data: &[u8]| {
            let p = SegParams {
                max_j,
                num_segments: n,
            };
            let pbuf = upload_buffer(
                device,
                queue,
                "seg_params",
                wgpu::BufferUsages::UNIFORM,
                bytemuck::bytes_of(&p),
            );
            let sbuf = upload_buffer(device, queue, "seg_data", wgpu::BufferUsages::STORAGE, data);
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("seg_bg"),
                layout: &self.segment_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pbuf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: sbuf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: coeff_buf.as_entire_binding(),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n.div_ceil(wg_size), 1, 1);
        };

        if !lines.is_empty() {
            dispatch_type(
                encoder,
                &self.line_pipeline,
                (lines.len() / 4) as u32,
                64,
                bytemuck::cast_slice(&lines),
            );
        }
        if !quads.is_empty() {
            dispatch_type(
                encoder,
                &self.quad_pipeline,
                (quads.len() / 8) as u32,
                64,
                bytemuck::cast_slice(&quads),
            );
        }
        if !cubics.is_empty() {
            dispatch_type(
                encoder,
                &self.cubic_pipeline,
                (cubics.len() / 8) as u32,
                256,
                bytemuck::cast_slice(&cubics),
            );
        }
        if !arcs.is_empty() {
            dispatch_type(
                encoder,
                &self.arc_pipeline,
                (arcs.len() / 12) as u32,
                64,
                bytemuck::cast_slice(&arcs),
            );
        }
        if !superellipses.is_empty() {
            dispatch_type(
                encoder,
                &self.se_pipeline,
                (superellipses.len() / 12) as u32,
                64,
                bytemuck::cast_slice(&superellipses),
            );
        }

        GpuCoeffData {
            area,
            buffer: coeff_buf,
            total_cells: total,
        }
    }

    /// Compute sparse coefficients on GPU and submit immediately.
    pub fn compute_on_gpu_sparse(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        raster: &Rasterizer,
        shape: Shape,
    ) -> GpuSparseCoeffData {
        let mut encoder = device.create_command_encoder(&Default::default());
        let data = self.encode_compute_sparse(device, queue, raster, shape, &mut encoder);
        queue.submit([encoder.finish()]);
        data
    }

    /// Encode sparse coefficient computation into an existing command encoder.
    fn encode_compute_sparse(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        raster: &Rasterizer,
        shape: Shape,
        encoder: &mut wgpu::CommandEncoder,
    ) -> GpuSparseCoeffData {
        let wh = raster.grid_size() as f64;
        let inv = 1.0 / wh;
        let max_j = raster.max_level();

        // ---- Normalize segments, compute area, pack GPU buffers ----
        // (Same as dense path — we need the normalized bboxes to compute active cells.)
        let (mut n_line, mut n_quad, mut n_cubic, mut n_arc, mut n_se) = (0, 0, 0, 0, 0);
        for seg in &shape.segments {
            match seg {
                Segment::Line(..) => n_line += 1,
                Segment::QuadBez(..) => n_quad += 1,
                Segment::CubicBez(..) => n_cubic += 1,
                Segment::CircularArc(..) => n_arc += 1,
                Segment::Superellipse(..) => n_se += 1,
            }
        }
        let mut lines: Vec<f32> = Vec::with_capacity(n_line * 4);
        let mut quads: Vec<f32> = Vec::with_capacity(n_quad * 8);
        let mut cubics: Vec<f32> = Vec::with_capacity(n_cubic * 8);
        let mut arcs: Vec<f32> = Vec::with_capacity(n_arc * 12);
        let mut superellipses: Vec<f32> = Vec::with_capacity(n_se * 12);
        let mut area = 0.0f64;
        let det = |ax: f64, ay: f64, bx: f64, by: f64| ax * by - ay * bx;

        // Collect normalized bounding boxes for active cell computation.
        let mut norm_bboxes: Vec<NormBbox> = Vec::with_capacity(shape.segments.len());

        for seg in &shape.segments {
            match seg {
                Segment::Line(Line { p0, p1 }) => {
                    let (x0, y0) = (p0.x * inv, p0.y * inv);
                    let (x1, y1) = (p1.x * inv, p1.y * inv);
                    area += 0.5 * det(x0, y0, x1, y1);
                    let (fx0, fy0, fx1, fy1) = (x0 as f32, y0 as f32, x1 as f32, y1 as f32);
                    lines.extend_from_slice(&[fx0, fy0, fx1, fy1]);
                    norm_bboxes.push(NormBbox {
                        x_min: fx0.min(fx1),
                        y_min: fy0.min(fy1),
                        x_max: fx0.max(fx1),
                        y_max: fy0.max(fy1),
                    });
                }
                Segment::QuadBez(QuadBez { p0, p1, p2 }) => {
                    let (x0, y0) = (p0.x * inv, p0.y * inv);
                    let (x1, y1) = (p1.x * inv, p1.y * inv);
                    let (x2, y2) = (p2.x * inv, p2.y * inv);
                    area += (1.0 / 3.0) * det(x0, y0, x1, y1)
                        + (1.0 / 3.0) * det(x1, y1, x2, y2)
                        + (1.0 / 6.0) * det(x0, y0, x2, y2);
                    let (fx0, fy0) = (x0 as f32, y0 as f32);
                    let (fx1, fy1) = (x1 as f32, y1 as f32);
                    let (fx2, fy2) = (x2 as f32, y2 as f32);
                    quads.extend_from_slice(&[fx0, fy0, fx1, fy1, fx2, fy2, 0.0, 0.0]);
                    norm_bboxes.push(NormBbox {
                        x_min: fx0.min(fx1).min(fx2),
                        y_min: fy0.min(fy1).min(fy2),
                        x_max: fx0.max(fx1).max(fx2),
                        y_max: fy0.max(fy1).max(fy2),
                    });
                }
                Segment::CubicBez(CubicBez { p0, p1, p2, p3 }) => {
                    let (x0, y0) = (p0.x * inv, p0.y * inv);
                    let (x1, y1) = (p1.x * inv, p1.y * inv);
                    let (x2, y2) = (p2.x * inv, p2.y * inv);
                    let (x3, y3) = (p3.x * inv, p3.y * inv);
                    area += 0.5
                        * (0.6 * det(x0, y0, x1, y1)
                            + 0.3 * det(x0, y0, x2, y2)
                            + 0.1 * det(x0, y0, x3, y3)
                            + 0.3 * det(x1, y1, x2, y2)
                            + 0.3 * det(x1, y1, x3, y3)
                            + 0.6 * det(x2, y2, x3, y3));
                    let (fx0, fy0) = (x0 as f32, y0 as f32);
                    let (fx1, fy1) = (x1 as f32, y1 as f32);
                    let (fx2, fy2) = (x2 as f32, y2 as f32);
                    let (fx3, fy3) = (x3 as f32, y3 as f32);
                    cubics.extend_from_slice(&[fx0, fy0, fx1, fy1, fx2, fy2, fx3, fy3]);
                    norm_bboxes.push(NormBbox {
                        x_min: fx0.min(fx1).min(fx2).min(fx3),
                        y_min: fy0.min(fy1).min(fy2).min(fy3),
                        x_max: fx0.max(fx1).max(fx2).max(fx3),
                        y_max: fy0.max(fy1).max(fy2).max(fy3),
                    });
                }
                Segment::CircularArc(CircularArc {
                    center,
                    radius,
                    theta0,
                    theta1,
                }) => {
                    let (cx, cy) = (center.x * inv, center.y * inv);
                    let r = *radius * inv;
                    let (sin0, cos0) = theta0.sin_cos();
                    let (sin1, cos1) = theta1.sin_cos();
                    area += 0.5
                        * (r * cx * (sin1 - sin0) - r * cy * (cos1 - cos0)
                            + r * r * (theta1 - theta0));
                    let bb = seg.bbox();
                    let nb = NormBbox {
                        x_min: (bb.x0 * inv) as f32,
                        y_min: (bb.y0 * inv) as f32,
                        x_max: (bb.x1 * inv) as f32,
                        y_max: (bb.y1 * inv) as f32,
                    };
                    arcs.extend_from_slice(&[
                        cx as f32,
                        cy as f32,
                        r as f32,
                        0.0,
                        *theta0 as f32,
                        *theta1 as f32,
                        nb.x_min,
                        nb.y_min,
                        nb.x_max,
                        nb.y_max,
                        0.0,
                        0.0,
                    ]);
                    norm_bboxes.push(nb);
                }
                Segment::Superellipse(Superellipse {
                    center,
                    a,
                    b,
                    n,
                    quadrants,
                }) => {
                    let (cx, cy) = (center.x * inv, center.y * inv);
                    let sa = *a * inv;
                    let sb = *b * inv;
                    let mut norm_seg = seg.clone();
                    if let Segment::Superellipse(Superellipse {
                        center: c,
                        a: ra,
                        b: rb,
                        ..
                    }) = &mut norm_seg
                    {
                        c.x = cx;
                        c.y = cy;
                        *ra = sa;
                        *rb = sb;
                    }
                    area += norm_seg.area_contribution();
                    let bb = seg.bbox();
                    let nb = NormBbox {
                        x_min: (bb.x0 * inv) as f32,
                        y_min: (bb.y0 * inv) as f32,
                        x_max: (bb.x1 * inv) as f32,
                        y_max: (bb.y1 * inv) as f32,
                    };
                    superellipses.extend_from_slice(&[
                        cx as f32,
                        cy as f32,
                        sa as f32,
                        sb as f32,
                        *n as f32,
                        *quadrants as f32,
                        nb.x_min,
                        nb.y_min,
                        nb.x_max,
                        nb.y_max,
                        0.0,
                        0.0,
                    ]);
                    norm_bboxes.push(nb);
                }
            }
        }
        let area = area as f32;

        // ---- Compute active cells per level ----
        let sparse_map = compute_active_cells(max_j, &norm_bboxes);
        let n_active = sparse_map.n_active;

        // Note: caller can access n_active via the returned GpuSparseCoeffData.

        // ---- Create GPU buffers ----
        let coeff_size = (n_active.max(1) * 3 * 4) as u64;
        let coeff_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sparse_coeffs_atomic"),
            size: coeff_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.clear_buffer(&coeff_buf, 0, None);

        // Level info buffer: array of (start: u32, count: u32) per level
        let mut level_info_data: Vec<u32> = sparse_map
            .level_info
            .iter()
            .flat_map(|&(s, c)| [s, c])
            .collect();
        // Ensure non-empty (wgpu requires buffers to have size > 0)
        if level_info_data.is_empty() {
            level_info_data.push(0);
            level_info_data.push(0);
        }
        let level_info_buf = upload_buffer(
            device,
            queue,
            "sparse_level_info",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            bytemuck::cast_slice(&level_info_data),
        );

        // Cell indices buffer (ensure non-empty)
        let cell_indices_data = if sparse_map.cell_indices.is_empty() {
            &[0u32][..]
        } else {
            &sparse_map.cell_indices[..]
        };
        let cell_indices_buf = upload_buffer(
            device,
            queue,
            "sparse_cell_indices",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            bytemuck::cast_slice(cell_indices_data),
        );

        // ---- Dispatch sparse compute passes ----
        #[derive(Clone, Copy)]
        #[repr(C)]
        struct SegParams {
            max_j: u32,
            num_segments: u32,
        }
        unsafe impl bytemuck::Pod for SegParams {}
        unsafe impl bytemuck::Zeroable for SegParams {}

        let dispatch_sparse = |encoder: &mut wgpu::CommandEncoder,
                               pipeline: &wgpu::ComputePipeline,
                               n: u32,
                               wg_size: u32,
                               data: &[u8]| {
            let p = SegParams {
                max_j,
                num_segments: n,
            };
            let pbuf = upload_buffer(
                device,
                queue,
                "seg_params",
                wgpu::BufferUsages::UNIFORM,
                bytemuck::bytes_of(&p),
            );
            let sbuf = upload_buffer(device, queue, "seg_data", wgpu::BufferUsages::STORAGE, data);
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("seg_sparse_bg"),
                layout: &self.sparse_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pbuf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: sbuf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: coeff_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: level_info_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: cell_indices_buf.as_entire_binding(),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n.div_ceil(wg_size), 1, 1);
        };

        if !lines.is_empty() {
            dispatch_sparse(
                encoder,
                &self.sparse_line_pipeline,
                (lines.len() / 4) as u32,
                64,
                bytemuck::cast_slice(&lines),
            );
        }
        if !quads.is_empty() {
            dispatch_sparse(
                encoder,
                &self.sparse_quad_pipeline,
                (quads.len() / 8) as u32,
                64,
                bytemuck::cast_slice(&quads),
            );
        }
        if !cubics.is_empty() {
            dispatch_sparse(
                encoder,
                &self.sparse_cubic_pipeline,
                (cubics.len() / 8) as u32,
                256,
                bytemuck::cast_slice(&cubics),
            );
        }
        if !arcs.is_empty() {
            dispatch_sparse(
                encoder,
                &self.sparse_arc_pipeline,
                (arcs.len() / 12) as u32,
                64,
                bytemuck::cast_slice(&arcs),
            );
        }
        if !superellipses.is_empty() {
            dispatch_sparse(
                encoder,
                &self.sparse_se_pipeline,
                (superellipses.len() / 12) as u32,
                64,
                bytemuck::cast_slice(&superellipses),
            );
        }

        GpuSparseCoeffData {
            area,
            buffer: coeff_buf,
            n_active,
            level_info_buf,
            cell_indices_buf,
            max_j,
        }
    }
}

/// Normalized bounding box (in [0,1] coordinate space).
struct NormBbox {
    x_min: f32,
    y_min: f32,
    x_max: f32,
    y_max: f32,
}

/// Compute the set of active cells for each wavelet level.
///
/// Collects all (level, cell_key) pairs from segment bounding boxes, sorts,
/// and deduplicates to produce a compact index per level.
fn compute_active_cells(max_j: u32, norm_bboxes: &[NormBbox]) -> SparseIndexMap {
    let n_levels = (max_j + 1) as usize;

    // Collect all (level, cell_key) pairs
    let mut pairs: Vec<(u32, u32)> = Vec::new();
    for bb in norm_bboxes {
        for j in 0..=max_j {
            let cells = 1u32 << j;
            let cellsf = cells as f32;
            let kx_lo = (bb.x_min * cellsf).floor().max(0.0) as u32;
            let kx_hi = ((bb.x_max * cellsf).ceil() as u32).min(cells);
            let ky_lo = (bb.y_min * cellsf).floor().max(0.0) as u32;
            let ky_hi = ((bb.y_max * cellsf).ceil() as u32).min(cells);
            for kx in kx_lo..kx_hi {
                for ky in ky_lo..ky_hi {
                    pairs.push((j, kx * cells + ky));
                }
            }
        }
    }

    // Sort by (level, cell_key) and dedup
    pairs.sort_unstable();
    pairs.dedup();

    // Split into per-level sorted arrays
    let mut cell_indices: Vec<u32> = Vec::with_capacity(pairs.len());
    let mut level_info: Vec<(u32, u32)> = Vec::with_capacity(n_levels);
    let mut pair_idx = 0;
    for j in 0..n_levels {
        let start = cell_indices.len() as u32;
        while pair_idx < pairs.len() && pairs[pair_idx].0 == j as u32 {
            cell_indices.push(pairs[pair_idx].1);
            pair_idx += 1;
        }
        let count = cell_indices.len() as u32 - start;
        level_info.push((start, count));
    }

    let n_active = cell_indices.len();
    SparseIndexMap {
        level_info,
        cell_indices,
        n_active,
    }
}

// ---- Helpers ----

fn bgl_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn upload_buffer(
    device: &wgpu::Device,
    _queue: &wgpu::Queue,
    label: &str,
    usage: wgpu::BufferUsages,
    data: &[u8],
) -> wgpu::Buffer {
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: data.len() as u64,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });
    buf.slice(..).get_mapped_range_mut().copy_from_slice(data);
    buf.unmap();
    buf
}

fn readback_bytes(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    size: u64,
) -> Vec<u8> {
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(src, 0, &readback, 0, size);
    queue.submit([encoder.finish()]);

    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).ok();

    let data = slice.get_mapped_range();
    let result = data.to_vec();
    drop(data);
    readback.unmap();
    result
}
