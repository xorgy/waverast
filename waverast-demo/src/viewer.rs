//! Interactive viewer with zoom/pan. Re-rasterizes via GPU on each view change.
//!
//! `--async-coeff` mode computes coefficients for the full shape at the current
//! zoom level (no pan baked in). Panning only re-synthesizes with a view offset,
//! avoiding recomputation. Zoom changes trigger recomputation.

use std::sync::Arc;
use std::time::Instant;

use waverast::contour::{
    CircularArc, CubicBez, Line, Point, QuadBez, Segment, Shape, Superellipse,
};
use waverast::gpu::{GpuCoeffComputer, GpuCoeffData, GpuSynthesizer};
use waverast::rasterizer::Rasterizer;
use winit::application::ApplicationHandler;
use winit::event::{MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

#[path = "svg_path.rs"]
mod svg_path;

struct ViewerApp {
    segments: Vec<Segment>,
    w: usize,
    h: usize,
    force_f32: bool,
    evenodd: bool,
    sparse: bool,
    async_coeff: bool,
    zoom: f64,
    pan_x: f64,
    pan_y: f64,
    dragging: bool,
    last_mouse: (f64, f64),
    state: Option<ViewerState>,
    dirty: bool,
    // Async mode: cached coefficients (computed at a specific zoom, no pan)
    display_coeffs: Option<DisplayCoeffs>,
    coeff_stale: bool,
}

/// Coefficients for the full shape at a specific zoom level.
/// Pan is NOT baked in — it's applied in the synthesis view transform.
struct DisplayCoeffs {
    data: GpuCoeffData,
    raster: Rasterizer,
    zoom: f64,
}

#[allow(dead_code)]
struct ViewerState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_config: wgpu::SurfaceConfiguration,
    gpu_coeff: GpuCoeffComputer,
    gpu_synth: GpuSynthesizer,
    blit_pipeline: wgpu::RenderPipeline,
    blit_bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    blit_bg: wgpu::BindGroup,
}

impl ViewerApp {
    fn render_frame(&mut self) {
        let t0 = Instant::now();

        if !self.async_coeff {
            let state = self.state.as_ref().unwrap();
            let _ = state.device.poll(wgpu::PollType::wait_indefinitely());
        }

        let texture = if self.async_coeff {
            self.render_async()
        } else {
            self.render_sync()
        };

        // Blit + present
        let texture_view = texture.create_view(&Default::default());
        let state = self.state.as_mut().unwrap();
        state.blit_bg = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit_bg"),
            layout: &state.blit_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&state.sampler),
                },
            ],
        });
        let frame = match state.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(tex)
            | wgpu::CurrentSurfaceTexture::Suboptimal(tex) => tex,
            _ => {
                self.dirty = false;
                return;
            }
        };
        let view = frame.texture.create_view(&Default::default());
        let mut encoder = state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("blit"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                ..Default::default()
            });
            pass.set_pipeline(&state.blit_pipeline);
            pass.set_bind_group(0, &state.blit_bg, &[]);
            pass.draw(0..6, 0..1);
        }
        state.queue.submit([encoder.finish()]);

        if !self.async_coeff {
            let _ = state.device.poll(wgpu::PollType::wait_indefinitely());
        }
        frame.present();

        // After present: in async mode, recompute coefficients if zoom changed
        if self.async_coeff && self.coeff_stale {
            self.recompute_coeffs();
        }

        let elapsed = t0.elapsed();
        eprintln!(
            "Frame: {:>7.1?}  zoom={:.2}x  {}x{}{}",
            elapsed,
            self.zoom,
            self.w,
            self.h,
            if self.async_coeff && self.coeff_stale {
                " (stale)"
            } else {
                ""
            },
        );
        self.dirty = false;
    }

    fn render_sync(&mut self) -> wgpu::Texture {
        let state = self.state.as_mut().unwrap();
        let transformed: Vec<_> = self
            .segments
            .iter()
            .map(|seg| transform_segment(seg, self.zoom, self.pan_x, self.pan_y))
            .collect();
        let shape = Shape::new(transformed);
        let raster = Rasterizer::for_size(self.w, self.h);
        if self.sparse {
            state
                .gpu_synth
                .compute_and_rasterize_sparse(
                    &state.gpu_coeff,
                    &state.device,
                    &state.queue,
                    &raster,
                    shape,
                    self.evenodd,
                )
                .1
        } else {
            state
                .gpu_synth
                .compute_and_rasterize(
                    &state.gpu_coeff,
                    &state.device,
                    &state.queue,
                    &raster,
                    shape,
                    self.evenodd,
                )
                .1
        }
    }

    fn render_async(&mut self) -> wgpu::Texture {
        let state = self.state.as_mut().unwrap();
        if let Some(ref dc) = self.display_coeffs {
            // Coefficients cover the full shape scaled by dc.zoom (no pan).
            // Map display pixel (col, row) to coefficient-space normalized coord:
            //   shape_point = col / self.zoom + self.pan_x
            //   coeff_pixel = shape_point * dc.zoom
            //   normalized  = coeff_pixel / wh
            //               = col * (dc.zoom / self.zoom) / wh + self.pan_x * dc.zoom / wh
            let view_scale = (dc.zoom / self.zoom) as f32;
            let wh = dc.raster.grid_size() as f64;
            let view_offset_x = (self.pan_x * dc.zoom / wh) as f32;
            let view_offset_y = (self.pan_y * dc.zoom / wh) as f32;
            state.gpu_synth.rasterize_to_texture_with_view(
                &state.device,
                &state.queue,
                &dc.raster,
                &dc.data,
                self.evenodd,
                view_scale,
                (view_offset_x, view_offset_y),
                self.w as u32,
                self.h as u32,
            )
        } else {
            // First frame: compute coefficients synchronously
            let dc = self.compute_coeffs_for_zoom(self.zoom);
            let state = self.state.as_ref().unwrap();
            let tex = state.gpu_synth.rasterize_to_texture_with_view(
                &state.device,
                &state.queue,
                &dc.raster,
                &dc.data,
                self.evenodd,
                1.0,
                (0.0, 0.0),
                self.w as u32,
                self.h as u32,
            );
            self.display_coeffs = Some(dc);
            self.coeff_stale = false;
            tex
        }
    }

    /// Compute coefficients for the full shape at a given zoom level (blocking).
    fn compute_coeffs_for_zoom(&self, zoom: f64) -> DisplayCoeffs {
        let state = self.state.as_ref().unwrap();
        let scaled: Vec<_> = self
            .segments
            .iter()
            .map(|seg| scale_segment(seg, zoom))
            .collect();
        let shape = Shape::new(scaled);
        let raster = rasterizer_for_shape(&shape, self.w, self.h);
        let data = state
            .gpu_coeff
            .compute_on_gpu(&state.device, &state.queue, &raster, shape);
        let _ = state.device.poll(wgpu::PollType::wait_indefinitely());
        DisplayCoeffs { data, raster, zoom }
    }

    /// Recompute coefficients for the current zoom level (blocking, after present).
    fn recompute_coeffs(&mut self) {
        let dc = self.compute_coeffs_for_zoom(self.zoom);
        self.display_coeffs = Some(dc);
        self.coeff_stale = false;
        // Trigger a redraw with fresh coefficients
        self.dirty = true;
        self.state.as_ref().unwrap().window.request_redraw();
    }
}

impl ApplicationHandler for ViewerApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("waverast viewer")
                        .with_inner_size(winit::dpi::PhysicalSize::new(
                            self.w as u32,
                            self.h as u32,
                        )),
                )
                .expect("failed to create window"),
        );

        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let surface = instance
            .create_surface(window.clone())
            .expect("failed to create surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .expect("no suitable GPU adapter");

        eprintln!("GPU: {}", adapter.get_info().name);

        let want_f16 = !self.force_f32 && adapter.features().contains(wgpu::Features::SHADER_F16);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("waverast"),
            required_features: if want_f16 {
                wgpu::Features::SHADER_F16
            } else {
                wgpu::Features::empty()
            },
            required_limits: wgpu::Limits {
                max_storage_buffer_binding_size: adapter.limits().max_storage_buffer_binding_size,
                max_buffer_size: adapter.limits().max_buffer_size,
                ..Default::default()
            },
            ..Default::default()
        }))
        .expect("failed to get device");

        let gpu_synth = GpuSynthesizer::new(&device, self.force_f32);
        let gpu_coeff = GpuCoeffComputer::new(&device);

        let mut surface_config = surface
            .get_default_config(&adapter, self.w as u32, self.h as u32)
            .expect("surface not supported");
        surface_config.format = match surface_config.format {
            wgpu::TextureFormat::Bgra8UnormSrgb => wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Rgba8UnormSrgb => wgpu::TextureFormat::Rgba8Unorm,
            f => f,
        };
        let caps = surface.get_capabilities(&adapter);
        if caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
            surface_config.present_mode = wgpu::PresentMode::Mailbox;
        }
        surface.configure(&device, &surface_config);

        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit"),
            source: wgpu::ShaderSource::Wgsl(BLIT_SHADER.into()),
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());
        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let placeholder_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("placeholder"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let placeholder_view = placeholder_tex.create_view(&Default::default());
        let blit_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit_bg"),
            layout: &blit_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&placeholder_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let blit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blit_pl"),
            bind_group_layouts: &[Some(&blit_bgl)],
            immediate_size: 0,
        });
        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit"),
            layout: Some(&blit_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        self.state = Some(ViewerState {
            window,
            surface,
            device,
            queue,
            surface_config,
            gpu_coeff,
            gpu_synth,
            blit_pipeline,
            blit_bgl,
            sampler,
            blit_bg,
        });

        self.dirty = true;
        self.coeff_stale = true;
        self.render_frame();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = &self.state else { return };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                let new_w = size.width.max(1) as usize;
                let new_h = size.height.max(1) as usize;
                if new_w != self.w || new_h != self.h {
                    self.w = new_w;
                    self.h = new_h;
                    let state = self.state.as_mut().unwrap();
                    state.surface_config.width = new_w as u32;
                    state.surface_config.height = new_h as u32;
                    state
                        .surface
                        .configure(&state.device, &state.surface_config);
                    self.dirty = true;
                    self.coeff_stale = true;
                    self.state.as_ref().unwrap().window.request_redraw();
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let dy = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y as f64,
                    MouseScrollDelta::PixelDelta(pos) => pos.y / 40.0,
                };
                let factor = (1.0 + dy * 0.1).max(0.1);
                let (mx, my) = self.last_mouse;
                let shape_x = mx / self.zoom + self.pan_x;
                let shape_y = my / self.zoom + self.pan_y;
                self.zoom *= factor;
                self.pan_x = shape_x - mx / self.zoom;
                self.pan_y = shape_y - my / self.zoom;
                self.dirty = true;
                self.coeff_stale = true; // zoom changed
                state.window.request_redraw();
            }

            WindowEvent::MouseInput {
                state: button_state,
                button: MouseButton::Left,
                ..
            } => {
                self.dragging = button_state.is_pressed();
            }

            WindowEvent::CursorMoved { position, .. } => {
                let (mx, my) = (position.x, position.y);
                if self.dragging {
                    let dx = mx - self.last_mouse.0;
                    let dy = my - self.last_mouse.1;
                    self.pan_x -= dx / self.zoom;
                    self.pan_y -= dy / self.zoom;
                    self.dirty = true;
                    // Pan only — no coeff recomputation needed
                    state.window.request_redraw();
                }
                self.last_mouse = (mx, my);
            }

            WindowEvent::RedrawRequested if self.dirty => {
                self.render_frame();
            }
            _ => {}
        }
    }
}

/// Create a rasterizer whose grid covers the full shape extent.
fn rasterizer_for_shape(shape: &Shape, win_w: usize, win_h: usize) -> Rasterizer {
    let mut max_extent = win_w.max(win_h) as f64;
    for seg in &shape.segments {
        let bb = seg.bbox();
        max_extent = max_extent.max(bb.x1.ceil()).max(bb.y1.ceil());
    }
    Rasterizer::for_size(max_extent.ceil() as usize, max_extent.ceil() as usize)
}

/// Scale a segment by zoom only (no pan).
fn scale_segment(seg: &Segment, zoom: f64) -> Segment {
    transform_segment(seg, zoom, 0.0, 0.0)
}

/// Apply view transform: screen = (shape - pan) * zoom
fn transform_segment(seg: &Segment, zoom: f64, pan_x: f64, pan_y: f64) -> Segment {
    let tp = |p: &Point| Point::new((p.x - pan_x) * zoom, (p.y - pan_y) * zoom);
    match seg {
        Segment::Line(Line { p0, p1 }) => Segment::Line(Line::new(tp(p0), tp(p1))),
        Segment::QuadBez(QuadBez { p0, p1, p2 }) => {
            Segment::QuadBez(QuadBez::new(tp(p0), tp(p1), tp(p2)))
        }
        Segment::CubicBez(CubicBez { p0, p1, p2, p3 }) => {
            Segment::CubicBez(CubicBez::new(tp(p0), tp(p1), tp(p2), tp(p3)))
        }
        Segment::CircularArc(CircularArc {
            center,
            radius,
            theta0,
            theta1,
        }) => Segment::CircularArc(CircularArc {
            center: tp(center),
            radius: *radius * zoom,
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
            center: tp(center),
            a: *a * zoom,
            b: *b * zoom,
            n: *n,
            quadrants: *quadrants,
        }),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "Usage: viewer <path-file> <width> <height> [--f32] [--evenodd] [--sparse] [--async-coeff]"
        );
        eprintln!("  --f32          Force f32 synthesis (disable f16)");
        eprintln!("  --evenodd      Use even-odd fill rule (default: nonzero)");
        eprintln!("  --sparse       Use sparse wavelet coefficients");
        eprintln!("  --async-coeff  Pan via synthesis view offset (no recomputation)");
        eprintln!();
        eprintln!("Controls:");
        eprintln!("  Mouse wheel  Zoom (toward cursor)");
        eprintln!("  Left drag    Pan");
        std::process::exit(1);
    }

    let path_file = &args[1];
    let w: usize = args[2].parse().expect("invalid width");
    let h: usize = args[3].parse().expect("invalid height");
    let force_f32 = args.iter().any(|a| a == "--f32");
    let evenodd = args.iter().any(|a| a == "--evenodd");
    let sparse = args.iter().any(|a| a == "--sparse");
    let async_coeff = args.iter().any(|a| a == "--async-coeff");

    let t0 = Instant::now();
    let path_data = std::fs::read_to_string(path_file).expect("failed to read path file");
    let segments = svg_path::parse_svg_path(&path_data);
    eprintln!("Parsed {} segments in {:?}", segments.len(), t0.elapsed());

    let event_loop = EventLoop::new().expect("failed to create event loop");
    let mut app = ViewerApp {
        segments,
        w,
        h,
        force_f32,
        evenodd,
        sparse,
        async_coeff,
        zoom: 1.0,
        pan_x: 0.0,
        pan_y: 0.0,
        dragging: false,
        last_mouse: (0.0, 0.0),
        state: None,
        dirty: true,
        display_coeffs: None,
        coeff_stale: true,
    };
    event_loop.run_app(&mut app).expect("event loop error");
}

const BLIT_SHADER: &str = "
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    let positions = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
        vec2(1.0, -1.0),  vec2(1.0,  1.0), vec2(-1.0, 1.0),
    );
    let uvs = array<vec2<f32>, 6>(
        vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(0.0, 0.0),
        vec2(1.0, 1.0), vec2(1.0, 0.0), vec2(0.0, 0.0),
    );
    var out: VsOut;
    out.pos = vec4(positions[vi], 0.0, 1.0);
    out.uv = uvs[vi];
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(tex, samp, in.uv);
}
";
